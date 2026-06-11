# Quantum Engine

Quantum Engine is an **open-core C++ vectorized execution engine** designed to replace Apache Spark's JVM core with a high-performance native implementation. Inspired by proprietary solutions like Databricks Photon, Quantum Engine brings bare-metal, SIMD-accelerated execution to the open-source big-data ecosystem — without vendor lock-in.

> **Status:** Phases 1–5 complete. The full classical SQL surface — `SELECT … FROM A [JOIN B [JOIN C…]] WHERE … GROUP BY … HAVING … ORDER BY … LIMIT …` — translates from Spark Catalyst to a native protobuf plan IR, runs on the C++ engine over in-memory **and** Parquet (single-file, multi-file, Hive-partitioned) **and** Delta sources, and returns rows back to Spark.

See [`quantum-engine-architecture.md`](./quantum-engine-architecture.md) for the full seven-layer design.

---

## What's Implemented

**Phase 1 — Foundation**
- **Columnar data format** — `ColumnBatch`, `ColumnVector`, `ValidityBitmask`, and a `PositionList` with an identity fast path so filters update indices instead of compacting data.
- **Off-heap memory manager** — arena-backed allocator with per-query budgets and RAII `Buffer`s, 64-byte aligned for cache-line / AVX use.
- **Expression engine** — `Literal`, `ColumnRef`, `BinaryExpr` over `Int32 / Int64 / Float64 / Bool` with arithmetic (+ − ×) and comparison (= < >) kernels and null propagation.
- **Pull-based operators** — `ScanOperator`, `FilterOperator`, `ProjectOperator` implementing `open() / getNext() / close()`.

**Phase 2 — Complete**
- **`HashAggregateOperator`** — `GROUP BY` over **any number** of fixed-width key columns with `SUM`, `COUNT(*)`, `MIN`, `MAX`, `AVG` aggregates. Each key column's validity is encoded into the composite key bytes so `(NULL, 1)` and `(NULL, 2)` form distinct SQL groups.
- **`HashJoinOperator`** — `Inner` and `LeftOuter` equi-join on any number of fixed-width key columns. Build side is densified into a row store and chained via a `next[]` array indexed by `BytesHashTable`; probe side streams batch-by-batch. NULLs in any key column never match.
- **Hash infrastructure** — `HashTable` (int64-keyed, used by tests as a perf baseline) and `BytesHashTable` (opaque fixed-width byte-string key, used by both Hash* operators). `KeyCodec` encodes composite group/join keys with validity bits inline.

**Phase 3 — Complete**
- **`STRING` / variable-width values** — `ColumnVector` gains a second `offsets` buffer (Arrow-compatible layout). `StringBuilder` constructs string columns row-by-row. Strings work as values throughout: scan → filter → project → aggregate-pass-through → join build/probe materialization.
- **String comparisons** — `BinaryExpr` `Eq` / `Lt` / `Gt` over `STRING` columns via lexicographic `std::string_view` compare; null propagation unchanged.
- **String GROUP BY / JOIN keys** — `KeyCodec` grows a variable-width encoding path with length-prefixed string columns and inline column-validity bits (so `("", "x")` and `(NULL, "x")` are distinct groups). New `VarBytesHashTable` stores out-of-line keys in a growing arena. `HashAggregate` / `HashJoin` route through it automatically when any key column is `STRING`. Null string keys never match in joins.
- **`RightOuter` and `FullOuter` joins** — `HashJoinOperator` tracks a build-side matched bitmap during probe and emits one trailing batch of unmatched build rows (with NULL probe columns) after the probe stream ends. NULL-keyed build rows naturally appear in the right tail since they were never insertable into the index.
- **`MinString` / `MaxString` aggregates** — lexicographic MIN/MAX over `STRING` columns, NULL-skipping, NULL for all-null groups. Per-group state is an `int32` index into an aggregate-owned `std::vector<std::string>`; finalize emits via `StringBuilder`.
- **`LikeExpr`** — SQL `LIKE` with `%` (any substring) and `_` (any single char). Anchored, iterative non-backtracking matcher. Null inputs propagate; pattern is a constant `std::string` for Phase 3.
- **Dictionary-encoded STRING columns** — `ColumnVector` grows a third layout: `values_` holds `int32` codes; `dictOffsets_` + `dictValues_` hold the shared dictionary. `DictStringBuilder` builds dict columns row-by-row with dedup. `stringAt()` dispatches uniformly so existing kernels (compare, LIKE, MIN/MAX) and GROUP BY/JOIN key codecs work unchanged. Dict layout is preserved through `clone` / `selectActive` / `HashJoin gather`, and through `HashJoin concatDense` when every build batch is dict-encoded (per-batch dictionaries are merged/deduped with a code remap — O(dict entries) hashing, not O(rows)); mixed dict/plain build batches still decode to regular STRING.
- **Dict-aware equality fast path** — `Literal(std::string)` now emits a dict-encoded broadcast (`dictSize=1`), and the STRING `Eq` kernel detects dict-vs-dict inputs, precomputes a `rhsCode → lhsCode` translation table, and compares int32s per row. Measured **~4× throughput** on `dict_size=10` vs the per-row `stringAt` path; gains shrink as dict size grows.
- **Dict-aware LIKE fast path** — `LikeExpr::eval` on a dict-encoded column evaluates the pattern matcher once per dictionary entry (into a `dictResult[dictSize]` table), then per row just indexes into that table by code. Measured **~3× throughput** at `dict_size=10`, **~3.6×** at `dict_size=100`, **~2.4×** at `dict_size=1000` — LIKE benefits more than `Eq` because per-row pattern matching is the expensive bit.
- **Code-aware GROUP BY (single STRING key)** — `HashAggregateOperator` adds a `StringFastPath` selected for single-STRING-key aggregates. Maintains a global dictionary across batches; for dict-encoded inputs per-row work shrinks to a `batchCode → globalCode` array lookup plus an array-indexed group state slot. Non-dict batches still intern per row into the same global dictionary so results are identical. Measured on 1M-row aggregation: **~4× throughput** at `dict_size=10` (128M vs 33M items/s), **~3.9×** at 100, **~1.6×** at 1000.
- **Spill-to-disk for HashAggregate** — `AggregateFunction::merge` added on the numeric aggregates; new `SpillFile` writes `(keyLen, key, state)` records to a tempfile via `std::filesystem::temp_directory_path()`. Spill is wired through **all three operator paths** (`StringFastPath` / `Fixed` / `Var`); the trigger is configurable as a group-count (`spillTriggerGroups`) and/or a memory-fraction (`spillFraction`) — the latter polls `MemoryManager::usedFraction()`. Before emit, every spill file is drained back into the live state via `findOrInsert` + `agg->merge`. Out-of-line aggregate state (`MinString`/`MaxString` values, `CountDistinct*` sets) round-trips through spill via per-aggregate `serializeState`/`deserializeState` hooks (records carry a per-record `stateLen`); after a spill the side storage is released via `clearOutOfLineState`.
- **Dict-aware HashJoin probe** — when the probe-side join key is a single dict-encoded `STRING` column, the operator precomputes `dictCode → buildSlot` once per unique dict entry; per-row work collapses to an array load + chain walk.

**Phase 4 — Complete**
- **Plan IR + JSON wire format + builder** — new `src/plan/` module defines a `PlanNode` / `ScalarExpr` IR (Scan / Filter / Project / HashAggregate / HashJoin + Literal / ColumnRef / BinaryOp / Like). `PlanJson` round-trips the IR through JSON via nlohmann-json. `PlanBuilder` walks the IR and constructs the engine's `Operator` tree, with a `SourceCatalog` (string-keyed map of `vector<ColumnBatch>`) supplying actual data to each scan node at execution time.
- **Protobuf wire format** — `src/plan/proto/quantum_plan.proto` mirrors the IR 1:1 (oneof-based unions for `ScalarExpr` and `PlanNode`, enum-typed kinds, repeated children). Generated via `protobuf_generate_cpp` at CMake configure time. `PlanProto` (`planToProtoBytes` / `planFromProtoBytes`) provides binary round-trip; JSON stays as the human-readable debug format.
- **C ABI (`include/quantum/quantum_c.h`)** — stable `extern "C"` surface for JNI / JNA / Panama. Opaque handles for engine / column / batch / result; column builders for `int64` and `string`; protobuf-encoded plan execution; iteration over result batches; per-column accessors. Exceptions are translated to a thread-local error string surfaced through `q_last_error()`.
- **Arrow C Data Interface (`include/quantum/quantum_arrow.h`)** — `ArrowSchema` / `ArrowArray` structs declared verbatim from the [Arrow spec](https://arrow.apache.org/docs/format/CDataInterface.html). `q_column_export_arrow` populates them with pointers INTO an existing `QColumn`'s buffers (zero-copy). `q_column_import_arrow` deep-copies an Arrow array into a new engine-owned column. Supports `"l"` (int64) and `"u"` (utf8) currently.
- **JVM bridge (`java/`)** — Maven module with JNA bindings to the C ABI and Java protobuf classes generated from the same `.proto`. `QuantumEngine` class wraps the opaque handles and exposes a fluent Java API. `quantum_core` is built SHARED so JNA can `dlopen` it.
- **Spark extension + plan rewriter (`quantum.spark.QuantumExtension`)** — Scala `SparkSessionExtensions` hook + `QuantumStrategy` planner. `QuantumStrategy.tryTranslateLongAggregate` recognises `Aggregate([k:Long], [Alias(k), Alias(Sum(v:Long))], LocalRelation(long, long))` (peeling through `SubqueryAlias` / column-renaming `Project`) and replaces it with a `QuantumExec` LeafExecNode. `QuantumExec.doExecute` extracts column-major Long data, builds a protobuf PlanNode, calls `QuantumEngine.execute`, and yields the native results as Spark `InternalRow`s. The headline test runs `spark.sql("SELECT k, SUM(v) FROM kv GROUP BY k")` through this path and verifies per-group sums match.

**Phase 5 — Complete** (46 slices)

A Spark Catalyst plan now translates end-to-end to a native protobuf IR that runs on the C++ engine and returns rows back to Spark. The covered SQL surface is `SELECT … FROM A [JOIN B [JOIN C …]] WHERE … [GROUP BY … HAVING …] ORDER BY … LIMIT …` — GROUP BY is optional: plain query roots (filter/join/sort/limit without an aggregate) route natively too — plus `UNION ALL` between any two peelable subqueries.

*Storage I/O.* Pluggable `FileFormatReader` interface (`open`/`nextBatch`/`close`/`schema`) covers CSV, Parquet (single + multi-file via `ConcatFileReader`), Hive-style partitioned directories (`PartitionedFileReader` synthesizes partition columns from the path), and Delta tables (`DeltaTableReader` seeds the live-file set from the latest checkpoint parquet via `_last_checkpoint` — single- and multi-part — then replays only newer `_delta_log/*.json` commits, so tables survive Delta's log cleanup; streams the surviving Parquet files. The latest `metaData.schemaString` — from JSON commits or the checkpoint — provides logical names, types, and column-mapping physical names: file columns are matched by physical name and remapped to table-schema order, with add-column schema evolution null-filled; tables without a usable schemaString fall back to footer-derived schema). Apache Iceberg v1 tables read via `IcebergTableReader` (`version-hint.text` → metadata JSON → `current-snapshot-id` → manifest-list Avro → manifests; non-deleted PARQUET entries stream through ParquetReader; avro-cpp parses the manifests). `io::FileSystem` URI dispatch — `file://`, `s3://`, `gs://`, `abfs[s]://` — via Arrow's `FileSystemFromUri`, with S3 lazily-initialised on first use. A local-disk cache (`setDiskCacheDir` / `QUANTUM_DISK_CACHE_DIR`) transparently downloads remote objects once — keyed by URI hash, atomic rename, hit/miss counters — and serves all subsequent opens from local disk (object-store data files are immutable, so no invalidation is needed). New plan IR `FileScanNode { paths, format, partition_columns, partition_values }` lets the native engine open files itself (no JNA row marshalling for file sources).

*Plan IR + bridges.* `PlanNode` variant grew `FileScanNode`, `LimitNode`, `SortNode`, `UnionAllNode`; proto adds the matching messages + `oneof` cases. `PlanBuilder` dispatches each new node to the right operator. C ABI grew `q_column_float64` (mirror of `q_column_int64`); `q_result_next` now densely materializes any non-identity `PositionList` at the JVM boundary (so a FilterNode at the plan root — HAVING, post-join filter — can't silently leak filtered-out rows through `numRows`). JNA Java bindings added `columnFloat64` + `columnString`; Java proto regenerated with the new nodes/enums. **Arrow IPC transport** (`quantum_ipc.h`): `q_register_source_ipc` ingests a whole source from one Arrow IPC stream and `q_result_next_ipc` returns each result batch as one IPC buffer — one JNA crossing per direction instead of per-column/per-row calls. `QuantumExec` marshals exclusively over IPC (arrow-java pinned to 15.x: Spark 3.5's bundled arrow 12 predates JDK 21's `DirectByteBuffer` change). The bridge handle structs moved to a shared `bridge/Handles.hpp`.

*Spark strategy (`quantum.spark.QuantumStrategy`).* The strategy was refactored mid-phase from a `DataSource`-returning `peel` to a `PeelResult { plan, idMap, output, sources, filterCount }` that builds the proto plan tree bottom-up. Each `LocalRelation` gets its own `sourceId`; `QuantumExec` registers `Seq[(String, Array[InputColumn])]` before executing, so a plan with N joined LocalRelations registers N sources. Recognised plan shapes (recursive — they compose freely):
- **Sources**: `LocalRelation`, `LogicalRelation(HadoopFsRelation(ParquetFileFormat))` — both single-file and multi-file (the strategy passes all `inputFiles` in sorted order). Delta-spark tables surface as Parquet HadoopFsRelations and route through the same path; partition columns from Hive-style dirs are extracted via `location.listFiles(Nil, Nil)` and emitted into the proto with per-file values.
- **Filters / projections**: `Filter`, `Project` (rename-only via Alias / bare AttributeReference; column-pruning Projects collapse into an idMap rebind without shrinking `peel.output`), `SubqueryAlias`.
- **Joins**: `Join(L, R, joinType, condition)` for `Inner` / `LeftOuter` / `RightOuter` / `FullOuter`. Equi-keys extracted from AND-chained `EqualTo(leftAttr, rightAttr)` (mirror tolerated); composite keys via `KeyCodec`. N-way joins work via recursive peel.
- **Aggregates**: `Aggregate(groupingExprs, aggregateExprs, child)` with multi-column `GROUP BY` over any mix of Long/Int/String/Double keys. Supported aggregates: `SUM(long|double)`, `MIN/MAX(long|double|string)`, `AVG(long|double)`, `COUNT(*)`, `COUNT(col)`, `COUNT(DISTINCT long|double|string)`. The `isDistinct` flag on `AggregateExpression` routes through a distinct classifier.
- **HAVING**: `Filter(having, Aggregate(...))` pushes the HAVING condition as `FilterNode`s above the `HashAggregateNode`, translated against the post-aggregate exprId-to-engine-column map.
- **LIMIT / ORDER BY**: the strategy matches through the planner's `ReturnAnswer` wrapper, so it claims outer `GlobalLimit / LocalLimit / Sort` before Spark's `SpecialLimits` can plan `CollectLimitExec` / `TakeOrderedAndProjectExec` / `SortExec`. Limits strip via `stripOuterLimits` (tightest combined) and emit a `LimitNode`; a root `Sort` over the Aggregate translates its keys against the post-aggregate exprId map and emits a `SortNode` between the HAVING filters and the LimitNode (HAVING → ORDER BY → LIMIT). `QuantumExec` yields a single partition, so the native total order survives `collect()`. A `Sort` below the Aggregate still peels via the child path. ORDER BY over a non-selected aggregate (Catalyst's `Limit(Project(_, Sort(…)))` shape, where a hidden ordering aggregate is added to the Aggregate and projected away above the Sort) is claimed too: the Project becomes a pure `physicalIndex` remap in `QuantumExec`, so the hidden column is computed and sorted on natively but never marshalled back to Spark.
- **Plain (non-aggregate) roots**: when the whole root plan peels (scans / filters / joins / sort / limit, no Aggregate), the strategy claims it directly — gated to the `ReturnAnswer` root so it claims whole end-user queries, not subtrees of plans Spark is mid-planning. The root plan's output attributes (or a stripped outer Project) drive QuantumExec's output mapping.
- **Set ops**: `Union(children, …)` (UNION ALL) — each child is peeled recursively, their `sources` maps are merged, and a `UnionAllNode` wraps the per-child plans (the Union's idMap canonicalizes to the first child's exprIds). UNION DISTINCT surfaces as `Aggregate(Union(…))` and routes through the Aggregate path naturally when every child peels.

*Predicates.* The translator covers `=` `<` `<=` `>` `>=` over any combination of `AttributeReference`s and Long / Int / Double / Boolean / String / `UTF8String` literals; boolean `AND` (top-level flattened to chained FilterNodes), `OR` and nested `And` (Kleene three-valued logic — `T OR NULL = T`, `F AND NULL = F`); unary `NOT` and `IS NULL` (always-valid output); `IS NOT NULL` translated as `NOT(IsNull)` (or stripped when null-propagation makes it redundant); `IN (list)` rewritten to OR-chains; `LIKE` plus Catalyst's `StartsWith`/`EndsWith`/`Contains` simplifications, all routed through the engine's `LikeOp`. The `stripRedundantIsNotNull` rule treats all comparison and string-predicate ops as null-propagating, so Spark's auto-added `isnotnull(col)` doesn't surface as an extra Filter.

*Native operators added in Phase 5.* `FileScanOperator`, `ConcatFileReader`, `PartitionedFileReader` (constant column broadcast), `LimitOperator` (PositionList truncation), `SortOperator` (materialize + permutation sort over any of INT32/INT64/FLOAT64/BOOL/STRING; with a row-count or memory-fraction trigger it switches to an external merge sort — sorted runs spill to disk as `[keys][row]` SpillFile records, then k-way merge by head keys streams 4096-row batches), `UnionAllOperator` (N-ary; drains one child at a time, output schema = first child's). Aggregate functions added: `SumF64`, `MinF64`, `MaxF64`, `AvgF64`, `CountColumn`, `CountDistinctI64`, plus Double/String COUNT(DISTINCT) variants. New expression machinery: `UnaryExpr` (Not, IsNull), `kernels::LogicalKernels` (Kleene And/Or), comparison kernels gained `Gte/Lte`. Plan IR added `BinaryOp::And/Or/Lte/Gte`, `UnaryOp::Not/IsNull`, `AggregateKind::Count/CountDistinct`.

*Caveats & deferred to Phase 6.* Three items left Phase 5 deliberately: the **exchange operator** (intra-operator parallelism) needs the Phase 6 task scheduler — the Phase 1 architecture leans on Spark for task-level parallelism, and single-partition `QuantumExec` output is what makes native ORDER BY/LIMIT semantics hold; **`LogicalQueryStage` peeling** (shuffle-bearing AQE plans) is blocked on exchange, since a materialized shuffle stage has no native-readable source; and **AVX2 kernel measurement** requires x86 hardware (the kernels exist behind `QUANTUM_HAS_AVX2` but the dev box is arm64).

*Benchmark headline.* `bench_file_scan_hash_aggregate.cpp` measures the headline path — 1M rows × (k:int64, v:int64) written as Parquet, scanned through `FileScanNode`, aggregated by `k`. On an Apple M-class laptop: ~3.3 M items/sec across plan-construct + Arrow Parquet decode + native HashAgg. Cardinality has small effect (16 vs 65536: 3.45 → 3.25 M/s), confirming the IR / operator / hash-table overhead is negligible relative to Parquet decoding.

**Phase 6 — In progress**

*AQE `LogicalQueryStage` peeling (6.6).* When AQE re-optimizes after materializing a shuffle stage, the strategy now peels `LogicalQueryStage` as a leaf source — gated to PURE-reshuffle stages (`Repartition` / `RepartitionByExpression`), whose rows are raw child rows, collected from the already-finished stage at execution time (`PeelResult.sources` became lazy thunks for this). Stages containing partial aggregation decline: re-aggregating partial buffers (e.g. COUNT partials) under the original logical Aggregate would double-count. Net effect: `df.repartition(2, $"k").groupBy("k").sum("v")` ends with `Quantum` as the AQE *final* plan — the initial two-phase plan is declined, and re-planning over the materialized stage claims the whole aggregate.

*Substrait decoder (6.5).* `plan/SubstraitDecoder` turns a serialized `substrait.Plan` (protos vendored at `third_party/substrait`, v0.42.1) into the engine's IR, so external producers — Ibis, Isthmus/Calcite, DuckDB — can target the engine directly via the new `q_execute_substrait` C ABI entry point (NamedTable reads resolve against registered sources). Covered: ReadRel/FilterRel/ProjectRel (incl. `emit` remaps)/AggregateRel (sum/min/max/avg/count + COUNT(DISTINCT) via the DISTINCT invocation)/FetchRel/SortRel; field references, bool/i32/i64/fp64/string literals, comparison + Kleene logic + `is_null` + arithmetic + `like` scalar functions resolved through the plan's extension declarations. Unsupported constructs raise `NotImplemented` naming the offender.

*Exchange operator + parallel aggregation (6.4).* `ExchangeOperator` — the in-process hash repartition: drains its child once and routes every row to one of N partitions by the hash of its key expressions (per-type hashing incl. nulls-to-a-fixed-seed, so null keys cluster). As a plain pull operator its output is hash-clustered (each partition contiguous); parallel consumers call `takePartitions()` instead. First consumer: `HashAggregateNode.parallelism > 1` builds a `ParallelHashAggregateOperator` — exchange by the GROUP BY keys, then one independent `HashAggregateOperator` per partition fanned across the `TaskScheduler`. Groups are disjoint across hash partitions, so per-partition results concatenate into the global answer with no merge phase. Plan IR + proto grew `ExchangeNode` and the `parallelism` field.

*Task scheduler + parallel scan (6.3).* `exec/TaskScheduler` — a fixed worker pool with `submit` and an exception-propagating `parallelFor` (caller participates; first error rethrown). Sized by `QUANTUM_SCHEDULER_THREADS`, defaulting to half the hardware threads — Spark supplies task-level parallelism around the engine, so the pool deliberately undersubscribes. First consumer: multi-file Parquet scans fan per-file decode (the dominant cost on the benchmark headline path) across the pool — each file is one independent task touching only the thread-safe `MemoryManager`. This is the building block the exchange operator needs.

*Plugin API (6.2).* The open-core seam: `plan/OperatorRegistry` maps an extension type (e.g. `vendor.geo_join`) to an `OperatorFactory`; plans reference plugins via a new `ExtensionNode { extension_type, payload, children }` (IR + proto). `PlanBuilder` builds the children (metrics-wrapped like everything else), then hands `(MemoryManager, children, payload)` to the factory. The payload is opaque to the engine — by convention a serialized Substrait relation the plugin decodes itself; decoding *standard* Substrait rels into built-in IR remains on the roadmap. Plugin operators flow through proto round-trips, the C ABI, and the metrics trace with zero engine changes.

*Observability (6.1).* `PlanBuilder` wraps every operator it builds in a `MetricsOperator` decorator (operators untouched; two `steady_clock` reads per batch) recording rows / batches / open-next-close timings into a per-query `QueryMetrics`. `q_result_trace_json` returns the structured trace (operators in build order, plan root last); `q_metrics_prometheus` exposes process-wide cumulative counters in Prometheus text format (`quantum_queries_total`, `quantum_rows_emitted_total`, `quantum_spills_total` — fed by the HashAggregate and Sort spill sites — and `quantum_exec_nanos_total`). `QuantumExec` captures the trace per query (`QuantumExec.lastTraceJson`); JNA surfaces both calls.

**Tests** — 244 C++ tests + 57 JVM tests (3 EngineEndToEnd + 1 Delta + 53 strategy / agg / predicate / join / source-shape).

## Build & Test

Requires CMake ≥ 3.24, Ninja, [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set, plus Homebrew `apache-arrow` and `avro-cpp` (Arrow/Parquet I/O and Iceberg manifests).

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 244 tests, ~20s (Parquet + Arrow S3 startup)

cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks

# JVM bridge (requires JDK 17+, Maven, Scala 2.13 + Spark 3.5 + Delta 3.2)
cd java && JAVA_HOME=/path/to/jdk-21 mvn test    # 57 tests (JNA + Spark + Delta)
```

Other presets: `release`, `relwithdebinfo`, `asan`. Sanitizers toggle via `-DQUANTUM_ENABLE_ASAN=ON`, `_UBSAN`, `_TSAN`.

## Architecture at a Glance

```
bridge/      → C ABI (quantum_c) + Arrow C Data Interface (quantum_arrow) + Arrow IPC (quantum_ipc)
plan/        → Plan IR + PlanBuilder (IR -> Operator) + OperatorRegistry (plugins) + PlanJson + PlanProto wire formats
exec/        → TaskScheduler (worker pool for intra-operator parallelism)
io/          → FileFormat + CsvReader + ParquetReader + DeltaTableReader + FileSystem (file:// / s3:// / gs:// / abfs://)
operator/    → Scan, FileScan, Filter, Project, HashAggregate, HashJoin, Limit, Sort, UnionAll, Exchange, ParallelHashAggregate + AggregateFunction
expression/  → Expression, Literal, ColumnRef, BinaryExpr, LikeExpr + SIMD kernels
hash/        → HashTable (int64), BytesHashTable (fixed composite), VarBytesHashTable (variable), KeyCodec
spill/       → SpillFile (HashAggregate v1)
vector/      → ColumnBatch, ColumnVector, PositionList, ValidityBitmask, StringBuilder, DictStringBuilder
memory/      → MemoryManager (budget) + Arena + Buffer (RAII off-heap)
common/      → DataType, Schema, Status, Bits, Macros
```

Key invariants the codebase enforces:

- Every `Buffer` from `MemoryManager::allocate` is 64-byte aligned — aligned SIMD loads are safe without runtime checks.
- `FilterOperator` updates the position list only; no column data is copied or compacted. This is the engine's core performance invariant.
- The pull-based operator interface uses `std::optional<ColumnBatch>` from `getNext()` (`nullopt` = end of stream).

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| **1** | Native execution core + columnar format with Scan/Filter/Project | done |
| **2** | HashAggregate + HashJoin with composite keys; MIN/MAX/AVG; LEFT OUTER join | done |
| **3** | STRING / variable-width values through scan/filter/project/join | done |
| **3** | STRING GROUP BY / JOIN keys (inner + LEFT OUTER) | done |
| **3** | RIGHT OUTER and FULL OUTER joins (with build-side matched tracking) | done |
| **3** | MIN/MAX(string) aggregates; SQL `LIKE` expression with `%` / `_` | done |
| **3** | Dictionary-encoded STRING columns (layout + builder + dict-aware gather) | done |
| **3** | Dict-aware `Eq` fast path (~4× on low-cardinality dict-vs-literal) | done |
| **3** | Dict-aware `LIKE` fast path (~3× on low-cardinality dict-vs-pattern) | done |
| **3** | Code-aware GROUP BY for single STRING key (~4× on 1M-row low-cardinality agg) | done |
| **3** | Spill-to-disk for HashAggregate (StringFastPath + Fixed + Var, group-count + memory-fraction triggers) | done |
| **3** | Dict-aware HashJoin probe (precomputed `dictCode → buildSlot` table) | done |
| **4** | Plan IR + JSON wire format + PlanBuilder | done |
| **4** | Protobuf wire format (binary, language-portable) | done |
| **4** | C ABI (`quantum_c.h`) — opaque handles, columns, batches, plan exec | done |
| **4** | Arrow C Data Interface (zero-copy column transport for int64 + utf8) | done |
| **4** | JVM bridge (Maven + JNA + Java protobuf, end-to-end query from Java) | done |
| **4** | Spark SessionExtension scaffold (`QuantumExtension` + planner strategy) | done |
| **4** | Spark plan rewriter: `Aggregate(group=long, Sum(long))` over `LocalRelation` runs natively end-to-end | done |
| **5** | Storage I/O — CSV / Parquet (single+multi-file+partitioned) / Delta; FileSystem URI dispatch (`file://` / `s3://` / `gs://` / `abfs[s]://`) | done |
| **5** | Plan IR — `FileScanNode`, `LimitNode`, `SortNode`; new comparison ops (`Gte`/`Lte`), logical ops (`And`/`Or`), unary ops (`Not`/`IsNull`); aggregate kinds `Count`/`CountDistinct` | done |
| **5** | Spark strategy — full classic SQL (`SELECT … FROM A [JOIN B …] WHERE … GROUP BY … HAVING … ORDER BY … LIMIT …`) over LocalRelation + Parquet + Delta sources | done |
| **5** | Predicates — `=` `<` `<=` `>` `>=` `AND` `OR` (Kleene) `NOT` `IS NULL` `IS NOT NULL` `IN(list)` `LIKE` / `StartsWith` / `EndsWith` / `Contains` over Long/Int/Double/Bool/String literals | done |
| **5** | Aggregates — `SUM` / `MIN` / `MAX` / `AVG` (Long + Double), `MIN`/`MAX` (String), `COUNT(*)`, `COUNT(col)`, `COUNT(DISTINCT long/string/double)`; multi-column GROUP BY (any type mix via `KeyCodec`) | done |
| **5** | Joins — Inner / Left / Right / Full Outer; composite (multi-key) joins; N-way; over any source mix | done |
| **5** | `UNION ALL` — native N-ary `UnionAllOperator` + plan IR `UnionAllNode` + Spark `Union` peel | done |
| **5** | End-to-end FileScan→HashAgg benchmark (~3.3 M items/sec, dominated by Parquet decode) | done |
| **5** | Native LIMIT / ORDER BY routing end-to-end — strategy matches through `ReturnAnswer`, beating Spark's `SpecialLimits` (`CollectLimitExec` / `TakeOrderedAndProjectExec` / `SortExec`) | done |
| **5** | ORDER BY non-selected aggregates — `Limit(Project(_, Sort(…)))` roots claimed; hidden ordering column computed/sorted natively, projected via `physicalIndex` remap | done |
| **5** | Plain query roots — `SELECT … WHERE … ORDER BY … LIMIT` without GROUP BY (incl. joins, projection/rename) claimed at the `ReturnAnswer` root | done |
| **6** | AQE `LogicalQueryStage` peeling — pure-reshuffle stages consumed as materialized sources (lazy collect); partial-agg stages decline | done |
| **5** | Iceberg v1 table reader — metadata JSON / manifest-list / manifest Avro replay, deleted entries skipped, `FileScanNode format="iceberg"` | done |
| **5** | Delta — checkpoint `.parquet` log support (`_last_checkpoint`, single- + multi-part; only post-checkpoint JSON commits replayed) | done |
| **5** | Delta — column mapping (physical-name remap from `metaData.schemaString`) + add-column schema evolution (null-fill) | done |
| **5** | `MinString`/`MaxString`/`CountDistinct*` spill — per-aggregate state serialization (`serializeState`/`deserializeState`), variable-length spill records, side storage released on spill | done |
| **5** | Sort spill — external merge sort in `SortOperator` (sorted runs on disk, k-way merge, streaming output batches) | done |
| **5** | Local-disk (NVMe) cache for object-store reads — cache-through `openReadable`, URI-hash keys, atomic install, env/API config | done |
| **5** | Arrow IPC cross-JVM batch transport — `q_register_source_ipc` / `q_result_next_ipc`, QuantumExec marshals whole batches in one JNA crossing per direction | done |
| **6** | Task scheduler (`exec/TaskScheduler`) + first intra-operator parallelism — multi-file Parquet decode fans out across the pool | done |
| **6** | Exchange operator — `ExchangeNode` hash repartition + `ParallelHashAggregateOperator` (per-partition aggregation across the scheduler, merge-free) | done |
| 6 | AVX2 kernels measured on x86 hardware (hardware-dependent; arm64 dev box) | planned |
| **5** | Dict-merge in HashJoin `concatDense` — build-side dictionaries merged/deduped with code remap; dict layout preserved through joins | done |
| **6** | Observability v1 — per-operator metrics (`MetricsOperator`), JSON query trace (`q_result_trace_json`), Prometheus counters (`q_metrics_prometheus`) | done |
| **6** | Plugin API — `OperatorRegistry` + `ExtensionNode` (opaque fragment payload, Substrait by convention); plugins compose with metrics/proto/C ABI | done |
| **6** | Substrait integration — vendored protos + `SubstraitDecoder` (Read/Filter/Project/Aggregate/Fetch/Sort) + `q_execute_substrait` | done |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
