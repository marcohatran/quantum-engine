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
- **Dictionary-encoded STRING columns** — `ColumnVector` grows a third layout: `values_` holds `int32` codes; `dictOffsets_` + `dictValues_` hold the shared dictionary. `DictStringBuilder` builds dict columns row-by-row with dedup. `stringAt()` dispatches uniformly so existing kernels (compare, LIKE, MIN/MAX) and GROUP BY/JOIN key codecs work unchanged. Dict layout is preserved through `clone` / `selectActive` / `HashJoin gather`; `HashJoin concatDense` currently decodes mixed-dict inputs to regular STRING (dict-merge is a follow-up).
- **Dict-aware equality fast path** — `Literal(std::string)` now emits a dict-encoded broadcast (`dictSize=1`), and the STRING `Eq` kernel detects dict-vs-dict inputs, precomputes a `rhsCode → lhsCode` translation table, and compares int32s per row. Measured **~4× throughput** on `dict_size=10` vs the per-row `stringAt` path; gains shrink as dict size grows.
- **Dict-aware LIKE fast path** — `LikeExpr::eval` on a dict-encoded column evaluates the pattern matcher once per dictionary entry (into a `dictResult[dictSize]` table), then per row just indexes into that table by code. Measured **~3× throughput** at `dict_size=10`, **~3.6×** at `dict_size=100`, **~2.4×** at `dict_size=1000` — LIKE benefits more than `Eq` because per-row pattern matching is the expensive bit.
- **Code-aware GROUP BY (single STRING key)** — `HashAggregateOperator` adds a `StringFastPath` selected for single-STRING-key aggregates. Maintains a global dictionary across batches; for dict-encoded inputs per-row work shrinks to a `batchCode → globalCode` array lookup plus an array-indexed group state slot. Non-dict batches still intern per row into the same global dictionary so results are identical. Measured on 1M-row aggregation: **~4× throughput** at `dict_size=10` (128M vs 33M items/s), **~3.9×** at 100, **~1.6×** at 1000.
- **Spill-to-disk for HashAggregate** — `AggregateFunction::merge` added on the numeric aggregates; new `SpillFile` writes `(keyLen, key, state)` records to a tempfile via `std::filesystem::temp_directory_path()`. Spill is wired through **all three operator paths** (`StringFastPath` / `Fixed` / `Var`); the trigger is configurable as a group-count (`spillTriggerGroups`) and/or a memory-fraction (`spillFraction`) — the latter polls `MemoryManager::usedFraction()`. Before emit, every spill file is drained back into the live state via `findOrInsert` + `agg->merge`. `MinString`/`MaxString` raise `NotImplemented` on spill (out-of-line state — Phase 4).
- **Dict-aware HashJoin probe** — when the probe-side join key is a single dict-encoded `STRING` column, the operator precomputes `dictCode → buildSlot` once per unique dict entry; per-row work collapses to an array load + chain walk.

**Phase 4 — Complete**
- **Plan IR + JSON wire format + builder** — new `src/plan/` module defines a `PlanNode` / `ScalarExpr` IR (Scan / Filter / Project / HashAggregate / HashJoin + Literal / ColumnRef / BinaryOp / Like). `PlanJson` round-trips the IR through JSON via nlohmann-json. `PlanBuilder` walks the IR and constructs the engine's `Operator` tree, with a `SourceCatalog` (string-keyed map of `vector<ColumnBatch>`) supplying actual data to each scan node at execution time.
- **Protobuf wire format** — `src/plan/proto/quantum_plan.proto` mirrors the IR 1:1 (oneof-based unions for `ScalarExpr` and `PlanNode`, enum-typed kinds, repeated children). Generated via `protobuf_generate_cpp` at CMake configure time. `PlanProto` (`planToProtoBytes` / `planFromProtoBytes`) provides binary round-trip; JSON stays as the human-readable debug format.
- **C ABI (`include/quantum/quantum_c.h`)** — stable `extern "C"` surface for JNI / JNA / Panama. Opaque handles for engine / column / batch / result; column builders for `int64` and `string`; protobuf-encoded plan execution; iteration over result batches; per-column accessors. Exceptions are translated to a thread-local error string surfaced through `q_last_error()`.
- **Arrow C Data Interface (`include/quantum/quantum_arrow.h`)** — `ArrowSchema` / `ArrowArray` structs declared verbatim from the [Arrow spec](https://arrow.apache.org/docs/format/CDataInterface.html). `q_column_export_arrow` populates them with pointers INTO an existing `QColumn`'s buffers (zero-copy). `q_column_import_arrow` deep-copies an Arrow array into a new engine-owned column. Supports `"l"` (int64) and `"u"` (utf8) currently.
- **JVM bridge (`java/`)** — Maven module with JNA bindings to the C ABI and Java protobuf classes generated from the same `.proto`. `QuantumEngine` class wraps the opaque handles and exposes a fluent Java API. `quantum_core` is built SHARED so JNA can `dlopen` it.
- **Spark extension + plan rewriter (`quantum.spark.QuantumExtension`)** — Scala `SparkSessionExtensions` hook + `QuantumStrategy` planner. `QuantumStrategy.tryTranslateLongAggregate` recognises `Aggregate([k:Long], [Alias(k), Alias(Sum(v:Long))], LocalRelation(long, long))` (peeling through `SubqueryAlias` / column-renaming `Project`) and replaces it with a `QuantumExec` LeafExecNode. `QuantumExec.doExecute` extracts column-major Long data, builds a protobuf PlanNode, calls `QuantumEngine.execute`, and yields the native results as Spark `InternalRow`s. The headline test runs `spark.sql("SELECT k, SUM(v) FROM kv GROUP BY k")` through this path and verifies per-group sums match.

**Phase 5 — Complete** (37 slices)

A Spark Catalyst plan now translates end-to-end to a native protobuf IR that runs on the C++ engine and returns rows back to Spark. The covered SQL surface is `SELECT … FROM A [JOIN B [JOIN C …]] WHERE … GROUP BY … HAVING … ORDER BY … LIMIT …`, plus `UNION ALL` between any two peelable subqueries.

*Storage I/O.* Pluggable `FileFormatReader` interface (`open`/`nextBatch`/`close`/`schema`) covers CSV, Parquet (single + multi-file via `ConcatFileReader`), Hive-style partitioned directories (`PartitionedFileReader` synthesizes partition columns from the path), and Delta tables (`DeltaTableReader` replays `_delta_log/*.json`, streams the surviving Parquet files). `io::FileSystem` URI dispatch — `file://`, `s3://`, `gs://`, `abfs[s]://` — via Arrow's `FileSystemFromUri`, with S3 lazily-initialised on first use. New plan IR `FileScanNode { paths, format, partition_columns, partition_values }` lets the native engine open files itself (no JNA row marshalling for file sources).

*Plan IR + bridges.* `PlanNode` variant grew `FileScanNode`, `LimitNode`, `SortNode`, `UnionAllNode`; proto adds the matching messages + `oneof` cases. `PlanBuilder` dispatches each new node to the right operator. C ABI grew `q_column_float64` (mirror of `q_column_int64`); `q_result_next` now densely materializes any non-identity `PositionList` at the JVM boundary (so a FilterNode at the plan root — HAVING, post-join filter — can't silently leak filtered-out rows through `numRows`). JNA Java bindings added `columnFloat64` + `columnString`; Java proto regenerated with the new nodes/enums.

*Spark strategy (`quantum.spark.QuantumStrategy`).* The strategy was refactored mid-phase from a `DataSource`-returning `peel` to a `PeelResult { plan, idMap, output, sources, filterCount }` that builds the proto plan tree bottom-up. Each `LocalRelation` gets its own `sourceId`; `QuantumExec` registers `Seq[(String, Array[InputColumn])]` before executing, so a plan with N joined LocalRelations registers N sources. Recognised plan shapes (recursive — they compose freely):
- **Sources**: `LocalRelation`, `LogicalRelation(HadoopFsRelation(ParquetFileFormat))` — both single-file and multi-file (the strategy passes all `inputFiles` in sorted order). Delta-spark tables surface as Parquet HadoopFsRelations and route through the same path; partition columns from Hive-style dirs are extracted via `location.listFiles(Nil, Nil)` and emitted into the proto with per-file values.
- **Filters / projections**: `Filter`, `Project` (rename-only via Alias / bare AttributeReference; column-pruning Projects collapse into an idMap rebind without shrinking `peel.output`), `SubqueryAlias`.
- **Joins**: `Join(L, R, joinType, condition)` for `Inner` / `LeftOuter` / `RightOuter` / `FullOuter`. Equi-keys extracted from AND-chained `EqualTo(leftAttr, rightAttr)` (mirror tolerated); composite keys via `KeyCodec`. N-way joins work via recursive peel.
- **Aggregates**: `Aggregate(groupingExprs, aggregateExprs, child)` with multi-column `GROUP BY` over any mix of Long/Int/String/Double keys. Supported aggregates: `SUM(long|double)`, `MIN/MAX(long|double|string)`, `AVG(long|double)`, `COUNT(*)`, `COUNT(col)`, `COUNT(DISTINCT long|double|string)`. The `isDistinct` flag on `AggregateExpression` routes through a distinct classifier.
- **HAVING**: `Filter(having, Aggregate(...))` pushes the HAVING condition as `FilterNode`s above the `HashAggregateNode`, translated against the post-aggregate exprId-to-engine-column map.
- **LIMIT / ORDER BY**: the strategy matches through the planner's `ReturnAnswer` wrapper, so it claims outer `GlobalLimit / LocalLimit / Sort` before Spark's `SpecialLimits` can plan `CollectLimitExec` / `TakeOrderedAndProjectExec` / `SortExec`. Limits strip via `stripOuterLimits` (tightest combined) and emit a `LimitNode`; a root `Sort` over the Aggregate translates its keys against the post-aggregate exprId map and emits a `SortNode` between the HAVING filters and the LimitNode (HAVING → ORDER BY → LIMIT). `QuantumExec` yields a single partition, so the native total order survives `collect()`. A `Sort` below the Aggregate still peels via the child path. ORDER BY over a non-selected aggregate (Catalyst's `Limit(Project(_, Sort(…)))` shape, where a hidden ordering aggregate is added to the Aggregate and projected away above the Sort) is claimed too: the Project becomes a pure `physicalIndex` remap in `QuantumExec`, so the hidden column is computed and sorted on natively but never marshalled back to Spark.
- **Set ops**: `Union(children, …)` (UNION ALL) — each child is peeled recursively, their `sources` maps are merged, and a `UnionAllNode` wraps the per-child plans (the Union's idMap canonicalizes to the first child's exprIds). UNION DISTINCT surfaces as `Aggregate(Union(…))` and routes through the Aggregate path naturally when every child peels.

*Predicates.* The translator covers `=` `<` `<=` `>` `>=` over any combination of `AttributeReference`s and Long / Int / Double / Boolean / String / `UTF8String` literals; boolean `AND` (top-level flattened to chained FilterNodes), `OR` and nested `And` (Kleene three-valued logic — `T OR NULL = T`, `F AND NULL = F`); unary `NOT` and `IS NULL` (always-valid output); `IS NOT NULL` translated as `NOT(IsNull)` (or stripped when null-propagation makes it redundant); `IN (list)` rewritten to OR-chains; `LIKE` plus Catalyst's `StartsWith`/`EndsWith`/`Contains` simplifications, all routed through the engine's `LikeOp`. The `stripRedundantIsNotNull` rule treats all comparison and string-predicate ops as null-propagating, so Spark's auto-added `isnotnull(col)` doesn't surface as an extra Filter.

*Native operators added in Phase 5.* `FileScanOperator`, `ConcatFileReader`, `PartitionedFileReader` (constant column broadcast), `LimitOperator` (PositionList truncation), `SortOperator` (materialize + permutation sort over any of INT32/INT64/FLOAT64/BOOL/STRING), `UnionAllOperator` (N-ary; drains one child at a time, output schema = first child's). Aggregate functions added: `SumF64`, `MinF64`, `MaxF64`, `AvgF64`, `CountColumn`, `CountDistinctI64`, plus Double/String COUNT(DISTINCT) variants. New expression machinery: `UnaryExpr` (Not, IsNull), `kernels::LogicalKernels` (Kleene And/Or), comparison kernels gained `Gte/Lte`. Plan IR added `BinaryOp::And/Or/Lte/Gte`, `UnaryOp::Not/IsNull`, `AggregateKind::Count/CountDistinct`.

*Caveats & known follow-ups.* `COUNT(DISTINCT)` covers Long/Double/String but is in-memory only (distinct-set spill not implemented). `LogicalQueryStage` peeling (for shuffle-bearing AQE plans), Iceberg, NVMe disk cache, Arrow IPC cross-JVM transport, and an exchange operator for intra-operator parallelism remain on the roadmap.

*Benchmark headline.* `bench_file_scan_hash_aggregate.cpp` measures the headline path — 1M rows × (k:int64, v:int64) written as Parquet, scanned through `FileScanNode`, aggregated by `k`. On an Apple M-class laptop: ~3.3 M items/sec across plan-construct + Arrow Parquet decode + native HashAgg. Cardinality has small effect (16 vs 65536: 3.45 → 3.25 M/s), confirming the IR / operator / hash-table overhead is negligible relative to Parquet decoding.

**Tests** — 194 C++ tests + 49 JVM tests (2 EngineEndToEnd + 1 Delta + 46 strategy / agg / predicate / join / source-shape).

## Build & Test

Requires CMake ≥ 3.24, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set.

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 194 tests, ~20s (Parquet + Arrow S3 startup)

cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks

# JVM bridge (requires JDK 17+, Maven, Scala 2.13 + Spark 3.5 + Delta 3.2)
cd java && JAVA_HOME=/path/to/jdk-21 mvn test    # 49 tests (JNA + Spark + Delta)
```

Other presets: `release`, `relwithdebinfo`, `asan`. Sanitizers toggle via `-DQUANTUM_ENABLE_ASAN=ON`, `_UBSAN`, `_TSAN`.

## Architecture at a Glance

```
bridge/      → C ABI (quantum_c) + Arrow C Data Interface (quantum_arrow)
plan/        → Plan IR + PlanBuilder (IR -> Operator) + PlanJson + PlanProto wire formats
io/          → FileFormat + CsvReader + ParquetReader + DeltaTableReader + FileSystem (file:// / s3:// / gs:// / abfs://)
operator/    → Scan, FileScan, Filter, Project, HashAggregate, HashJoin, Limit, Sort, UnionAll + AggregateFunction
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
| 5 | `LogicalQueryStage` peeling for shuffle-bearing AQE plans | planned |
| 5 | Iceberg table reader (needs Avro for manifest files) | planned |
| 5 | Delta — checkpoint `.parquet` log support, schema evolution, column mapping | planned |
| 5 | Sort spill; `MinString`/`MaxString`/`CountDistinct*` spill (out-of-line state serialization) | planned |
| 5 | NVMe disk cache for object-store reads | planned |
| 5 | Arrow IPC for cross-JVM batch transport (replaces per-column JNA marshalling) | planned |
| 5 | Exchange operator, intra-operator parallelism (needs task scheduler) | planned |
| 5 | AVX2 kernels measured on x86 hardware (no-op on arm64 dev box) | planned |
| 5 | Dict-merge in HashJoin `concatDense` (preserve dict layout through joins) | planned |
| 6 | Observability — Prometheus metrics, query trace, Substrait plugin API | planned |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
