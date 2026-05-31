# Quantum Engine

Quantum Engine is an **open-core C++ vectorized execution engine** designed to replace Apache Spark's JVM core with a high-performance native implementation. Inspired by proprietary solutions like Databricks Photon, Quantum Engine brings bare-metal, SIMD-accelerated execution to the open-source big-data ecosystem — without vendor lock-in.

> **Status:** Phases 1–4 complete. Phase 5 in progress — pluggable `FileFormat` abstraction with a CSV reader landed; Parquet / Iceberg / Delta backends slot into the same shape. A `SELECT k, SUM(v) FROM kv GROUP BY k` written in Spark SQL now runs natively in the C++ engine via `QuantumExtension` → Catalyst-plan rewrite → protobuf IR → JNA → C ABI → native operators, then returns rows back to Spark.

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

**Phase 5 — In Progress**
- **Pluggable `FileFormat` reader (`src/io/FileFormat.hpp`)** — abstract interface for on-disk readers. `open()` / `nextBatch() → optional<ColumnBatch>` / `close()` / `schema()`. The same shape is what the future Parquet / Iceberg / Delta readers will implement.
- **CSV reader (`src/io/CsvReader.{hpp,cpp}`)** — line-based parser covering `BOOL` / `INT32` / `INT64` / `FLOAT64` / `STRING`, with empty-field-as-NULL and double-quoted fields (incl. `""` for an embedded quote). Configurable batch size via `FileFormatOptions`.
- **`FileScanOperator` (`src/operator/FileScanOperator.hpp`)** — thin adapter that wraps any `FileFormatReader` as an `Operator`, so CSV input slots in below `Filter` / `Project` / `HashAggregate` / `HashJoin` exactly like the existing in-memory `ScanOperator`.
- **Parquet reader (`src/io/ParquetReader.{hpp,cpp}`)** — built on Apache Arrow's `parquet::arrow` C++ API (installed via Homebrew to sidestep vcpkg's thrift/bison friction). Supports `BOOL` / `INT32` / `INT64` / `FLOAT64` / `STRING` (regular + LARGE_STRING) Parquet columns. The full file is loaded into an Arrow `Table` at construction so the schema is available before `open()` (required by downstream operators like `ProjectOperator` that compute their output schema at construction time); `nextBatch` yields slices of `options.batchSize` rows. Dict-encoded Parquet columns are read as plain STRING for now (re-dict-encoding into the engine's `DictStringBuilder` layout is a follow-up).

**Tests & benchmarks** — 155 C++ tests + 4 JVM tests. New since previous slice: Parquet roundtrip (ints + strings), Parquet batched read with configurable batchSize, and an end-to-end `Parquet → Filter(v>15) → Project(k, v*10)` pipeline against a tempfile written via Arrow's parquet writer.

## Build & Test

Requires CMake ≥ 3.24, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set.

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 155 tests, ~9s (Parquet adds Arrow startup)

cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks

# JVM bridge (requires JDK 17+, Maven, Scala 2.13 + Spark 3.5)
cd java && JAVA_HOME=/path/to/jdk-21 mvn test    # 4 tests (JNA + Spark extension)
```

Other presets: `release`, `relwithdebinfo`, `asan`. Sanitizers toggle via `-DQUANTUM_ENABLE_ASAN=ON`, `_UBSAN`, `_TSAN`.

## Architecture at a Glance

```
bridge/      → C ABI (quantum_c) + Arrow C Data Interface (quantum_arrow)
plan/        → Plan IR + PlanBuilder (IR -> Operator) + PlanJson + PlanProto wire formats
io/          → FileFormat reader interface + CsvReader + ParquetReader (Iceberg/Delta to come)
operator/    → Scan, FileScan, Filter, Project, HashAggregate, HashJoin + AggregateFunction
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
| **5** | Pluggable `FileFormat` reader interface + `CsvReader` + `FileScanOperator` | done |
| **5** | Parquet reader (via Apache Arrow's `parquet::arrow` C++ API, int / float / string) | done |
| 5 | Iceberg / Delta table-format readers | planned |
| 5 | S3 / ADLS / GCS object-store backends + NVMe disk cache | planned |
| 5 | Broader Spark plan-shape coverage (other aggregates, joins, multi-column keys, AQE) | planned |
| 5 | Arrow IPC for cross-JVM batch transport (replaces per-column JNA marshalling) | planned |
| 5 | Exchange operator, intra-operator parallelism (needs task scheduler) | planned |
| 5 | AVX2 kernels measured on x86 hardware (no-op on arm64 dev box) | planned |
| 5 | `MinString`/`MaxString` spill (out-of-line state serialization) | planned |
| 5 | Dict-merge in HashJoin `concatDense` (preserve dict layout through joins) | planned |
| 6 | Observability — Prometheus metrics, query trace, Substrait plugin API | planned |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
