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
- **Object-store filesystem layer (`src/io/FileSystem.{hpp,cpp}`)** — single `io::openReadable(uri)` helper dispatches on URI scheme through `arrow::fs::FileSystemFromUri`. Bare paths and `file://` go to the local FS; `s3://bucket/key`, `gs://bucket/key`, and `abfs[s]://…` go to the corresponding Arrow filesystem (provided the Arrow build includes them — Homebrew's does). S3 is initialized lazily on first use via `arrow::fs::InitializeS3`. Plumbed into `ParquetReader`, so `ParquetReader("s3://bucket/key.parquet", &mm)` works the same as a local path. Credentials follow the AWS SDK's standard chain (env vars / `~/.aws/credentials` / instance profile).
- **Delta Lake table reader (`src/io/DeltaTableReader.{hpp,cpp}`)** — minimum-viable Delta reader. Lists `<root>/_delta_log/*.json`, replays `add` / `remove` actions in version order, and streams batches from each surviving Parquet file via the existing `ParquetReader`. Implements `FileFormatReader` so it slots into `FileScanOperator` (and the rest of the operator tree) with no special-casing. URI scheme inherits from the FS layer — `DeltaTableReader("s3://bucket/my-delta-table", &mm)` works the same as a local directory. Schema is derived from the first surviving Parquet file; checkpoint `.parquet` log files are ignored for now (commit-log-only replay).
- **Broader Spark aggregate coverage (`quantum.spark.QuantumStrategy`)** — the strategy now recognises `SUM`, `MIN`, `MAX`, `AVG` of `LongType` and `COUNT(*)`, in any combination, against a `LocalRelation` of Long columns. The peel through `SubqueryAlias` / column-renaming `Project` is now exprId-tracking, so column renames don't break translation. `QuantumExec` was generalized: it takes the Spark output `Seq[Attribute]` plus a physical-index map (native column order is `[group keys][aggs]`; Spark's output may reorder them), and dispatches per-column read on `LongType` vs `DoubleType` (AVG returns `Double`). Added `q_batch_col_float64` to the C ABI to plumb Double columns through JNA. Rows are projected via `UnsafeProjection` so the resulting `RDD[InternalRow]` carries `UnsafeRow`s that Spark's downstream paths (collect / shuffle / codegen) accept.
- **Multi-column GROUP BY + WHERE pushdown (`quantum.spark.QuantumStrategy`)** — `GROUP BY (k1, k2, …)` over any number of Long columns now routes through the native composite-key path (`BytesHashTable` + `KeyCodec`). `WHERE …` clauses are translated to `FilterNode`s wrapped around the Scan; conjunctions are flattened, so `WHERE a > 1 AND b < 5` becomes a chain of two single-clause Filters (the native `BinaryOp` has no `AND` yet). Predicates can reference any combination of Long `AttributeReference`s and Long/Int literals via `>`, `<`, `=`; unsupported predicates make the strategy decline (Spark falls back to its own planner). Each predicate is translated against the column-map valid at the depth where the `Filter` was peeled, so renaming `Project`s above the `Filter` don't break translation.
- **String columns through the JNA bridge (`QuantumEngine.columnString`, `QuantumExec`'s `InputColumn` ADT)** — Spark `StringType` columns marshal to the native engine via `q_column_string`. JVM-side helper packs UTF-8 bytes into per-row JNA `Memory` blocks plus a (ptr-array, len-array) pair, with an optional validity bitmap when any value is null. `QuantumExec` now takes a sealed `InputColumn` ADT (`LongColumn | StringColumn`) and dispatches at column-build time. Output reading dispatches per declared dataType: `LongType` and `DoubleType` pull contiguous arrays once per batch, `StringType` reads row-by-row via `q_batch_col_string_at` and wraps in `UTF8String` for Spark.
- **String GROUP BY (`quantum.spark.QuantumStrategy`)** — the strategy's LocalRelation type check now accepts `StringType` columns in addition to `LongType`. Schema fields in the proto plan reflect each column's declared type (`TYPE_INT64` / `TYPE_STRING`). Group keys, agg inputs, and outputs work for any supported mix of Long and String. A `GROUP BY name:String` query routes through the engine's `StringFastPath` (single STRING key, global dict); a `GROUP BY (name:String, k2:Long)` composite-key query routes through `VarBytesHashTable` via `KeyCodec`.
- **AQE compatibility** — the test suite no longer disables `spark.sql.adaptive.enabled` (Spark's AQE). The `LogicalQueryStage`-hides-Aggregate concern from the original 4.7 scaffold doesn't apply to our LocalRelation-backed queries because there's no Exchange below the Aggregate; AQE therefore doesn't create a stage boundary that would wrap the Aggregate. (Multi-partition / shuffle-bearing queries from a real source would still need explicit `LogicalQueryStage` peeling — listed in the roadmap.)
- **Native file-source scans (`plan::FileScanNode`)** — new plan IR node that lets the native engine open files itself rather than receiving pre-materialized row data through JNA. The proto adds a `FileScanNode { repeated string paths; string format; Schema schema; }`; `PlanBuilder` dispatches on `format` ("parquet" → `ParquetReader`; "delta" → `DeltaTableReader`). Multi-file Parquet (a directory of part files) is rejected for now — listed as a follow-up.
- **Spark → file-source end-to-end (`quantum.spark.QuantumStrategy`)** — strategy now peels through `LogicalRelation(HadoopFsRelation(ParquetFileFormat, …))` in addition to `LocalRelation`. For Parquet relations, the strategy extracts the file paths from `HadoopFsRelation.location.inputFiles` and emits a `FileScanNode` in the plan instead of marshalling row data. `QuantumExec` skips source registration when the input column list is empty — the engine reads the file on its own. The introduced `DataSource` sealed trait (`InMemorySource` | `FileSource`) keeps the two cases distinguishable in one place. A `spark.read.parquet("/path").groupBy("k").agg(sum("v"))` now runs natively without ever marshalling row data through JNA.
- **Multi-file Parquet (`src/io/ConcatFileReader.hpp`)** — generic `FileFormatReader` shim that chains a sequence of underlying readers, opening each lazily and closing it before advancing. `PlanBuilder` uses it for `FileScanNode { paths: [N >= 2], format: "parquet" }` — each path becomes its own `ParquetReader` and `ConcatFileReader` yields batches across them in order. Single-path FileScans still go through a bare `ParquetReader` to keep the common case allocation-free. The Spark strategy now passes all of `HadoopFsRelation.location.inputFiles` (in sorted order) instead of declining on the multi-file case, so `spark.read.parquet("dir")` over an N-partition write routes natively.
- **WHERE pushdown over FileScan (`quantum.spark.QuantumStrategy`)** — the predicate stack already composes with `FileScanNode`s (FilterNode wraps any child including FileScans), so `spark.read.parquet(path).filter("v > 25").groupBy("k").sum("v")` translates as `HashAgg(Filter(v > 25, FileScan))`. Spark's optimizer wraps user predicates over file sources in an extra `isnotnull(col)` clause (helps parquet skip NULL pages); the strategy now strips a redundant `IsNotNull(col)` conjunct whenever the same column appears in a comparison conjunct, because the native engine's comparison ops already null-propagate (NULL → NULL → excluded by Filter).
- **GTE / LTE comparison ops (`BinaryOp::Gte`, `BinaryOp::Lte`)** — `BinaryOp` and `kernels::CmpOp` enums extended; comparison kernels handle the two new ops for INT32 / INT64 / FLOAT64 / BOOL / STRING; plan IR variant + proto enum (`BINOP_GTE` / `BINOP_LTE`); PlanBuilder + PlanProto encode / decode round-trips. Spark strategy now translates `GreaterThanOrEqual` / `LessThanOrEqual` directly (previously made the strategy decline).
- **Spark → Delta end-to-end (no new strategy code)** — delta-spark exposes a Delta table as `LogicalRelation(HadoopFsRelation(TahoeLogFileIndex, ParquetFileFormat, …))` whose `inputFiles` is the active parquet file set per Delta's current snapshot. The Phase 5.10 multi-file Parquet path already handles that shape, so `spark.read.format("delta").load(dir).groupBy("k").sum("v")` routes natively via `FileScanNode(format="parquet", paths=[N active files])` — no new Scala code was needed. The only real fix was in `io::FileSystem`: Delta hands paths to us in Hadoop URI form (`file:/some/path`, single slash, no authority) and the local-path fallback was double-prepending `file://`. `normalizeUri` now rewrites `file:/x` → `file:///x` so Arrow's `FileSystemFromUri` accepts it.
- **Boolean `AND` / `OR` ops (`BinaryOp::And/Or`, `kernels::LogicalOp`)** — new `src/expression/kernels/LogicalKernels.{hpp,cpp}` implements SQL Kleene three-valued logic: `T OR NULL = T` (truth dominates), `F AND NULL = F` (falsity dominates), other NULL combinations are NULL. Plan IR + proto (`BINOP_AND`, `BINOP_OR`); PlanBuilder + PlanProto round-trip. Spark strategy translates `Or(l, r)` and nested `And(l, r)` directly — top-level `AND` is still flattened into chained FilterNodes (one per conjunct), but a top-level `OR` becomes one Filter whose predicate is the BinaryOp tree. This unlocks predicates like `WHERE k = 1 OR k = 3 OR v > 45`.
- **Non-Long literals in predicates (`quantum.spark.QuantumStrategy.translateScalarWith`)** — translator now recognises `Literal(UTF8String, StringType)` → proto `setStringValue`, `Literal(java.lang.Double, DoubleType)` → `setDoubleValue`, and `Literal(java.lang.Boolean, BooleanType)` → `setBoolValue`. Catalyst stores string literals as `UTF8String` (not Java `String`), so the pattern-match dispatches on the `UTF8String` class. This makes `WHERE name = 'apple' OR name = 'banana'` push as a single Filter against the native engine's STRING comparison kernel.
- **Hive-style partition columns (`src/io/PartitionedFileReader.{hpp,cpp}`)** — new `FileFormatReader` shim that wraps an inner reader (typically `ParquetReader`) and synthesizes constant partition columns at the end of each batch. Output schema is `[inner.schema fields] ++ [partition fields]`. `FileScanNode` (proto + plan IR) gained `partition_columns: Schema` + `partition_values: PartitionValueRow[]` parallel to `paths`; `PlanBuilder` wraps each per-file `ParquetReader` in a `PartitionedFileReader` when partitions are present, then chains them through `ConcatFileReader` as before. The Spark strategy walks `HadoopFsRelation.location.listFiles(Nil, Nil)` to extract per-file partition values from each `PartitionDirectory.values` row, and emits them into the proto. Spark infers `IntegerType` for partition values like `year=2024`, so the strategy widens `IntegerType` → `INT64` internally and `QuantumExec` narrows back to `Int` when materializing the row.
- **Partition filter pushdown** — verified via JVM test: `spark.read.parquet(dir).filter("year = 2025")` over a 3-partition dataset routes natively, and only the `year=2025` partition is in the strategy's `paths`. Spark's `FileIndex` prunes the file set BEFORE the strategy sees the `LogicalRelation`, so no engine-side work was needed.
- **End-to-end FileScan → HashAgg benchmark (`benchmarks/bench_file_scan_hash_aggregate.cpp`)** — measures the Phase 5 headline path: 1M rows × (k:int64, v:int64), written as Parquet, scanned through `FileScanNode`, aggregated by `k`. Per-iteration timing includes plan construction + `PlanBuilder` + `FileSystem::openReadable` + Arrow Parquet decode + HashAgg over `cardinality` groups. On an Apple M-class laptop with macOS Arrow: ~3.3 M rows/sec across the full path; cardinality has small effect (cardinality=16 → 3.45 M/s vs cardinality=65536 → 3.25 M/s), confirming the IR / operator / hash-table overhead is negligible relative to Parquet decoding.
- **Unary `NOT` (`src/expression/UnaryExpr.{hpp,cpp}`)** — first unary expression in the engine. Validity-preserving boolean negation: NULL → NULL, true ↔ false. Plan IR adds `UnaryOpExpr` to the `ScalarExpr` variant + `UnaryOpType { Not }` enum; proto adds `UnaryOp` message and `UNOP_NOT` enum (the `ScalarExpr` oneof gets a fifth case). PlanBuilder + PlanProto round-trip. Spark translator handles Catalyst's `Not(child: Expression)` directly. Catalyst's optimizer rewrites `NOT (a = b)` to `a != b` and `NOT (a > b)` to `a <= b`, so the Spark-visible cases for `Not` are mostly `NOT IN`, `NOT EXISTS`, and explicit `NOT(bool_expr)`.
- **`COUNT(col)` non-star aggregate** — new `CountColumn` `AggregateFunction` that counts active rows whose input column value is non-null. New `AggregateKind::Count` (proto `AGG_COUNT = 5`); PlanBuilder dispatches to `CountColumn`. Spark translator recognises `Count(Seq(ar: AttributeReference))` → `AGG_COUNT` (in addition to the existing `Count(Seq(Literal))` → `AGG_COUNT_STAR`). Real fix that surfaced during the JVM test: `InMemorySource` marshaling was dropping NULL information from `LongColumn` and `IntegerType` columns (always passed `validity=null` to the C ABI). The Scala-side `LongColumn` now carries an optional `Array[Byte]` validity, materialized only when at least one row in the LocalRelation has a NULL for that column.
- **`IS NULL` / `IS NOT NULL` predicates** — second `UnaryOp` variant: `IsNull` reads only the input's validity bitmap, output is always valid (SQL: `IS NULL` never returns NULL). Works on any input type — no per-type kernel needed. `UnaryOpType::IsNull` + proto `UNOP_IS_NULL = 1`; PlanBuilder + PlanProto encode/decode dispatch on the op. Spark translator emits `UNOP_IS_NULL` for `IsNull(child)` and `NOT(IsNull(child))` for standalone `IsNotNull(child)`. The earlier `stripRedundantIsNotNull` rule still fires when the same column appears in a comparison conjunct (where the comparison's null-propagation already excludes the row), so an extra `Not(IsNull)` isn't generated in those cases.
- **Spark JOIN translation (`quantum.spark.QuantumStrategy` refactor)** — `peel` now returns `PeelResult { plan, idMap, output, sources, filterCount }` — the fully-built proto plan tree, the exprId→column map, the output schema, and the list of in-memory sources to register (each `LocalRelation` gets its own `sourceId`). `QuantumExec` correspondingly takes `sources: Seq[(String, Array[InputColumn])]` and registers each before executing. The new `Join` peel case recursively peels both sides, extracts equi-join key pairs from the condition (`AND`-chained `EqualTo(leftAttr, rightAttr)`, with mirror), and emits a `HashJoinNode` wrapping the two sub-plans; combined `idMap` shifts the right side's positions by `leftR.output.size`. `Inner` / `LeftOuter` / `RightOuter` / `FullOuter` are supported. The pre-existing predicate / partition / IS NULL / OR / NOT / AGG paths still work end-to-end through the refactored peel.
- **Multi-source and N-way joins (verified)** — the recursive `Join` peel composes with: (a) Parquet file sources on either or both sides (`spark.read.parquet(A) JOIN spark.read.parquet(B)` registers zero in-memory sources — both leaves are `FileScanNode`s); (b) 3-way joins (`ta JOIN tb JOIN tc` plans as `Join(Join(A,B), C)`, recursive peel shifts the right side's idMap positions at each level). No code changes — both shapes drop out of 5.22's design.
- **LEFT OUTER / RIGHT OUTER / FULL OUTER JOIN** — verified end-to-end. Surfaced a naming-convention mismatch: the native engine's `LEFT_OUTER` preserves the *probe* side and `RIGHT_OUTER` preserves the *build* side, but our Spark strategy puts Spark's LEFT input into `build`. So Spark's `LeftOuter` (preserve LEFT) maps to engine's `RIGHT_OUTER` (preserve build), and Spark's `RightOuter` (preserve RIGHT) maps to engine's `LEFT_OUTER` (preserve probe). `FullOuter` and `Inner` pass straight through.
- **Composite-key JOIN** — `JOIN ON a.k1 = b.k1 AND a.k2 = b.k2` over mixed Long+String keys. The strategy's `extractEquiKeys` already decomposes the `AND`-chained `EqualTo`s into a sequence of `(build_col, probe_col)` pairs; the native `HashJoin` uses `KeyCodec` for composite keys (Phase 2/3). Verified end-to-end including duplicate matches on one side and unmatched composites.
- **`IN` predicate + JOIN-with-WHERE** — `In(value, list)` is translated by rewriting to an OR-chain (`value = a OR value = b OR …`) before recursing through the existing translator. Catalyst sometimes performs this rewrite itself but not always; handling it in the strategy makes us robust to either shape. JOIN-with-WHERE (`SELECT … FROM A JOIN B WHERE jwb.v >= 25 GROUP BY …`) drops out of the existing Filter peel — the Filter above the Join becomes a `FilterNode` wrapping the `HashJoinNode`.
- **`HAVING` clause (native pushdown)** — the strategy now matches `Filter(having, Aggregate(...))` and translates the `having` condition against `agg.output.zip(physicalIndex)` (the post-aggregate output exprId → engine column map). Each AND-conjunct becomes its own `FilterNode` chained above the `HashAggregateNode`. Untranslatable HAVING → the strategy declines, Spark falls back to `FilterExec(QuantumExec)`.
- **C ABI: dense materialization at result boundary** — surfaced by HAVING. The architecture preserves `PositionList`s for performance (filters update indices, don't compact data), but JVM consumers iterate `0..numRows-1` and have no way to consume a position list. Result: a `FilterNode` at the plan's root (HAVING, post-join filter, …) would silently leak filtered-out rows through the C ABI. Fix: `q_result_next` now calls `selectActive` on any batch whose position list isn't identity, producing densely-packed columns at the JVM boundary. One-time copy per output batch; only triggers when the root has a non-identity position list.

**Tests & benchmarks** — 183 C++ tests + 37 JVM tests. New since previous slice: JVM `havingClauseAfterGroupBy` (`SELECT k, SUM(v) FROM kv GROUP BY k HAVING SUM(v) > 100`).

## Build & Test

Requires CMake ≥ 3.24, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set.

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 167 tests, ~19s (Parquet + Arrow S3 startup)

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
io/          → FileFormat + CsvReader + ParquetReader + DeltaTableReader + FileSystem (file:// / s3:// / gs:// / abfs://)
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
| **5** | Object-store filesystem layer (S3 / GCS / Azure Blob via Arrow URI dispatch) | done |
| **5** | Delta Lake table reader (JSON commit-log replay, add/remove resolution) | done |
| **5** | Spark: SUM / COUNT(*) / MIN / MAX / AVG of Long, multi-agg per query, Double output via JNA | done |
| **5** | Spark: multi-column GROUP BY (any number of Long keys via KeyCodec) | done |
| **5** | Spark: WHERE pushdown — Long `<` / `>` / `=` over col/lit; AND-flattened to chained FilterNodes | done |
| **5** | Spark: String columns through JNA + GROUP BY String / mixed (String, Long) composite keys | done |
| **5** | Spark: AQE-on by default — LocalRelation queries route natively under `spark.sql.adaptive.enabled=true` | done |
| **5** | Plan IR: `FileScanNode` — native engine opens files itself (no JNA row marshalling) | done |
| **5** | Spark: `LogicalRelation(HadoopFsRelation)` for Parquet → `FileScanNode` end-to-end | done |
| **5** | Multi-file Parquet directory → `ConcatFileReader` chains `ParquetReader`s end-to-end | done |
| **5** | Spark: WHERE pushdown composes with FileScan (and redundant `IsNotNull` stripping) | done |
| **5** | Native + Spark: `Gte` / `Lte` comparison ops (kernels, plan IR, proto, translator) | done |
| **5** | Spark → Delta end-to-end via the existing multi-file Parquet path (Delta files surface as Parquet) | done |
| **5** | Native + Spark: `Or` (Kleene logic) and nested `And` — top-level AND still flattens to chained Filters | done |
| **5** | Spark: WHERE literals for `String` / `Double` / `Boolean` (Catalyst `UTF8String`, Java Double / Boolean) | done |
| **5** | Hive-style partition columns (`PartitionedFileReader`, per-file constant broadcast) — end-to-end from Spark | done |
| **5** | Partition filter pushdown (Spark's `FileIndex` prunes before peel — verified by JVM test) | done |
| **5** | End-to-end FileScan→HashAgg benchmark (Parquet read + plan build + native execute) | done |
| **5** | Unary `NOT` expression — first unary op in the engine + plan IR + proto + Spark translator | done |
| **5** | `COUNT(col)` non-star aggregate — counts non-null rows + InMemorySource validity plumbing | done |
| **5** | `IS NULL` / `IS NOT NULL` — second `UnaryOp` variant; works on any input type | done |
| **5** | Spark JOIN translation — `Join(L, R, joinType, EqualTo-chain)` → native `HashJoinNode`; multi-source `QuantumExec` | done |
| **5** | JOIN over Parquet file sources + N-way join (recursive Join peel) — verified by JVM test | done |
| **5** | LEFT / RIGHT / FULL OUTER JOIN — fixed Spark↔engine naming-convention swap; JVM-tested | done |
| **5** | Composite-key JOIN — `ON a.k1 = b.k1 AND a.k2 = b.k2` over mixed Long+String keys (`KeyCodec`) | done |
| **5** | `IN (list)` predicate — Catalyst-shape-robust OR-chain rewrite; JOIN-with-WHERE over joined output | done |
| **5** | `HAVING` clause native pushdown + C ABI dense-materialization fix at result boundary | done |
| 5 | Spark: WHERE on String / Int / Double; GTE / LTE / OR / NOT (needs new BinaryOp / UnaryOp kinds) | planned |
| 5 | Spark: `LogicalQueryStage` peeling for shuffle-bearing AQE plans | planned |
| 5 | Iceberg table reader (needs Avro for manifest files) | planned |
| 5 | Delta: checkpoint `.parquet` log support, partition pruning, schema evolution | planned |
| 5 | NVMe disk cache for object-store reads | planned |
| 5 | Broader Spark plan-shape coverage (other aggregates, joins, multi-column keys, AQE) | planned |
| 5 | Arrow IPC for cross-JVM batch transport (replaces per-column JNA marshalling) | planned |
| 5 | Exchange operator, intra-operator parallelism (needs task scheduler) | planned |
| 5 | AVX2 kernels measured on x86 hardware (no-op on arm64 dev box) | planned |
| 5 | `MinString`/`MaxString` spill (out-of-line state serialization) | planned |
| 5 | Dict-merge in HashJoin `concatDense` (preserve dict layout through joins) | planned |
| 6 | Observability — Prometheus metrics, query trace, Substrait plugin API | planned |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
