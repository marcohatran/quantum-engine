# Quantum Engine

Quantum Engine is an **open-core C++ vectorized execution engine** designed to replace Apache Spark's JVM core with a high-performance native implementation. Inspired by proprietary solutions like Databricks Photon, Quantum Engine brings bare-metal, SIMD-accelerated execution to the open-source big-data ecosystem — without vendor lock-in.

> **Status:** Phase 1 + 2 complete, Phase 3 in progress. Full SQL join family (Inner / Left / Right / Full Outer) over composite int + string keys; STRING and dictionary-encoded STRING fully integrated across Scan / Filter / Project / Aggregate / Join. Spill, Exchange, and AVX2 still pending.

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

**Phase 3 — In Progress**
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

**Tests & benchmarks** — 122 GoogleTest cases. New since previous slice: StringFastPath GROUP BY across dict / non-dict / mixed batches, NULL keys as their own group, many-groups codeStates_ resize path. New `bench_string_group_by` sweeps dict cardinality {10, 100, 1000} for dict-vs-plain GROUP BY throughput.

## Build & Test

Requires CMake ≥ 3.24, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set.

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 122 tests, ~1.4s

cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks
```

Other presets: `release`, `relwithdebinfo`, `asan`. Sanitizers toggle via `-DQUANTUM_ENABLE_ASAN=ON`, `_UBSAN`, `_TSAN`.

## Architecture at a Glance

```
operator/    → Scan, Filter, Project, HashAggregate, HashJoin + AggregateFunction
expression/  → Expression, Literal, ColumnRef, BinaryExpr, LikeExpr + SIMD kernels
hash/        → HashTable (int64), BytesHashTable (fixed composite), VarBytesHashTable (variable), KeyCodec
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
| 3 | Spill-to-disk, AVX2 kernels, Exchange operator, intra-operator parallelism, dict-merge in HashJoin concat, code-aware HashJoin keys | in progress |
| 4 | JNI bridge with Protobuf wire format + `QuantumPlanner` Spark extension | planned |
| 5 | Storage I/O — Parquet, Iceberg, Delta over S3 / ADLS / GCS, NVMe cache | planned |
| 6 | Observability — Prometheus metrics, query trace, Substrait plugin API | planned |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
