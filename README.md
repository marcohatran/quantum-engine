# Quantum Engine

Quantum Engine is an **open-core C++ vectorized execution engine** designed to replace Apache Spark's JVM core with a high-performance native implementation. Inspired by proprietary solutions like Databricks Photon, Quantum Engine brings bare-metal, SIMD-accelerated execution to the open-source big-data ecosystem — without vendor lock-in.

> **Status:** Phase 1 + 2 complete, Phase 3 in progress. Full SQL join family (Inner / Left / Right / Full Outer) over composite int + string keys; STRING fully integrated across Scan / Filter / Project / Aggregate / Join. Dictionary encoding, spill, Exchange, and AVX2 still pending.

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

**Tests & benchmarks** — 90 GoogleTest cases. New since previous slice: 7 outer-join tests (RIGHT OUTER basic, FULL OUTER basic, empty probe / empty build / both empty, NULL build keys in tail, duplicate build key). Google Benchmark suite covers validity ops, filter selectivity sweeps, projection, full pipeline, HashAggregate cardinality sweeps, and HashJoin cardinality sweeps.

## Build & Test

Requires CMake ≥ 3.24, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set.

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 90 tests, ~0.9s

cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks
```

Other presets: `release`, `relwithdebinfo`, `asan`. Sanitizers toggle via `-DQUANTUM_ENABLE_ASAN=ON`, `_UBSAN`, `_TSAN`.

## Architecture at a Glance

```
operator/    → Scan, Filter, Project, HashAggregate, HashJoin + AggregateFunction
expression/  → Expression, Literal, ColumnRef, BinaryExpr + SIMD kernels
hash/        → HashTable (int64), BytesHashTable (fixed composite), VarBytesHashTable (variable), KeyCodec
vector/      → ColumnBatch, ColumnVector, PositionList, ValidityBitmask, StringBuilder
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
| 3 | Dictionary encoding, spill-to-disk, AVX2 kernels, Exchange operator, intra-operator parallelism | in progress |
| 4 | JNI bridge with Protobuf wire format + `QuantumPlanner` Spark extension | planned |
| 5 | Storage I/O — Parquet, Iceberg, Delta over S3 / ADLS / GCS, NVMe cache | planned |
| 6 | Observability — Prometheus metrics, query trace, Substrait plugin API | planned |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
