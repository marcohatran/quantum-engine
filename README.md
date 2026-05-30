# Quantum Engine

Quantum Engine is an **open-core C++ vectorized execution engine** designed to replace Apache Spark's JVM core with a high-performance native implementation. Inspired by proprietary solutions like Databricks Photon, Quantum Engine brings bare-metal, SIMD-accelerated execution to the open-source big-data ecosystem — without vendor lock-in.

> **Status:** Phase 1 complete; Phase 2 in progress (HashAggregate + HashJoin landed). Self-contained C++ pipeline; Spark / JNI integration is on the roadmap.

See [`quantum-engine-architecture.md`](./quantum-engine-architecture.md) for the full seven-layer design.

---

## What's Implemented

**Phase 1 — Foundation**
- **Columnar data format** — `ColumnBatch`, `ColumnVector`, `ValidityBitmask`, and a `PositionList` with an identity fast path so filters update indices instead of compacting data.
- **Off-heap memory manager** — arena-backed allocator with per-query budgets and RAII `Buffer`s, 64-byte aligned for cache-line / AVX use.
- **Expression engine** — `Literal`, `ColumnRef`, `BinaryExpr` over `Int32 / Int64 / Float64 / Bool` with arithmetic (+ − ×) and comparison (= < >) kernels and null propagation.
- **Pull-based operators** — `ScanOperator`, `FilterOperator`, `ProjectOperator` implementing `open() / getNext() / close()`.

**Phase 2 — In Progress**
- **`HashAggregateOperator`** — `GROUP BY` over an `Int64` key with `SUM` and `COUNT(*)` aggregates. Builds the per-group state table then emits one batch of `(key, agg...)`. Null keys form a dedicated group.
- **`HashJoinOperator`** — inner equi-join on a single `Int64` key. Build side is densified and chained via a `next[]` array indexed by `HashTable`; probe side streams batch-by-batch. Null keys never match (SQL semantics).
- **`HashTable`** — open-addressing, linear probing, 0.7-load-factor rehash, off-heap and memory-budgeted. Shared by both `HashAggregateOperator` and `HashJoinOperator`.

**Tests & benchmarks** — 59 GoogleTest cases (E2E pipelines + HashAggregate group/null paths + HashJoin one-to-one / multi-match / null-key / multi-batch-build / empty-side coverage). Google Benchmark suite covers validity ops, filter selectivity sweeps, projection, full-pipeline throughput, HashAggregate cardinality sweeps, and HashJoin cardinality sweeps.

## Build & Test

Requires CMake ≥ 3.24, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set.

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 59 tests, ~0.5s

cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks
```

Other presets: `release`, `relwithdebinfo`, `asan`. Sanitizers toggle via `-DQUANTUM_ENABLE_ASAN=ON`, `_UBSAN`, `_TSAN`.

## Architecture at a Glance

```
operator/    → Scan, Filter, Project, HashAggregate, HashJoin + AggregateFunction
expression/  → Expression, Literal, ColumnRef, BinaryExpr + SIMD kernels
hash/        → HashTable (open addressing, linear probing, off-heap)
vector/      → ColumnBatch, ColumnVector, PositionList, ValidityBitmask
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
| **2** | HashAggregate (int64 key, SUM/COUNT) | done |
| **2** | HashJoin (inner, int64 key) | done |
| 2 | Multi-column / string keys, outer joins, spill-to-disk, dictionary encoding, more aggregates (MIN/MAX/AVG), AVX2 kernels | in progress |
| 3 | Exchange operator, intra-operator parallelism | planned |
| 4 | JNI bridge with Protobuf wire format + `QuantumPlanner` Spark extension | planned |
| 5 | Storage I/O — Parquet, Iceberg, Delta over S3 / ADLS / GCS, NVMe cache | planned |
| 6 | Observability — Prometheus metrics, query trace, Substrait plugin API | planned |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
