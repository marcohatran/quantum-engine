# Quantum Engine

Quantum Engine is an **open-core C++ vectorized execution engine** designed to replace Apache Spark's JVM core with a high-performance native implementation. Inspired by proprietary solutions like Databricks Photon, Quantum Engine brings bare-metal, SIMD-accelerated execution to the open-source big-data ecosystem — without vendor lock-in.

> **Status:** Phase 1 — native execution core (L3) and columnar format (L4). Self-contained C++ pipeline; Spark / JNI integration is on the roadmap.

See [`quantum-engine-architecture.md`](./quantum-engine-architecture.md) for the full seven-layer design.

---

## What's in Phase 1

- **Columnar data format** — `ColumnBatch`, `ColumnVector`, `ValidityBitmask`, and a `PositionList` with an identity fast path so filters update indices instead of compacting data.
- **Off-heap memory manager** — arena-backed allocator with per-query budgets and RAII `Buffer`s, 64-byte aligned for cache-line / AVX use.
- **Expression engine** — `Literal`, `ColumnRef`, `BinaryExpr` over `Int32 / Int64 / Float64 / Bool` with arithmetic (+ − ×) and comparison (= < >) kernels and null propagation.
- **Pull-based operators** — `ScanOperator`, `FilterOperator`, `ProjectOperator` implementing `open() / getNext() / close()`.
- **Tests & benchmarks** — 40 GoogleTest cases (including 3 end-to-end `Scan → Filter → Project` scenarios) and a Google Benchmark suite covering validity ops, filter selectivity sweeps, projection, and full pipeline throughput.

## Build & Test

Requires CMake ≥ 3.24, Ninja, and [vcpkg](https://github.com/microsoft/vcpkg) with `$VCPKG_ROOT` set.

```bash
export VCPKG_ROOT=/path/to/vcpkg

cmake --preset=debug                 # configure (vcpkg installs deps)
cmake --build --preset=debug         # build
ctest --preset=debug                 # 40 tests, ~0.3s

cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks
```

Other presets: `release`, `relwithdebinfo`, `asan`. Sanitizers toggle via `-DQUANTUM_ENABLE_ASAN=ON`, `_UBSAN`, `_TSAN`.

## Architecture at a Glance

```
operator/    → ScanOperator, FilterOperator, ProjectOperator   (pull-based execution)
expression/  → Expression, Literal, ColumnRef, BinaryExpr + SIMD kernels
vector/      → ColumnBatch, ColumnVector, PositionList, ValidityBitmask
memory/      → MemoryManager (budget) + Arena + Buffer (RAII off-heap)
common/      → DataType, Schema, Status, Bits, Macros
```

Key invariants the codebase enforces:

- Every `Buffer` from `MemoryManager::allocate` is 64-byte aligned — aligned SIMD loads are safe without runtime checks.
- `FilterOperator` updates the position list only; no column data is copied or compacted. This is the engine's core performance invariant.
- The pull-based operator interface uses `std::optional<ColumnBatch>` from `getNext()` (`nullopt` = end of stream).

## Roadmap

| Phase | Deliverable |
|---|---|
| **1 (done)** | Native execution core + columnar format with Scan/Filter/Project |
| 2 | Hash aggregate / hash join, spill-to-disk, dictionary encoding, string types, AVX2 kernels |
| 3 | Exchange operator, intra-operator parallelism |
| 4 | JNI bridge with Protobuf wire format + `QuantumPlanner` Spark extension |
| 5 | Storage I/O — Parquet, Iceberg, Delta over S3 / ADLS / GCS, NVMe cache |
| 6 | Observability — Prometheus metrics, query trace, Substrait plugin API |

## License

Open-core. Engine source is open source; enterprise plugins (planned) are commercial.
