# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantum Engine is an open-core C++ vectorized execution engine designed to replace Apache Spark's JVM core with a high-performance native implementation, similar in spirit to Databricks Photon. The full architecture is documented in `quantum-engine-architecture.md` (seven layers L0–L6 spanning Spark integration, JNI bridge, native core, columnar format, storage I/O, and observability).

**Current phase:** Phase 1 — native execution core (L3) and columnar format (L4) only. No JNI bridge, no Spark integration, no storage I/O yet.

## Build & Test

CMake + vcpkg + Ninja. Presets are defined in `CMakePresets.json`. `$VCPKG_ROOT` must be set.

```bash
cmake --preset=debug                 # configure
cmake --build --preset=debug         # build
ctest --preset=debug                 # run all tests
ctest --preset=debug -R BinaryExpr   # single test by name
cmake --build --preset=release       # release build (benchmarks)
./build-release/benchmarks/quantum_benchmarks --benchmark_filter=Pipeline
```

Slash commands `/build`, `/test`, `/sanitize`, `/benchmark`, `/format`, `/lint`, `/profile` are defined in `.claude/commands/` and wrap these invocations.

Build options (toggle via `-D` or preset cache vars): `QUANTUM_BUILD_TESTS`, `QUANTUM_BUILD_BENCHMARKS`, `QUANTUM_ENABLE_ASAN`, `QUANTUM_ENABLE_UBSAN`, `QUANTUM_ENABLE_TSAN`, `QUANTUM_ENABLE_AVX512`.

## Code Architecture

Single source tree under `src/`, headers and `.cpp` colocated. Public umbrella headers reserved at `include/quantum/` for future Phase 2 JNI consumers. One static lib target: `quantum_core`. One test binary: `quantum_tests`. One benchmark binary: `quantum_benchmarks`.

Module layering (lower depends on nothing above it):

```
operator/    → ScanOperator, FilterOperator, ProjectOperator    (pull-based execution)
expression/  → Expression, Literal, ColumnRef, BinaryExpr + kernels/
vector/      → ColumnBatch, ColumnVector, PositionList, ValidityBitmask
memory/      → MemoryManager (budget) + Arena + Buffer (RAII off-heap)
common/      → DataType, Schema, Status (QuantumError/QuantumOOM), Bits, Macros
```

### Key invariants

- **64-byte alignment** — every `Buffer` from `MemoryManager::allocate` is cache-line / AVX-512 aligned. Callers may use aligned SIMD loads (`_mm256_load_si256`) without checking.
- **Position-list filtering** — `FilterOperator` updates `PositionList` only; it **does not** copy or compact column data. This is the architecture's core perf invariant; preserve it when adding operators.
- **`PositionList` identity fast path** — pre-filter, the list stores no indices and `at(i)` returns `i` directly. Identity-mode hot paths should not allocate.
- **Pull-based operator interface** — `open()`, `getNext() → std::optional<ColumnBatch>` (nullopt = EOS), `close()`. Child ownership via `std::unique_ptr`.
- **Expression eval contract** — returns a `ColumnVector` of length `batch.numRows()` (not `activeRows()`). Slots outside the position list are undefined but the result's validity bit is cleared. Null propagation: `lhs.validity AND rhs.validity`.
- **Single-threaded operators** — `MemoryManager` is the only thread-safe component. Spark provides task-level parallelism; intra-operator parallelism is not in scope.

### Explicitly NOT in Phase 1

Spill, dictionary encoding, string/variable-width types, hash agg/join, exchange, JNI bridge, Protobuf wire format, storage I/O, observability, Substrait plugin, native UDF runtime, adaptive kernels, AVX-512, ARM. Do not add these unless extending into Phase 2+.

### Adding a new operator

1. Create `src/operator/Foo{Operator}.{hpp,cpp}`, derive from `Operator`.
2. Constructor takes `MemoryManager*`, child `OperatorPtr`, operator-specific args.
3. `getNext()` should preserve position lists where possible (never copy column data unnecessarily).
4. Add source to `src/CMakeLists.txt` and a test to `tests/operator/`.

### Adding a new expression / kernel

1. New expression types extend `Expression` in `src/expression/`.
2. New SIMD kernels go in `src/expression/kernels/`. Scalar implementation first; AVX2 fast path on the identity-position-list branch; scalar fallback for non-identity. Keep `QUANTUM_HAS_AVX2` guards.
3. `BinaryExpr::eval` dispatches via compile-time switch on `(op, TypeId)` — register new ops there.
