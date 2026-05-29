#include <benchmark/benchmark.h>

#include "memory/MemoryManager.hpp"
#include "vector/ValidityBitmask.hpp"

using namespace quantum;

static void BM_NullCount(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    MemoryManager mm(1U << 20);
    ValidityBitmask m(&mm, n);
    for (size_t i = 0; i < n; i += 7) m.setValid(i, false);
    for (auto _ : state) {
        benchmark::DoNotOptimize(m.nullCount());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_NullCount)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_AndInPlace(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    MemoryManager mm(1U << 20);
    ValidityBitmask a(&mm, n);
    ValidityBitmask b(&mm, n);
    for (auto _ : state) {
        a.andInPlace(b);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(a.sizeBytes()));
}
BENCHMARK(BM_AndInPlace)->Arg(4096)->Arg(16384)->Arg(65536);
