#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/HashJoinOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {
ColumnBatch kv(MemoryManager& mm, size_t n, size_t cardinality, size_t startKey) {
    auto k = ColumnVector::make(Int64, n, &mm);
    auto v = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        k.typed<int64_t>()[i] = static_cast<int64_t>(startKey + (i % cardinality));
        v.typed<int64_t>()[i] = static_cast<int64_t>(i);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

// Build = 16K rows; probe = 256K rows; sweep build-side key cardinality so
// match rate per probe row varies.
static void BM_HashJoin(benchmark::State& state) {
    const size_t buildRows  = 16 * 1024;
    const size_t probeRows  = 256 * 1024;
    const size_t batchSize  = 4096;
    const size_t card       = static_cast<size_t>(state.range(0));

    MemoryManager mm(512 * 1024 * 1024);

    for (auto _ : state) {
        state.PauseTiming();

        std::vector<ColumnBatch> buildSrc;
        for (size_t r = 0; r < buildRows; r += batchSize) {
            const size_t n = std::min(batchSize, buildRows - r);
            buildSrc.push_back(kv(mm, n, card, /*startKey=*/0));
        }
        std::vector<ColumnBatch> probeSrc;
        for (size_t r = 0; r < probeRows; r += batchSize) {
            const size_t n = std::min(batchSize, probeRows - r);
            probeSrc.push_back(kv(mm, n, card, /*startKey=*/0));
        }

        auto buildScan = std::make_unique<ScanOperator>(
            &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(buildSrc));
        auto probeScan = std::make_unique<ScanOperator>(
            &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(probeSrc));

        HashJoinOperator op(&mm, std::move(buildScan), std::move(probeScan),
                            std::make_unique<ColumnRef>(0),
                            std::make_unique<ColumnRef>(0),
                            {"bk", "bv", "pk", "pv"});
        op.open();
        state.ResumeTiming();

        size_t emitted = 0;
        while (auto b = op.getNext()) emitted += b->numRows();
        benchmark::DoNotOptimize(emitted);

        state.PauseTiming();
        op.close();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(probeRows));
}
BENCHMARK(BM_HashJoin)->Arg(16)->Arg(1024)->Arg(16384);
