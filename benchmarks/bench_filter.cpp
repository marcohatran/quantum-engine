#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {
ColumnBatch makeBatch(MemoryManager& mm, size_t n) {
    auto col = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) col.typed<int64_t>()[i] = static_cast<int64_t>(i);
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

// Sweep over predicate threshold to approximate selectivity values.
// Selectivity ~= (n - threshold) / n.
static void BM_Filter(benchmark::State& state) {
    const size_t n = 4096;
    const int64_t threshold = state.range(0);
    MemoryManager mm(64 * 1024 * 1024);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ColumnBatch> src;
        src.push_back(makeBatch(mm, n));
        auto scan = std::make_unique<ScanOperator>(&mm, Schema({{"a", Int64}}), std::move(src));
        auto pred = std::make_unique<BinaryExpr>(
            BinaryOp::Gt, std::make_unique<ColumnRef>(0), std::make_unique<Literal>(threshold));
        FilterOperator op(&mm, std::move(scan), std::move(pred));
        op.open();
        state.ResumeTiming();

        auto out = op.getNext();
        benchmark::DoNotOptimize(out);

        state.PauseTiming();
        op.close();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Filter)->Arg(4054)->Arg(3686)->Arg(2048)->Arg(409)->Arg(40);
