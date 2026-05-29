#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/ProjectOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {
ColumnBatch ab(MemoryManager& mm, size_t n, int64_t startA) {
    auto a = ColumnVector::make(Int64, n, &mm);
    auto b = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        a.typed<int64_t>()[i] = startA + static_cast<int64_t>(i);
        b.typed<int64_t>()[i] = static_cast<int64_t>(i * 2);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(a));
    cols.push_back(std::move(b));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

static void BM_Pipeline_FilterProject(benchmark::State& state) {
    const size_t batchSize  = 4096;
    const size_t numBatches = static_cast<size_t>(state.range(0));
    MemoryManager mm(256 * 1024 * 1024);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ColumnBatch> src;
        for (size_t i = 0; i < numBatches; ++i) {
            src.push_back(ab(mm, batchSize, static_cast<int64_t>(i * batchSize)));
        }
        auto scan = std::make_unique<ScanOperator>(
            &mm, Schema({{"a", Int64}, {"b", Int64}}), std::move(src));
        auto pred = std::make_unique<BinaryExpr>(
            BinaryOp::Gt, std::make_unique<ColumnRef>(0),
            std::make_unique<Literal>(int64_t{100}));
        auto filter = std::make_unique<FilterOperator>(&mm, std::move(scan), std::move(pred));
        std::vector<ExpressionPtr> exprs;
        exprs.push_back(std::make_unique<BinaryExpr>(
            BinaryOp::Add, std::make_unique<ColumnRef>(0), std::make_unique<ColumnRef>(1)));
        ProjectOperator op(&mm, std::move(filter), std::move(exprs), {"out"});
        op.open();
        state.ResumeTiming();

        while (auto b = op.getNext()) benchmark::DoNotOptimize(b);

        state.PauseTiming();
        op.close();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(batchSize * numBatches));
}
BENCHMARK(BM_Pipeline_FilterProject)->Arg(1)->Arg(8)->Arg(64);
