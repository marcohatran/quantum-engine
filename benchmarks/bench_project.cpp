#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/ProjectOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {
ColumnBatch ab(MemoryManager& mm, size_t n) {
    auto a = ColumnVector::make(Int64, n, &mm);
    auto b = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        a.typed<int64_t>()[i] = static_cast<int64_t>(i);
        b.typed<int64_t>()[i] = static_cast<int64_t>(i * 2);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(a));
    cols.push_back(std::move(b));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

static void BM_ProjectAddInt64(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    MemoryManager mm(64 * 1024 * 1024);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ColumnBatch> src;
        src.push_back(ab(mm, n));
        auto scan = std::make_unique<ScanOperator>(
            &mm, Schema({{"a", Int64}, {"b", Int64}}), std::move(src));
        std::vector<ExpressionPtr> exprs;
        exprs.push_back(std::make_unique<BinaryExpr>(
            BinaryOp::Add, std::make_unique<ColumnRef>(0), std::make_unique<ColumnRef>(1)));
        ProjectOperator op(&mm, std::move(scan), std::move(exprs), {"out"});
        op.open();
        state.ResumeTiming();

        auto out = op.getNext();
        benchmark::DoNotOptimize(out);

        state.PauseTiming();
        op.close();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * 8);
}
BENCHMARK(BM_ProjectAddInt64)->Arg(1024)->Arg(4096)->Arg(16384);
