#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {
ColumnBatch kv(MemoryManager& mm, size_t n, size_t cardinality) {
    auto k = ColumnVector::make(Int64, n, &mm);
    auto v = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        k.typed<int64_t>()[i] = static_cast<int64_t>(i % cardinality);
        v.typed<int64_t>()[i] = static_cast<int64_t>(i);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

static void BM_HashAggregate(benchmark::State& state) {
    const size_t totalRows   = 1U << 20;     // 1M rows
    const size_t batchSize   = 4096;
    const size_t cardinality = static_cast<size_t>(state.range(0));
    const size_t numBatches  = totalRows / batchSize;

    MemoryManager mm(256 * 1024 * 1024);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ColumnBatch> src;
        src.reserve(numBatches);
        for (size_t b = 0; b < numBatches; ++b) src.push_back(kv(mm, batchSize, cardinality));
        auto scan = std::make_unique<ScanOperator>(
            &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
        std::vector<ExpressionPtr> keys;
        keys.push_back(std::make_unique<ColumnRef>(0));
        HashAggregateOperator op(&mm, std::move(scan),
                                 std::move(keys),
                                 std::move(aggs),
                                 {"k", "sum"});
        op.open();
        state.ResumeTiming();

        while (auto b = op.getNext()) benchmark::DoNotOptimize(b);

        state.PauseTiming();
        op.close();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(totalRows));
}
BENCHMARK(BM_HashAggregate)->Arg(16)->Arg(1024)->Arg(65536);
