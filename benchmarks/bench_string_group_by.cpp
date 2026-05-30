#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "vector/DictStringBuilder.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch dictBatch(MemoryManager& mm, size_t n, size_t dictSize) {
    DictStringBuilder b(&mm);
    auto vCol = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        b.append("name_" + std::to_string(i % dictSize));
        vCol.typed<int64_t>()[i] = static_cast<int64_t>(i);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    cols.push_back(std::move(vCol));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

ColumnBatch plainBatch(MemoryManager& mm, size_t n, size_t dictSize) {
    StringBuilder b(&mm);
    auto vCol = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        b.append("name_" + std::to_string(i % dictSize));
        vCol.typed<int64_t>()[i] = static_cast<int64_t>(i);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    cols.push_back(std::move(vCol));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

OperatorPtr scan(MemoryManager& mm, std::vector<ColumnBatch> src) {
    return std::make_unique<ScanOperator>(
        &mm, Schema({{"name", DataType{TypeId::STRING}}, {"v", Int64}}),
        std::move(src));
}

std::vector<ExpressionPtr> keyOn(size_t i) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(i));
    return v;
}

} // namespace

static void BM_GroupByDict(benchmark::State& state) {
    const size_t totalRows = 1U << 20;       // 1M rows
    const size_t batchSize = 4096;
    const size_t dictSize  = static_cast<size_t>(state.range(0));
    const size_t numBatches = totalRows / batchSize;
    MemoryManager mm(256 * 1024 * 1024);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ColumnBatch> src;
        src.reserve(numBatches);
        for (size_t b = 0; b < numBatches; ++b) src.push_back(dictBatch(mm, batchSize, dictSize));
        auto s = scan(mm, std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
        HashAggregateOperator op(&mm, std::move(s), keyOn(0),
                                 std::move(aggs), {"name", "sum"});
        op.open();
        state.ResumeTiming();
        while (auto b = op.getNext()) benchmark::DoNotOptimize(b);
        state.PauseTiming();
        op.close();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(totalRows));
}
BENCHMARK(BM_GroupByDict)->Arg(10)->Arg(100)->Arg(1000);

static void BM_GroupByPlain(benchmark::State& state) {
    const size_t totalRows = 1U << 20;
    const size_t batchSize = 4096;
    const size_t dictSize  = static_cast<size_t>(state.range(0));
    const size_t numBatches = totalRows / batchSize;
    MemoryManager mm(256 * 1024 * 1024);

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ColumnBatch> src;
        src.reserve(numBatches);
        for (size_t b = 0; b < numBatches; ++b) src.push_back(plainBatch(mm, batchSize, dictSize));
        auto s = scan(mm, std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
        HashAggregateOperator op(&mm, std::move(s), keyOn(0),
                                 std::move(aggs), {"name", "sum"});
        op.open();
        state.ResumeTiming();
        while (auto b = op.getNext()) benchmark::DoNotOptimize(b);
        state.PauseTiming();
        op.close();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(totalRows));
}
BENCHMARK(BM_GroupByPlain)->Arg(10)->Arg(100)->Arg(1000);
