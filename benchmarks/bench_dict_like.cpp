#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "expression/LikeExpr.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/DictStringBuilder.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch dictNamesBatch(MemoryManager& mm, size_t n, size_t dictSize) {
    DictStringBuilder b(&mm);
    for (size_t i = 0; i < n; ++i) {
        b.append("name_" + std::to_string(i % dictSize));
    }
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

ColumnBatch plainNamesBatch(MemoryManager& mm, size_t n, size_t dictSize) {
    StringBuilder b(&mm);
    for (size_t i = 0; i < n; ++i) {
        b.append("name_" + std::to_string(i % dictSize));
    }
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

} // namespace

// LIKE over a dict-encoded column: pattern is evaluated against each unique
// dictionary entry once; rows just index into the result table.
static void BM_Like_Dict(benchmark::State& state) {
    const size_t n        = 4096;
    const size_t dictSize = static_cast<size_t>(state.range(0));
    MemoryManager mm(8 * 1024 * 1024);
    auto batch = dictNamesBatch(mm, n, dictSize);

    for (auto _ : state) {
        LikeExpr like(std::make_unique<ColumnRef>(0), "name_%3");
        auto r = like.eval(batch, &mm);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Like_Dict)->Arg(10)->Arg(100)->Arg(1000);

// Same LIKE over a regular (non-dict) STRING column: pattern is matched
// per row from scratch.
static void BM_Like_Plain(benchmark::State& state) {
    const size_t n        = 4096;
    const size_t dictSize = static_cast<size_t>(state.range(0));
    MemoryManager mm(8 * 1024 * 1024);
    auto batch = plainNamesBatch(mm, n, dictSize);

    for (auto _ : state) {
        LikeExpr like(std::make_unique<ColumnRef>(0), "name_%3");
        auto r = like.eval(batch, &mm);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Like_Plain)->Arg(10)->Arg(100)->Arg(1000);
