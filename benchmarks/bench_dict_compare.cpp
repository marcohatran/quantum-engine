#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/DictStringBuilder.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch dictBatch(MemoryManager& mm, size_t n, size_t dictSize) {
    DictStringBuilder b(&mm);
    for (size_t i = 0; i < n; ++i) {
        b.append("name_" + std::to_string(i % dictSize));
    }
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

ColumnBatch plainBatch(MemoryManager& mm, size_t n, size_t dictSize) {
    StringBuilder b(&mm);
    for (size_t i = 0; i < n; ++i) {
        b.append("name_" + std::to_string(i % dictSize));
    }
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

} // namespace

// Comparing a dict-encoded STRING column against a literal: the
// dict-vs-dict fast path translates the literal once and then per-row
// compares int32s.
static void BM_DictEq_Literal(benchmark::State& state) {
    const size_t n        = 4096;
    const size_t dictSize = static_cast<size_t>(state.range(0));
    MemoryManager mm(8 * 1024 * 1024);
    auto batch = dictBatch(mm, n, dictSize);

    for (auto _ : state) {
        BinaryExpr eq(BinaryOp::Eq,
                      std::make_unique<ColumnRef>(0),
                      std::make_unique<Literal>(std::string{"name_3"}));
        auto r = eq.eval(batch, &mm);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_DictEq_Literal)->Arg(10)->Arg(100)->Arg(1000);

// Same comparison against a regular (non-dict-encoded) STRING column —
// the slow path that decodes via stringAt() per row.
static void BM_PlainStringEq_Literal(benchmark::State& state) {
    const size_t n        = 4096;
    const size_t dictSize = static_cast<size_t>(state.range(0));
    MemoryManager mm(8 * 1024 * 1024);
    auto batch = plainBatch(mm, n, dictSize);

    for (auto _ : state) {
        BinaryExpr eq(BinaryOp::Eq,
                      std::make_unique<ColumnRef>(0),
                      std::make_unique<Literal>(std::string{"name_3"}));
        auto r = eq.eval(batch, &mm);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_PlainStringEq_Literal)->Arg(10)->Arg(100)->Arg(1000);
