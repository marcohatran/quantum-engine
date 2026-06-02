#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/ScanOperator.hpp"
#include "operator/SortOperator.hpp"

using namespace quantum;

namespace {

ColumnBatch kvBatch(MemoryManager& mm,
                    const std::vector<int64_t>& k,
                    const std::vector<int64_t>& v) {
    auto kc = ColumnVector::make(Int64, k.size(), &mm);
    auto vc = ColumnVector::make(Int64, v.size(), &mm);
    for (size_t i = 0; i < k.size(); ++i) {
        kc.typed<int64_t>()[i] = k[i];
        vc.typed<int64_t>()[i] = v[i];
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(kc));
    cols.push_back(std::move(vc));
    return ColumnBatch(std::move(cols), PositionList::identity(k.size()));
}

OperatorPtr mkScan(MemoryManager& mm, std::vector<ColumnBatch> batches) {
    return std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(batches));
}

std::vector<std::pair<int64_t,int64_t>> drainKV(Operator& op) {
    std::vector<std::pair<int64_t,int64_t>> out;
    op.open();
    while (auto b = op.getNext()) {
        const auto* k = b->column(0).typed<int64_t>();
        const auto* v = b->column(1).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) out.emplace_back(k[i], v[i]);
    }
    op.close();
    return out;
}

} // namespace

TEST(Sort, AscendingByFirstKey) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(kvBatch(mm, {3, 1, 5, 2}, {30, 10, 50, 20}));
    in.push_back(kvBatch(mm, {4},          {40}));

    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    SortOperator op(&mm, mkScan(mm, std::move(in)),
                    std::move(keys),
                    /*ascending=*/{true},
                    /*nullsFirst=*/{true});

    auto got = drainKV(op);
    using P = std::pair<int64_t,int64_t>;
    EXPECT_EQ(got, std::vector<P>({{1,10},{2,20},{3,30},{4,40},{5,50}}));
}

TEST(Sort, DescendingBySecondKey) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(kvBatch(mm, {1,1,1,2,2}, {30, 10, 20, 50, 40}));

    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    keys.push_back(std::make_unique<ColumnRef>(1));
    SortOperator op(&mm, mkScan(mm, std::move(in)),
                    std::move(keys),
                    /*ascending=*/{true, false},
                    /*nullsFirst=*/{true, false});

    auto got = drainKV(op);
    using P = std::pair<int64_t,int64_t>;
    // k=1 group sorted by v desc; k=2 group sorted by v desc; outer k asc.
    EXPECT_EQ(got, std::vector<P>({{1,30},{1,20},{1,10},{2,50},{2,40}}));
}

TEST(Sort, EmptyInput) {
    MemoryManager mm(64 * 1024);
    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    SortOperator op(&mm, mkScan(mm, {}),
                    std::move(keys), {true}, {true});
    auto got = drainKV(op);
    EXPECT_EQ(got.size(), 0u);
}
