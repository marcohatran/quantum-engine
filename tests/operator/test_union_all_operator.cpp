#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "memory/MemoryManager.hpp"
#include "operator/ScanOperator.hpp"
#include "operator/UnionAllOperator.hpp"

using namespace quantum;

namespace {

ColumnBatch intBatch(MemoryManager& mm, const std::vector<int64_t>& v) {
    auto col = ColumnVector::make(Int64, v.size(), &mm);
    auto* p  = col.typed<int64_t>();
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    return ColumnBatch(std::move(cols), PositionList::identity(v.size()));
}

OperatorPtr mkScan(MemoryManager& mm, std::vector<ColumnBatch> batches) {
    return std::make_unique<ScanOperator>(
        &mm, Schema({{"v", Int64}}), std::move(batches));
}

std::vector<int64_t> drain(Operator& op) {
    std::vector<int64_t> out;
    op.open();
    while (auto b = op.getNext()) {
        const auto* v = b->column(0).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) out.push_back(v[i]);
    }
    op.close();
    return out;
}

} // namespace

TEST(UnionAll, ConcatenatesTwoChildren) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> a; a.push_back(intBatch(mm, {1, 2, 3}));
    std::vector<ColumnBatch> b; b.push_back(intBatch(mm, {4, 5}));
    std::vector<OperatorPtr> kids;
    kids.push_back(mkScan(mm, std::move(a)));
    kids.push_back(mkScan(mm, std::move(b)));
    UnionAllOperator op(&mm, std::move(kids));
    EXPECT_EQ(drain(op), (std::vector<int64_t>{1, 2, 3, 4, 5}));
}

TEST(UnionAll, ThreeChildrenMultipleBatchesPerChild) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> a;
    a.push_back(intBatch(mm, {1, 2}));
    a.push_back(intBatch(mm, {3}));
    std::vector<ColumnBatch> b;
    b.push_back(intBatch(mm, {10, 20, 30}));
    std::vector<ColumnBatch> c;
    c.push_back(intBatch(mm, {100}));
    std::vector<OperatorPtr> kids;
    kids.push_back(mkScan(mm, std::move(a)));
    kids.push_back(mkScan(mm, std::move(b)));
    kids.push_back(mkScan(mm, std::move(c)));
    UnionAllOperator op(&mm, std::move(kids));
    EXPECT_EQ(drain(op),
              (std::vector<int64_t>{1, 2, 3, 10, 20, 30, 100}));
}
