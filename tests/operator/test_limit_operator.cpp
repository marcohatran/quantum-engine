#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "memory/MemoryManager.hpp"
#include "operator/LimitOperator.hpp"
#include "operator/ScanOperator.hpp"

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

size_t drain(LimitOperator& op) {
    size_t total = 0;
    op.open();
    while (auto b = op.getNext()) total += b->activeRows();
    op.close();
    return total;
}

} // namespace

TEST(Limit, AllowsAllWhenBelowLimit) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(intBatch(mm, {1, 2, 3}));
    in.push_back(intBatch(mm, {4, 5}));
    LimitOperator op(&mm, mkScan(mm, std::move(in)), /*limit=*/100);
    EXPECT_EQ(drain(op), 5u);
}

TEST(Limit, TruncatesMidBatch) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(intBatch(mm, {1, 2, 3, 4, 5}));
    in.push_back(intBatch(mm, {6, 7}));
    LimitOperator op(&mm, mkScan(mm, std::move(in)), /*limit=*/3);
    EXPECT_EQ(drain(op), 3u);
}

TEST(Limit, StopsAfterFirstBatch) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(intBatch(mm, {1, 2, 3}));
    in.push_back(intBatch(mm, {4, 5, 6, 7}));
    LimitOperator op(&mm, mkScan(mm, std::move(in)), /*limit=*/3);
    EXPECT_EQ(drain(op), 3u);
}

TEST(Limit, Zero) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(intBatch(mm, {1, 2, 3}));
    LimitOperator op(&mm, mkScan(mm, std::move(in)), /*limit=*/0);
    EXPECT_EQ(drain(op), 0u);
}
