#include <gtest/gtest.h>

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
std::vector<ColumnBatch> mkSource(MemoryManager& mm, size_t batchSize, size_t numBatches) {
    std::vector<ColumnBatch> src;
    for (size_t b = 0; b < numBatches; ++b) {
        auto col = ColumnVector::make(Int64, batchSize, &mm);
        for (size_t i = 0; i < batchSize; ++i) {
            col.typed<int64_t>()[i] = static_cast<int64_t>(b * batchSize + i);
        }
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(col));
        src.push_back(ColumnBatch(std::move(cols), PositionList::identity(batchSize)));
    }
    return src;
}
} // namespace

TEST(FilterOperator, KeepsRowsGreaterThanFive) {
    MemoryManager mm(256 * 1024);
    auto scan = std::make_unique<ScanOperator>(&mm, Schema({{"a", Int64}}),
                                               mkSource(mm, 8, 1));
    auto pred = std::make_unique<BinaryExpr>(BinaryOp::Gt,
                                             std::make_unique<ColumnRef>(0),
                                             std::make_unique<Literal>(int64_t{5}));
    FilterOperator op(&mm, std::move(scan), std::move(pred));
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->activeRows(), 2u);  // 6 and 7
    EXPECT_EQ(batch->positions().at(0), 6);
    EXPECT_EQ(batch->positions().at(1), 7);
}

TEST(FilterOperator, AllFalseProducesEmptyActiveSet) {
    MemoryManager mm(256 * 1024);
    auto scan = std::make_unique<ScanOperator>(&mm, Schema({{"a", Int64}}),
                                               mkSource(mm, 8, 1));
    auto pred = std::make_unique<BinaryExpr>(BinaryOp::Lt,
                                             std::make_unique<ColumnRef>(0),
                                             std::make_unique<Literal>(int64_t{0}));
    FilterOperator op(&mm, std::move(scan), std::move(pred));
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->activeRows(), 0u);
}
