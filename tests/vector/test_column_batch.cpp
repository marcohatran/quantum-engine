#include <gtest/gtest.h>

#include <vector>

#include "memory/MemoryManager.hpp"
#include "vector/ColumnBatch.hpp"

using namespace quantum;

namespace {
ColumnBatch makeIntBatch(MemoryManager& mm, size_t n) {
    auto col = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) col.typed<int64_t>()[i] = static_cast<int64_t>(i);
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

TEST(ColumnBatch, SizeAccessors) {
    MemoryManager mm(8192);
    auto b = makeIntBatch(mm, 32);
    EXPECT_EQ(b.numColumns(), 1u);
    EXPECT_EQ(b.numRows(),    32u);
    EXPECT_EQ(b.activeRows(), 32u);
}

TEST(ColumnBatch, SelectActiveFromIndices) {
    MemoryManager mm(8192);
    auto b = makeIntBatch(mm, 32);
    b.setPositions(PositionList::fromIndices({0, 3, 5, 10}));
    auto dense = b.selectActive(&mm);
    ASSERT_EQ(dense.numRows(), 4u);
    auto* p = dense.column(0).typed<int64_t>();
    EXPECT_EQ(p[0], 0);
    EXPECT_EQ(p[1], 3);
    EXPECT_EQ(p[2], 5);
    EXPECT_EQ(p[3], 10);
}
