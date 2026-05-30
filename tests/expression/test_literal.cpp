#include <gtest/gtest.h>

#include <vector>

#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

namespace {
ColumnBatch emptyBatchOf(MemoryManager& mm, size_t n) {
    auto v = ColumnVector::make(Int64, n, &mm);
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

TEST(Literal, BroadcastsInt64) {
    MemoryManager mm(4096);
    auto batch = emptyBatchOf(mm, 8);
    Literal lit(int64_t{42});
    auto result = lit.eval(batch, &mm);
    ASSERT_EQ(result.type(), Int64);
    ASSERT_EQ(result.numRows(), 8u);
    for (size_t i = 0; i < 8; ++i) EXPECT_EQ(result.typed<int64_t>()[i], 42);
}

TEST(Literal, BroadcastsDouble) {
    MemoryManager mm(4096);
    auto batch = emptyBatchOf(mm, 4);
    Literal lit(3.5);
    auto result = lit.eval(batch, &mm);
    EXPECT_EQ(result.type(), Float64);
    for (size_t i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(result.typed<double>()[i], 3.5);
}

TEST(Literal, BroadcastsString) {
    MemoryManager mm(4096);
    auto batch = emptyBatchOf(mm, 3);
    Literal lit(std::string{"hi"});
    auto result = lit.eval(batch, &mm);
    EXPECT_EQ(result.type().id, TypeId::STRING);
    ASSERT_EQ(result.numRows(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(result.stringAt(i), "hi");
        EXPECT_TRUE(result.validity().isValid(i));
    }
}
