#include <gtest/gtest.h>

#include "memory/MemoryManager.hpp"
#include "vector/ColumnVector.hpp"

using namespace quantum;

TEST(ColumnVector, AllocatesFixedWidth) {
    MemoryManager mm(4096);
    auto v = ColumnVector::make(Int64, 100, &mm);
    EXPECT_EQ(v.type(), Int64);
    EXPECT_EQ(v.numRows(), 100u);
    EXPECT_GE(v.values().size(), 100u * 8u);
}

TEST(ColumnVector, TypedAccessRoundTrip) {
    MemoryManager mm(4096);
    auto v = ColumnVector::make(Int64, 16, &mm);
    int64_t* p = v.typed<int64_t>();
    for (int i = 0; i < 16; ++i) p[i] = i * 7;
    for (int i = 0; i < 16; ++i) EXPECT_EQ(p[i], i * 7);
}

TEST(ColumnVector, ValidityDefaultsAllValid) {
    MemoryManager mm(4096);
    auto v = ColumnVector::make(Float64, 50, &mm);
    EXPECT_EQ(v.validity().nullCount(), 0u);
}
