#include <gtest/gtest.h>

#include "common/DataType.hpp"

using namespace quantum;

TEST(DataType, FixedWidth) {
    EXPECT_EQ(Int32.fixedWidth(),   4);
    EXPECT_EQ(Int64.fixedWidth(),   8);
    EXPECT_EQ(Float64.fixedWidth(), 8);
    EXPECT_EQ(Bool.fixedWidth(),    1);
    EXPECT_EQ(DataType(TypeId::STRING).fixedWidth(), -1);
}

TEST(DataType, IsNumeric) {
    EXPECT_TRUE(Int32.isNumeric());
    EXPECT_TRUE(Int64.isNumeric());
    EXPECT_TRUE(Float64.isNumeric());
    EXPECT_FALSE(Bool.isNumeric());
}

TEST(DataType, ToString) {
    EXPECT_EQ(Int64.toString(),   "INT64");
    EXPECT_EQ(Float64.toString(), "FLOAT64");
}

TEST(DataType, Equality) {
    EXPECT_EQ(Int64, DataType(TypeId::INT64));
    EXPECT_NE(Int64, Int32);
}

TEST(Schema, BasicConstruction) {
    Schema s({{"a", Int64}, {"b", Float64}});
    EXPECT_EQ(s.numFields(), 2u);
    EXPECT_EQ(s.field(0).name, "a");
    EXPECT_EQ(s.field(1).type, Float64);
}
