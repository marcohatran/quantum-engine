#include <gtest/gtest.h>

#include "memory/MemoryManager.hpp"
#include "vector/ValidityBitmask.hpp"

using namespace quantum;

TEST(ValidityBitmask, DefaultAllValid) {
    MemoryManager mm(4096);
    ValidityBitmask m(&mm, 200);
    for (size_t i = 0; i < 200; ++i) EXPECT_TRUE(m.isValid(i));
    EXPECT_EQ(m.nullCount(), 0u);
}

TEST(ValidityBitmask, SetAtBoundaries) {
    MemoryManager mm(4096);
    ValidityBitmask m(&mm, 128);
    for (size_t i : {0u, 7u, 8u, 9u, 63u, 64u, 65u, 127u}) {
        m.setValid(i, false);
        EXPECT_FALSE(m.isValid(i));
    }
    EXPECT_EQ(m.nullCount(), 8u);
}

TEST(ValidityBitmask, SetAllNullAndValid) {
    MemoryManager mm(4096);
    ValidityBitmask m(&mm, 100);
    m.setAllNull();
    EXPECT_EQ(m.nullCount(), 100u);
    m.setAllValid();
    EXPECT_EQ(m.nullCount(), 0u);
}

TEST(ValidityBitmask, AndInPlace) {
    MemoryManager mm(4096);
    ValidityBitmask a(&mm, 256);
    ValidityBitmask b(&mm, 256);
    a.setValid(5, false);
    b.setValid(10, false);
    a.andInPlace(b);
    EXPECT_FALSE(a.isValid(5));
    EXPECT_FALSE(a.isValid(10));
    EXPECT_TRUE (a.isValid(0));
    EXPECT_EQ(a.nullCount(), 2u);
}
