#include <gtest/gtest.h>

#include "vector/PositionList.hpp"

using namespace quantum;

TEST(PositionList, IdentityFastPath) {
    auto p = PositionList::identity(5);
    EXPECT_TRUE(p.isIdentity());
    EXPECT_EQ(p.size(), 5u);
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(p.at(i), static_cast<int32_t>(i));
    EXPECT_TRUE(p.indices().empty());
}

TEST(PositionList, FromIndicesPath) {
    auto p = PositionList::fromIndices({0, 3, 7});
    EXPECT_FALSE(p.isIdentity());
    EXPECT_EQ(p.size(), 3u);
    EXPECT_EQ(p.at(0), 0);
    EXPECT_EQ(p.at(1), 3);
    EXPECT_EQ(p.at(2), 7);
    EXPECT_EQ(p.indices().size(), 3u);
}
