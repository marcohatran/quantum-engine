#include <gtest/gtest.h>

#include "memory/MemoryManager.hpp"
#include "vector/DictStringBuilder.hpp"

using namespace quantum;

TEST(DictStringBuilder, BasicAppend) {
    MemoryManager mm(64 * 1024);
    DictStringBuilder b(&mm);
    b.append("alice");
    b.append("bob");
    b.append("alice");
    auto v = b.build();

    ASSERT_TRUE(v.isDictEncoded());
    ASSERT_EQ(v.numRows(), 3u);
    EXPECT_EQ(v.dictSize(), 2u);   // alice, bob — alice is deduped
    EXPECT_EQ(v.stringAt(0), "alice");
    EXPECT_EQ(v.stringAt(1), "bob");
    EXPECT_EQ(v.stringAt(2), "alice");
}

TEST(DictStringBuilder, AppendNullMarksInvalid) {
    MemoryManager mm(64 * 1024);
    DictStringBuilder b(&mm);
    b.append("ok");
    b.appendNull();
    b.append("ok");
    auto v = b.build();

    ASSERT_TRUE(v.isDictEncoded());
    EXPECT_TRUE (v.validity().isValid(0));
    EXPECT_FALSE(v.validity().isValid(1));
    EXPECT_TRUE (v.validity().isValid(2));
    // Dict still has only 1 entry (the null didn't contribute).
    EXPECT_EQ(v.dictSize(), 1u);
}

TEST(DictStringBuilder, EmptyBuild) {
    MemoryManager mm(64 * 1024);
    DictStringBuilder b(&mm);
    auto v = b.build();
    EXPECT_EQ(v.numRows(), 0u);
    EXPECT_EQ(v.dictSize(), 0u);
    EXPECT_TRUE(v.isDictEncoded());
}

TEST(DictStringBuilder, AssignsConsecutiveCodes) {
    MemoryManager mm(64 * 1024);
    DictStringBuilder b(&mm);
    b.append("a");
    b.append("b");
    b.append("c");
    auto v = b.build();
    const int32_t* codes = v.values().as<int32_t>();
    EXPECT_EQ(codes[0], 0);
    EXPECT_EQ(codes[1], 1);
    EXPECT_EQ(codes[2], 2);
}
