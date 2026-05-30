#include <gtest/gtest.h>

#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

TEST(StringBuilder, BuildsBasic) {
    MemoryManager mm(64 * 1024);
    StringBuilder b(&mm);
    b.append("hello");
    b.append("world");
    b.append("!");
    auto v = b.build();

    ASSERT_EQ(v.numRows(), 3u);
    EXPECT_EQ(v.stringAt(0), "hello");
    EXPECT_EQ(v.stringAt(1), "world");
    EXPECT_EQ(v.stringAt(2), "!");
    for (size_t i = 0; i < 3; ++i) EXPECT_TRUE(v.validity().isValid(i));
}

TEST(StringBuilder, AppendNullProducesEmptyStringAndInvalid) {
    MemoryManager mm(64 * 1024);
    StringBuilder b(&mm);
    b.append("ok");
    b.appendNull();
    b.append("after");
    auto v = b.build();

    ASSERT_EQ(v.numRows(), 3u);
    EXPECT_EQ(v.stringAt(0), "ok");
    EXPECT_EQ(v.stringAt(1).size(), 0u);
    EXPECT_FALSE(v.validity().isValid(1));
    EXPECT_EQ(v.stringAt(2), "after");
}

TEST(StringBuilder, EmptyBuild) {
    MemoryManager mm(64 * 1024);
    StringBuilder b(&mm);
    auto v = b.build();
    EXPECT_EQ(v.numRows(), 0u);
}

TEST(StringBuilder, MixedLengths) {
    MemoryManager mm(64 * 1024);
    StringBuilder b(&mm);
    b.append("a");
    b.append("");
    b.append("abcdefghij");
    auto v = b.build();
    EXPECT_EQ(v.stringAt(0), "a");
    EXPECT_EQ(v.stringAt(1), "");
    EXPECT_EQ(v.stringAt(2), "abcdefghij");
}
