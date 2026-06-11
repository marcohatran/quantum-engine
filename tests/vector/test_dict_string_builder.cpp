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

// --- ColumnVector views (P0.1 zero-copy ColumnRef) ---

TEST(ColumnVectorView, AliasesSourceAndNeverFrees) {
    MemoryManager mm(1 << 20);
    auto src = ColumnVector::make(Int64, 4, &mm);
    src.validity().setAllValid();
    for (size_t i = 0; i < 4; ++i) src.typed<int64_t>()[i] = (int64_t)i * 10;
    src.validity().setValid(2, false);

    {
        ColumnVector v = ColumnVector::view(src);
        EXPECT_TRUE(v.isView());
        EXPECT_EQ(v.numRows(), 4u);
        EXPECT_EQ(v.typed<int64_t>(), src.typed<int64_t>());  // same memory
        EXPECT_EQ(v.typed<int64_t>()[1], 10);
        EXPECT_FALSE(v.validity().isValid(2));
    }  // view destroyed — must not release the source's budget

    // Source still intact and readable after the view died.
    EXPECT_EQ(src.typed<int64_t>()[3], 30);
    EXPECT_TRUE(src.validity().isValid(3));
}

TEST(ColumnVectorView, EnsureOwnedClonesViewsAndMovesOwned) {
    MemoryManager mm(1 << 20);
    auto src = ColumnVector::make(Int64, 2, &mm);
    src.validity().setAllValid();
    src.typed<int64_t>()[0] = 7;
    src.typed<int64_t>()[1] = 8;

    ColumnVector owned = ColumnVector::view(src).ensureOwned(&mm);
    EXPECT_FALSE(owned.isView());
    EXPECT_NE(owned.typed<int64_t>(), src.typed<int64_t>());  // distinct memory
    EXPECT_EQ(owned.typed<int64_t>()[1], 8);

    ColumnVector moved = std::move(owned).ensureOwned(&mm);
    EXPECT_FALSE(moved.isView());
    EXPECT_EQ(moved.typed<int64_t>()[0], 7);
}
