#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "expression/LikeExpr.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch stringBatch(MemoryManager& mm,
                        const std::vector<std::string>& vals,
                        const std::vector<bool>& validity = {}) {
    StringBuilder sb(&mm);
    for (size_t i = 0; i < vals.size(); ++i) {
        if (!validity.empty() && !validity[i]) sb.appendNull();
        else                                    sb.append(vals[i]);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(sb.build());
    return ColumnBatch(std::move(cols), PositionList::identity(vals.size()));
}

std::vector<uint8_t> runLike(MemoryManager& mm,
                             const std::vector<std::string>& vals,
                             const std::string& pattern,
                             const std::vector<bool>& validity = {}) {
    auto batch = stringBatch(mm, vals, validity);
    LikeExpr expr(std::make_unique<ColumnRef>(0), pattern);
    auto out = expr.eval(batch, &mm);
    std::vector<uint8_t> result(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        result[i] = out.validity().isValid(i) ? out.typed<uint8_t>()[i] : 0xFF;
    }
    return result;
}

} // namespace

TEST(LikeExpr, ExactMatch) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"apple", "banana", "apple"}, "apple");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 0, 1}));
}

TEST(LikeExpr, PrefixWithPercent) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"foobar", "foo", "afoo"}, "foo%");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 1, 0}));
}

TEST(LikeExpr, SuffixWithPercent) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"foobar", "bar", "bartender"}, "%bar");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 1, 0}));
}

TEST(LikeExpr, ContainsWithDoublePercent) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"xxabcxx", "abc", "abcd", "ddabcd"}, "%abc%");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 1, 1, 1}));
}

TEST(LikeExpr, UnderscoreMatchesSingle) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"abc", "axc", "ac", "abcd"}, "a_c");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 1, 0, 0}));
}

TEST(LikeExpr, MixedPercentAndUnderscore) {
    MemoryManager mm(64 * 1024);
    // Pattern "a%c_" = 'a', then any substring, then 'c', then exactly one
    // more char. So "ac" and "abc" both fail (nothing after the c).
    auto got = runLike(mm, {"abcd", "axcd", "ac", "abc"}, "a%c_");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 1, 0, 0}));
}

TEST(LikeExpr, NullInputPropagates) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"abc", "x", "abc"}, "abc", {true, false, true});
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 0xFF, 1}));  // 0xFF = invalid
}

TEST(LikeExpr, EmptyPatternMatchesEmptyOnly) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"", "a"}, "");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 0}));
}

TEST(LikeExpr, OnlyPercentMatchesAll) {
    MemoryManager mm(64 * 1024);
    auto got = runLike(mm, {"", "abc", "x"}, "%");
    EXPECT_EQ(got, (std::vector<uint8_t>{1, 1, 1}));
}

#include "vector/DictStringBuilder.hpp"

TEST(LikeExpr, DictEncodedFastPath) {
    MemoryManager mm(64 * 1024);
    // A dict-encoded column where the same strings appear multiple times:
    // pattern is evaluated against each dict entry once, then projected.
    DictStringBuilder b(&mm);
    b.append("foobar"); b.append("baz"); b.append("foo");
    b.append("foobar"); b.append("nope"); b.append("baz");
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    ColumnBatch batch(std::move(cols), PositionList::identity(6));

    LikeExpr like(std::make_unique<ColumnRef>(0), "foo%");
    auto r = like.eval(batch, &mm);
    EXPECT_EQ(r.typed<uint8_t>()[0], 1);  // foobar
    EXPECT_EQ(r.typed<uint8_t>()[1], 0);  // baz
    EXPECT_EQ(r.typed<uint8_t>()[2], 1);  // foo
    EXPECT_EQ(r.typed<uint8_t>()[3], 1);  // foobar
    EXPECT_EQ(r.typed<uint8_t>()[4], 0);  // nope
    EXPECT_EQ(r.typed<uint8_t>()[5], 0);  // baz
}

TEST(LikeExpr, DictEncodedWithNulls) {
    MemoryManager mm(64 * 1024);
    DictStringBuilder b(&mm);
    b.append("yes");
    b.appendNull();
    b.append("no");
    b.append("yes");
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    ColumnBatch batch(std::move(cols), PositionList::identity(4));

    LikeExpr like(std::make_unique<ColumnRef>(0), "yes");
    auto r = like.eval(batch, &mm);
    EXPECT_TRUE(r.validity().isValid(0));   EXPECT_EQ(r.typed<uint8_t>()[0], 1);
    EXPECT_FALSE(r.validity().isValid(1));                                     // null
    EXPECT_TRUE(r.validity().isValid(2));   EXPECT_EQ(r.typed<uint8_t>()[2], 0);
    EXPECT_TRUE(r.validity().isValid(3));   EXPECT_EQ(r.typed<uint8_t>()[3], 1);
}
