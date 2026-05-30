#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/DictStringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch dictBatch(MemoryManager& mm, const std::vector<std::string>& vals) {
    DictStringBuilder b(&mm);
    for (const auto& s : vals) b.append(s);
    std::vector<ColumnVector> cols;
    cols.push_back(b.build());
    return ColumnBatch(std::move(cols), PositionList::identity(vals.size()));
}

ColumnBatch twoDictBatch(MemoryManager& mm,
                         const std::vector<std::string>& a,
                         const std::vector<std::string>& b) {
    DictStringBuilder ba(&mm), bb(&mm);
    for (const auto& s : a) ba.append(s);
    for (const auto& s : b) bb.append(s);
    std::vector<ColumnVector> cols;
    cols.push_back(ba.build());
    cols.push_back(bb.build());
    return ColumnBatch(std::move(cols), PositionList::identity(a.size()));
}

} // namespace

TEST(DictCompare, EqColumnVsLiteralPresent) {
    MemoryManager mm(64 * 1024);
    auto batch = dictBatch(mm, {"alice", "bob", "alice", "carol"});
    BinaryExpr eq(BinaryOp::Eq,
                  std::make_unique<ColumnRef>(0),
                  std::make_unique<Literal>(std::string{"alice"}));
    auto r = eq.eval(batch, &mm);
    EXPECT_EQ(r.typed<uint8_t>()[0], 1);
    EXPECT_EQ(r.typed<uint8_t>()[1], 0);
    EXPECT_EQ(r.typed<uint8_t>()[2], 1);
    EXPECT_EQ(r.typed<uint8_t>()[3], 0);
}

TEST(DictCompare, EqColumnVsLiteralAbsentYieldsAllFalse) {
    MemoryManager mm(64 * 1024);
    auto batch = dictBatch(mm, {"alice", "bob", "carol"});
    BinaryExpr eq(BinaryOp::Eq,
                  std::make_unique<ColumnRef>(0),
                  std::make_unique<Literal>(std::string{"dan"}));
    auto r = eq.eval(batch, &mm);
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(r.typed<uint8_t>()[i], 0);
}

TEST(DictCompare, EqTwoDictColumnsWithDisjointDicts) {
    MemoryManager mm(64 * 1024);
    auto batch = twoDictBatch(mm,
                              {"alice", "bob",   "carol"},
                              {"alice", "carol", "dan"});
    BinaryExpr eq(BinaryOp::Eq,
                  std::make_unique<ColumnRef>(0),
                  std::make_unique<ColumnRef>(1));
    auto r = eq.eval(batch, &mm);
    EXPECT_EQ(r.typed<uint8_t>()[0], 1);  // alice == alice
    EXPECT_EQ(r.typed<uint8_t>()[1], 0);  // bob   != carol
    EXPECT_EQ(r.typed<uint8_t>()[2], 0);  // carol != dan
}

TEST(DictCompare, EqTwoDictColumnsAllMatch) {
    MemoryManager mm(64 * 1024);
    auto batch = twoDictBatch(mm,
                              {"x", "y", "z"},
                              {"x", "y", "z"});
    BinaryExpr eq(BinaryOp::Eq,
                  std::make_unique<ColumnRef>(0),
                  std::make_unique<ColumnRef>(1));
    auto r = eq.eval(batch, &mm);
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(r.typed<uint8_t>()[i], 1);
}

TEST(DictCompare, LtFallsBackToStringCompare) {
    MemoryManager mm(64 * 1024);
    // Both sides dict-encoded; Lt has no fast path, so this verifies the
    // fallback through stringAt still produces correct lexicographic order.
    auto batch = twoDictBatch(mm,
                              {"apple", "banana", "cherry"},
                              {"banana", "banana", "banana"});
    BinaryExpr lt(BinaryOp::Lt,
                  std::make_unique<ColumnRef>(0),
                  std::make_unique<ColumnRef>(1));
    auto r = lt.eval(batch, &mm);
    EXPECT_EQ(r.typed<uint8_t>()[0], 1);  // apple < banana
    EXPECT_EQ(r.typed<uint8_t>()[1], 0);  // banana < banana: false
    EXPECT_EQ(r.typed<uint8_t>()[2], 0);  // cherry < banana: false
}
