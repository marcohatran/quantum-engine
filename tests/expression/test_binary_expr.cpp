#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

namespace {
ColumnBatch twoIntColumns(MemoryManager& mm, size_t n) {
    auto a = ColumnVector::make(Int64, n, &mm);
    auto b = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        a.typed<int64_t>()[i] = static_cast<int64_t>(i);
        b.typed<int64_t>()[i] = static_cast<int64_t>(i * 2);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(a));
    cols.push_back(std::move(b));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

TEST(BinaryExpr, AddInt64Identity) {
    MemoryManager mm(16 * 1024);
    auto batch = twoIntColumns(mm, 32);
    BinaryExpr add(BinaryOp::Add,
                   std::make_unique<ColumnRef>(0),
                   std::make_unique<ColumnRef>(1));
    auto result = add.eval(batch, &mm);
    ASSERT_EQ(result.type(), Int64);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(result.typed<int64_t>()[i], static_cast<int64_t>(i + i * 2));
    }
}

TEST(BinaryExpr, GtAgainstLiteral) {
    MemoryManager mm(16 * 1024);
    auto batch = twoIntColumns(mm, 8);
    BinaryExpr gt(BinaryOp::Gt,
                  std::make_unique<ColumnRef>(0),
                  std::make_unique<Literal>(int64_t{3}));
    auto result = gt.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);
    auto* p = result.typed<uint8_t>();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(p[i] != 0, i > 3) << "i=" << i;
    }
}

TEST(BinaryExpr, NullPropagation) {
    MemoryManager mm(16 * 1024);
    auto batch = twoIntColumns(mm, 8);
    batch.column(0).validity().setValid(2, false);
    batch.column(1).validity().setValid(5, false);

    BinaryExpr add(BinaryOp::Add,
                   std::make_unique<ColumnRef>(0),
                   std::make_unique<ColumnRef>(1));
    auto result = add.eval(batch, &mm);
    EXPECT_FALSE(result.validity().isValid(2));
    EXPECT_FALSE(result.validity().isValid(5));
    EXPECT_TRUE (result.validity().isValid(0));
}

#include "vector/StringBuilder.hpp"

TEST(BinaryExpr, StringEqAndLtAndGt) {
    MemoryManager mm(64 * 1024);

    StringBuilder b(&mm);
    b.append("apple"); b.append("banana"); b.append("cherry");
    auto names = b.build();
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(names));
    ColumnBatch batch(std::move(cols), PositionList::identity(3));

    auto checkOp = [&](BinaryOp op, const std::string& lit, std::vector<uint8_t> want) {
        BinaryExpr expr(op,
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<Literal>(lit));
        auto r = expr.eval(batch, &mm);
        ASSERT_EQ(r.type(), Bool);
        for (size_t i = 0; i < want.size(); ++i) {
            EXPECT_EQ(r.typed<uint8_t>()[i], want[i]) << "op=" << static_cast<int>(op) << " i=" << i;
        }
    };

    checkOp(BinaryOp::Eq, "banana", {0, 1, 0});
    checkOp(BinaryOp::Lt, "banana", {1, 0, 0});  // "apple" < "banana" only
    checkOp(BinaryOp::Gt, "banana", {0, 0, 1});  // "cherry" > "banana" only
}

TEST(BinaryExpr, GteAgainstLiteralInt64) {
    MemoryManager mm(16 * 1024);
    auto batch = twoIntColumns(mm, 8);
    BinaryExpr gte(BinaryOp::Gte,
                   std::make_unique<ColumnRef>(0),
                   std::make_unique<Literal>(int64_t{3}));
    auto result = gte.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);
    auto* p = result.typed<uint8_t>();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(p[i] != 0, i >= 3) << "i=" << i;
    }
}

TEST(BinaryExpr, LteAgainstLiteralInt64) {
    MemoryManager mm(16 * 1024);
    auto batch = twoIntColumns(mm, 8);
    BinaryExpr lte(BinaryOp::Lte,
                   std::make_unique<ColumnRef>(0),
                   std::make_unique<Literal>(int64_t{3}));
    auto result = lte.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);
    auto* p = result.typed<uint8_t>();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(p[i] != 0, i <= 3) << "i=" << i;
    }
}

TEST(BinaryExpr, OrBooleanColumns) {
    MemoryManager mm(16 * 1024);
    auto batch = twoIntColumns(mm, 8);  // col 0 = 0..7, col 1 = 2*(0..7)
    //  WHERE col0 = 3  OR  col1 = 4    (col1 = 4 when i = 2)
    BinaryExpr eq3(BinaryOp::Eq,
                   std::make_unique<ColumnRef>(0),
                   std::make_unique<Literal>(int64_t{3}));
    BinaryExpr eq4(BinaryOp::Eq,
                   std::make_unique<ColumnRef>(1),
                   std::make_unique<Literal>(int64_t{4}));
    BinaryExpr orExpr(BinaryOp::Or,
                      std::make_unique<BinaryExpr>(std::move(eq3)),
                      std::make_unique<BinaryExpr>(std::move(eq4)));
    auto result = orExpr.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);
    auto* p = result.typed<uint8_t>();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(p[i] != 0, i == 2 || i == 3) << "i=" << i;
    }
}

TEST(BinaryExpr, OrKleeneTruthDominatesNull) {
    // T OR NULL = T -- the row must be kept even if the right side is NULL.
    MemoryManager mm(16 * 1024);
    auto lhs = ColumnVector::make(Bool, 4, &mm);
    auto rhs = ColumnVector::make(Bool, 4, &mm);
    lhs.validity().setAllValid();
    rhs.validity().setAllValid();
    auto* la = lhs.typed<uint8_t>();
    auto* ra = rhs.typed<uint8_t>();
    la[0] = 1; la[1] = 0; la[2] = 0; la[3] = 1;
    ra[0] = 0; ra[1] = 0; ra[2] = 0; ra[3] = 0;
    rhs.validity().setValid(0, false);  // T OR NULL  -> T (valid)
    rhs.validity().setValid(1, false);  // F OR NULL  -> NULL (invalid)
    lhs.validity().setValid(2, false);  // NULL OR F  -> NULL (invalid)
    rhs.validity().setValid(3, false);  // T OR NULL  -> T (valid)

    std::vector<ColumnVector> cols;
    cols.push_back(std::move(lhs)); cols.push_back(std::move(rhs));
    ColumnBatch batch(std::move(cols), PositionList::identity(4));

    BinaryExpr orExpr(BinaryOp::Or,
                      std::make_unique<ColumnRef>(0),
                      std::make_unique<ColumnRef>(1));
    auto result = orExpr.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);
    EXPECT_TRUE (result.validity().isValid(0));
    EXPECT_EQ   (result.typed<uint8_t>()[0], 1U);
    EXPECT_FALSE(result.validity().isValid(1));
    EXPECT_FALSE(result.validity().isValid(2));
    EXPECT_TRUE (result.validity().isValid(3));
    EXPECT_EQ   (result.typed<uint8_t>()[3], 1U);
}

TEST(BinaryExpr, AndKleeneFalseDominatesNull) {
    // F AND NULL = F -- the row must be excluded by Filter even if the
    // other side is NULL.
    MemoryManager mm(16 * 1024);
    auto lhs = ColumnVector::make(Bool, 4, &mm);
    auto rhs = ColumnVector::make(Bool, 4, &mm);
    lhs.validity().setAllValid();
    rhs.validity().setAllValid();
    auto* la = lhs.typed<uint8_t>();
    auto* ra = rhs.typed<uint8_t>();
    la[0] = 1; la[1] = 0; la[2] = 1; la[3] = 1;
    ra[0] = 1; ra[1] = 1; ra[2] = 0; ra[3] = 0;
    rhs.validity().setValid(0, false);  // T AND NULL  -> NULL
    rhs.validity().setValid(2, false);  // T AND NULL  -> NULL
    rhs.validity().setValid(3, false);  // T AND NULL  -> NULL ... oops same shape
    // Override one to make F AND NULL = F (valid, false)
    la[3] = 0;

    std::vector<ColumnVector> cols;
    cols.push_back(std::move(lhs)); cols.push_back(std::move(rhs));
    ColumnBatch batch(std::move(cols), PositionList::identity(4));

    BinaryExpr andExpr(BinaryOp::And,
                       std::make_unique<ColumnRef>(0),
                       std::make_unique<ColumnRef>(1));
    auto result = andExpr.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);
    EXPECT_FALSE(result.validity().isValid(0));    // T AND NULL = NULL
    EXPECT_TRUE (result.validity().isValid(1));    // F AND T    = F
    EXPECT_EQ   (result.typed<uint8_t>()[1], 0U);
    EXPECT_TRUE (result.validity().isValid(3));    // F AND NULL = F (false dominates)
    EXPECT_EQ   (result.typed<uint8_t>()[3], 0U);
}
