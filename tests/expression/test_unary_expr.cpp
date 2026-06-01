#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "expression/UnaryExpr.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/ColumnBatch.hpp"

using namespace quantum;

namespace {

ColumnBatch oneInt(MemoryManager& mm, size_t n) {
    auto col = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) col.typed<int64_t>()[i] = static_cast<int64_t>(i);
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

} // namespace

TEST(UnaryExpr, NotFlipsBoolValues) {
    MemoryManager mm(16 * 1024);
    auto batch = oneInt(mm, 8);

    // NOT (col >= 4)  ->  true for rows 0..3, false for rows 4..7.
    BinaryExpr gte(BinaryOp::Gte,
                   std::make_unique<ColumnRef>(0),
                   std::make_unique<Literal>(int64_t{4}));
    UnaryExpr notExpr(UnaryOp::Not, std::make_unique<BinaryExpr>(std::move(gte)));

    auto result = notExpr.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);
    const auto* p = result.typed<uint8_t>();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(p[i] != 0, i < 4) << "i=" << i;
    }
}

TEST(UnaryExpr, IsNullOnInt64AlwaysValid) {
    MemoryManager mm(16 * 1024);
    auto col = ColumnVector::make(Int64, 5, &mm);
    col.validity().setAllValid();
    auto* a = col.typed<int64_t>();
    a[0] = 10; a[1] = 20; a[2] = 30; a[3] = 40; a[4] = 50;
    col.validity().setValid(1, false);  // row 1 is NULL
    col.validity().setValid(3, false);  // row 3 is NULL

    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    ColumnBatch batch(std::move(cols), PositionList::identity(5));

    UnaryExpr expr(UnaryOp::IsNull, std::make_unique<ColumnRef>(0));
    auto result = expr.eval(batch, &mm);
    ASSERT_EQ(result.type(), Bool);

    // IS NULL is always valid (SQL: IS NULL never returns NULL).
    for (size_t i = 0; i < 5; ++i) EXPECT_TRUE(result.validity().isValid(i));
    EXPECT_EQ(result.typed<uint8_t>()[0], 0U);  // 10  IS NULL = false
    EXPECT_EQ(result.typed<uint8_t>()[1], 1U);  // NULL IS NULL = true
    EXPECT_EQ(result.typed<uint8_t>()[2], 0U);
    EXPECT_EQ(result.typed<uint8_t>()[3], 1U);
    EXPECT_EQ(result.typed<uint8_t>()[4], 0U);
}

TEST(UnaryExpr, NotPreservesNullValidity) {
    // NULL value: NOT NULL must remain NULL (the validity bit stays cleared).
    MemoryManager mm(16 * 1024);
    auto col = ColumnVector::make(Bool, 4, &mm);
    col.validity().setAllValid();
    auto* a = col.typed<uint8_t>();
    a[0] = 0; a[1] = 1; a[2] = 0; a[3] = 1;
    col.validity().setValid(1, false);  // row 1 is NULL
    col.validity().setValid(3, false);  // row 3 is NULL

    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    ColumnBatch batch(std::move(cols), PositionList::identity(4));

    UnaryExpr notExpr(UnaryOp::Not, std::make_unique<ColumnRef>(0));
    auto result = notExpr.eval(batch, &mm);

    EXPECT_TRUE (result.validity().isValid(0));
    EXPECT_FALSE(result.validity().isValid(1));   // NULL stays NULL
    EXPECT_TRUE (result.validity().isValid(2));
    EXPECT_FALSE(result.validity().isValid(3));   // NULL stays NULL
    EXPECT_EQ   (result.typed<uint8_t>()[0], 1U); // !false  = true
    EXPECT_EQ   (result.typed<uint8_t>()[2], 1U); // !false  = true
}
