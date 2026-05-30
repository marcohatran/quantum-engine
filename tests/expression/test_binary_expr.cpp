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
