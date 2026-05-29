#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/ProjectOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {
ColumnBatch ab(MemoryManager& mm, size_t n) {
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

TEST(ProjectOperator, EvaluatesAddition) {
    MemoryManager mm(256 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(ab(mm, 16));
    auto scan = std::make_unique<ScanOperator>(
        &mm, Schema({{"a", Int64}, {"b", Int64}}), std::move(src));

    std::vector<ExpressionPtr> exprs;
    exprs.push_back(std::make_unique<BinaryExpr>(
        BinaryOp::Add,
        std::make_unique<ColumnRef>(0),
        std::make_unique<ColumnRef>(1)));

    ProjectOperator op(&mm, std::move(scan), std::move(exprs), {"sum"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numColumns(), 1u);
    auto* p = batch->column(0).typed<int64_t>();
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(p[i], static_cast<int64_t>(i + i * 2));
    }
}
