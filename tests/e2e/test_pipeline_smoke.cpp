#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/ProjectOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {

ColumnBatch makeBatch(MemoryManager& mm, size_t n, int64_t startA) {
    auto a = ColumnVector::make(Int64, n, &mm);
    auto b = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        a.typed<int64_t>()[i] = startA + static_cast<int64_t>(i);
        b.typed<int64_t>()[i] = static_cast<int64_t>(i * 2);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(a));
    cols.push_back(std::move(b));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

OperatorPtr buildPipeline(MemoryManager& mm,
                          ExpressionPtr predicate,
                          ExpressionPtr projection,
                          size_t batchSize,
                          size_t numBatches) {
    Schema schema({{"a", Int64}, {"b", Int64}});
    std::vector<ColumnBatch> src;
    for (size_t i = 0; i < numBatches; ++i) {
        src.push_back(makeBatch(mm, batchSize, static_cast<int64_t>(i * batchSize)));
    }

    auto scan = std::make_unique<ScanOperator>(&mm, schema, std::move(src));
    auto filter = std::make_unique<FilterOperator>(&mm, std::move(scan), std::move(predicate));

    std::vector<ExpressionPtr> projs;
    projs.push_back(std::move(projection));
    return std::make_unique<ProjectOperator>(
        &mm, std::move(filter), std::move(projs), std::vector<std::string>{"out"});
}

std::vector<int64_t> drain(Operator& op, MemoryManager& mm) {
    std::vector<int64_t> out;
    op.open();
    while (auto batch = op.getNext()) {
        auto dense = batch->selectActive(&mm);
        const auto* p = dense.column(0).typed<int64_t>();
        for (size_t i = 0; i < dense.numRows(); ++i) out.push_back(p[i]);
    }
    op.close();
    return out;
}

} // namespace

TEST(PipelineSmoke, ScanFilterProject) {
    MemoryManager mm(64 * 1024 * 1024);

    auto pred = std::make_unique<BinaryExpr>(
        BinaryOp::Gt, std::make_unique<ColumnRef>(0), std::make_unique<Literal>(int64_t{100}));
    auto proj = std::make_unique<BinaryExpr>(
        BinaryOp::Add, std::make_unique<ColumnRef>(0), std::make_unique<ColumnRef>(1));

    auto root = buildPipeline(mm, std::move(pred), std::move(proj), 4096, 2);
    auto got  = drain(*root, mm);

    std::vector<int64_t> expected;
    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 4096; ++i) {
            int64_t a = static_cast<int64_t>(b * 4096 + i);
            int64_t bb = static_cast<int64_t>(i * 2);
            if (a > 100) expected.push_back(a + bb);
        }
    }
    EXPECT_EQ(got, expected);
}

TEST(PipelineSmoke, AllRowsFilteredOut) {
    MemoryManager mm(64 * 1024 * 1024);
    auto pred = std::make_unique<BinaryExpr>(
        BinaryOp::Gt, std::make_unique<ColumnRef>(0),
        std::make_unique<Literal>(std::numeric_limits<int64_t>::max() / 2));
    auto proj = std::make_unique<BinaryExpr>(
        BinaryOp::Add, std::make_unique<ColumnRef>(0), std::make_unique<ColumnRef>(1));

    auto root = buildPipeline(mm, std::move(pred), std::move(proj), 4096, 2);
    auto got  = drain(*root, mm);
    EXPECT_TRUE(got.empty());
}

TEST(PipelineSmoke, NoOpFilter) {
    MemoryManager mm(64 * 1024 * 1024);
    auto pred = std::make_unique<BinaryExpr>(
        BinaryOp::Gt, std::make_unique<ColumnRef>(0), std::make_unique<Literal>(int64_t{-1}));
    auto proj = std::make_unique<BinaryExpr>(
        BinaryOp::Add, std::make_unique<ColumnRef>(0), std::make_unique<ColumnRef>(1));

    auto root = buildPipeline(mm, std::move(pred), std::move(proj), 4096, 2);
    auto got  = drain(*root, mm);
    EXPECT_EQ(got.size(), 8192u);
}
