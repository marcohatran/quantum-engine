#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include "memory/MemoryManager.hpp"
#include "plan/ColumnPruning.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

// Four columns (a,b,c,d : int64); a = i, b = 2i, c = 3i, d = i % 4.
std::string writeWideParquet(int rows) {
    arrow::Int64Builder a, b, c, d;
    for (int i = 0; i < rows; ++i) {
        a.Append(i).ok();
        b.Append(2 * i).ok();
        c.Append(3 * i).ok();
        d.Append(i % 4).ok();
    }
    std::shared_ptr<arrow::Array> aArr, bArr, cArr, dArr;
    a.Finish(&aArr).ok();
    b.Finish(&bArr).ok();
    c.Finish(&cArr).ok();
    d.Finish(&dArr).ok();
    auto schema = arrow::schema({
        arrow::field("a", arrow::int64()), arrow::field("b", arrow::int64()),
        arrow::field("c", arrow::int64()), arrow::field("d", arrow::int64())});
    auto table = arrow::Table::Make(schema, {aArr, bArr, cArr, dArr});

    auto dir  = std::filesystem::temp_directory_path();
    auto path = (dir / ("pruning" + std::to_string(std::random_device{}()) +
                        ".parquet")).string();
    auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1024)
        .ok();
    return path;
}

ScalarExpr gtLit(int32_t col, int64_t lit) {
    auto cmp = std::make_unique<BinaryOpExpr>();
    cmp->op  = BinaryOpType::Gt;
    cmp->lhs = ColumnRefExpr{col};
    cmp->rhs = LiteralExpr{int64_t{lit}};
    return cmp;
}

// SELECT d, SUM(c) FROM t WHERE c > lit GROUP BY d — uses c and d, not a/b.
PlanNode aggPlan(const std::string& path, int64_t lit) {
    FileScanNode fs;
    fs.format = "parquet";
    fs.paths  = {path};

    auto filter       = std::make_unique<FilterNode>();
    filter->child     = PlanNode{std::move(fs)};
    filter->predicate = gtLit(2, lit);

    auto agg   = std::make_unique<HashAggregateNode>();
    agg->child = PlanNode{std::move(filter)};
    agg->group_keys.push_back(ColumnRefExpr{3});
    AggregateSpec sum;
    sum.kind      = AggregateKind::Sum;
    sum.inputType = Int64;
    sum.input     = ScalarExpr{ColumnRefExpr{2}};
    agg->aggregates.push_back(std::move(sum));
    agg->output_names = {"d", "sum_c"};
    return PlanNode{std::move(agg)};
}

std::vector<std::pair<int64_t, int64_t>> run(PlanNode& plan, MemoryManager& mm) {
    PlanBuilder pb(&mm);
    SourceCatalog catalog;
    auto op = pb.build(plan, catalog);
    op->open();
    std::vector<std::pair<int64_t, int64_t>> got;
    while (auto batch = op->getNext()) {
        auto dense = batch->selectActive(&mm);
        for (size_t i = 0; i < dense.numRows(); ++i) {
            got.emplace_back(dense.column(0).typed<int64_t>()[i],
                             dense.column(1).typed<int64_t>()[i]);
        }
    }
    op->close();
    std::sort(got.begin(), got.end());
    return got;
}

} // namespace

TEST(ColumnPruning, PrunesScanAndRemapsRefs) {
    PlanNode plan = aggPlan("/nonexistent.parquet", 10);
    pruneScanColumns(plan);

    auto& agg    = *std::get<std::unique_ptr<HashAggregateNode>>(plan);
    auto& filter = *std::get<std::unique_ptr<FilterNode>>(agg.child);
    auto& scan   = std::get<FileScanNode>(filter.child);

    EXPECT_EQ(scan.projected_columns, (std::vector<int32_t>{2, 3}));
    // c -> 0, d -> 1 in the pruned space.
    EXPECT_EQ(std::get<ColumnRefExpr>(agg.group_keys[0]).index, 1);
    EXPECT_EQ(std::get<ColumnRefExpr>(*agg.aggregates[0].input).index, 0);
    auto& pred = *std::get<std::unique_ptr<BinaryOpExpr>>(filter.predicate);
    EXPECT_EQ(std::get<ColumnRefExpr>(pred.lhs).index, 0);

    // Idempotent: a second pass leaves the remapped plan untouched.
    pruneScanColumns(plan);
    EXPECT_EQ(scan.projected_columns, (std::vector<int32_t>{2, 3}));
    EXPECT_EQ(std::get<ColumnRefExpr>(agg.group_keys[0]).index, 1);
}

TEST(ColumnPruning, PrunedPlanProducesSameResults) {
    const std::string path = writeWideParquet(/*rows=*/10000);
    MemoryManager mm(64 << 20);

    PlanNode plain = aggPlan(path, 600);
    auto expected  = run(plain, mm);
    ASSERT_EQ(expected.size(), 4u);  // 4 groups in d

    PlanNode pruned = aggPlan(path, 600);
    pruneScanColumns(pruned);
    EXPECT_EQ(run(pruned, mm), expected);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(ColumnPruning, CountStarKeepsFullScan) {
    // No referenced columns: pruning must not produce a zero-column scan.
    FileScanNode fs;
    fs.format  = "parquet";
    fs.paths   = {"/nonexistent.parquet"};
    auto agg   = std::make_unique<HashAggregateNode>();
    agg->child = PlanNode{std::move(fs)};
    AggregateSpec count;
    count.kind = AggregateKind::CountStar;
    agg->aggregates.push_back(std::move(count));
    agg->output_names = {"n"};
    PlanNode plan{std::move(agg)};

    pruneScanColumns(plan);
    auto& aggRef = *std::get<std::unique_ptr<HashAggregateNode>>(plan);
    EXPECT_TRUE(std::get<FileScanNode>(aggRef.child).projected_columns.empty());
}

TEST(ColumnPruning, PartitionedScanUntouched) {
    FileScanNode fs;
    fs.format = "parquet";
    fs.paths  = {"/nonexistent.parquet"};
    fs.partition_columns = Schema({{"year", Int64}});
    fs.partition_values  = {{LiteralExpr{int64_t{2024}}}};

    auto agg   = std::make_unique<HashAggregateNode>();
    agg->child = PlanNode{std::move(fs)};
    agg->group_keys.push_back(ColumnRefExpr{1});
    AggregateSpec sum;
    sum.kind      = AggregateKind::Sum;
    sum.inputType = Int64;
    sum.input     = ScalarExpr{ColumnRefExpr{0}};
    agg->aggregates.push_back(std::move(sum));
    agg->output_names = {"k", "s"};
    PlanNode plan{std::move(agg)};

    pruneScanColumns(plan);
    auto& aggRef = *std::get<std::unique_ptr<HashAggregateNode>>(plan);
    EXPECT_TRUE(std::get<FileScanNode>(aggRef.child).projected_columns.empty());
    EXPECT_EQ(std::get<ColumnRefExpr>(aggRef.group_keys[0]).index, 1);
}
