#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "plan/PlanProto.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

std::string tempPath(const std::string& hint) {
    auto dir  = std::filesystem::temp_directory_path();
    return (dir / (hint + std::to_string(std::random_device{}()) + ".parquet")).string();
}

void writeKvParquet(const std::string&          path,
                    const std::vector<int64_t>& ks,
                    const std::vector<int64_t>& vs) {
    arrow::Int64Builder kb, vb;
    for (auto x : ks) kb.Append(x).ok();
    for (auto x : vs) vb.Append(x).ok();
    std::shared_ptr<arrow::Array> ka, va;
    kb.Finish(&ka).ok();
    vb.Finish(&va).ok();
    auto schema = arrow::schema({
        arrow::field("k", arrow::int64()),
        arrow::field("v", arrow::int64()),
    });
    auto table = arrow::Table::Make(schema, {ka, va});
    auto out_r = arrow::io::FileOutputStream::Open(path);
    ASSERT_TRUE(out_r.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), out_r.ValueOrDie()).ok());
}

} // namespace

TEST(FileScanPlan, ParquetFileScanThroughHashAgg) {
    auto path = tempPath("filescan_plan_");
    writeKvParquet(path, {1, 1, 2, 2, 3}, {10, 20, 30, 40, 50});

    // Build:   HashAgg(group=[col 0], aggs=[Sum(col 1)],  FileScan(parquet, path))
    auto fs = FileScanNode{};
    fs.paths.push_back(path);
    fs.format = "parquet";

    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = PlanNode{std::move(fs)};
    agg->group_keys.push_back(ColumnRefExpr{0});

    AggregateSpec sumSpec;
    sumSpec.kind      = AggregateKind::Sum;
    sumSpec.inputType = Int64;
    sumSpec.input     = ColumnRefExpr{1};
    agg->aggregates.push_back(std::move(sumSpec));
    agg->output_names = {"k", "sum_v"};

    PlanNode plan{std::move(agg)};

    MemoryManager mm(1 << 20);
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

    EXPECT_EQ(got, (std::vector<std::pair<int64_t, int64_t>>{
        {1, 30}, {2, 70}, {3, 50}}));

    std::filesystem::remove(path);
}

TEST(FileScanPlan, RoundTripsThroughProto) {
    FileScanNode fs;
    fs.paths = {"file:///foo/bar.parquet", "file:///foo/baz.parquet"};
    fs.format = "parquet";

    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = PlanNode{std::move(fs)};
    agg->group_keys.push_back(ColumnRefExpr{0});
    AggregateSpec c;
    c.kind = AggregateKind::CountStar;
    c.inputType = Int64;
    agg->aggregates.push_back(std::move(c));
    agg->output_names = {"k", "n"};

    PlanNode original{std::move(agg)};

    auto bytes = planToProtoBytes(original);
    auto round = planFromProtoBytes(bytes);

    // The shape we care about: outer is HashAgg, inner is FileScanNode with
    // matching paths + format.
    auto& agg2 = *std::get<std::unique_ptr<HashAggregateNode>>(round);
    auto& fs2  = std::get<FileScanNode>(agg2.child);
    EXPECT_EQ(fs2.format, "parquet");
    EXPECT_EQ(fs2.paths.size(), 2u);
    EXPECT_EQ(fs2.paths[0], "file:///foo/bar.parquet");
    EXPECT_EQ(fs2.paths[1], "file:///foo/baz.parquet");
    EXPECT_FALSE(fs2.schema.has_value());
    EXPECT_EQ(agg2.aggregates.front().kind, AggregateKind::CountStar);
}

TEST(FileScanPlan, MultiFileParquetConcatenatesThroughHashAgg) {
    auto p1 = tempPath("filescan_multi_a_");
    auto p2 = tempPath("filescan_multi_b_");
    writeKvParquet(p1, {1, 2}, {10, 20});
    writeKvParquet(p2, {1, 3}, {30, 40});

    FileScanNode fs;
    fs.paths  = {p1, p2};
    fs.format = "parquet";

    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = PlanNode{std::move(fs)};
    agg->group_keys.push_back(ColumnRefExpr{0});
    AggregateSpec sumSpec;
    sumSpec.kind      = AggregateKind::Sum;
    sumSpec.inputType = Int64;
    sumSpec.input     = ColumnRefExpr{1};
    agg->aggregates.push_back(std::move(sumSpec));
    agg->output_names = {"k", "sum_v"};

    PlanNode plan{std::move(agg)};

    MemoryManager mm(1 << 20);
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

    // Across both files: k=1 -> 10+30=40, k=2 -> 20, k=3 -> 40.
    EXPECT_EQ(got, (std::vector<std::pair<int64_t, int64_t>>{
        {1, 40}, {2, 20}, {3, 40}}));

    std::filesystem::remove(p1);
    std::filesystem::remove(p2);
}

TEST(FileScanPlan, UnknownFormatRejects) {
    FileScanNode fs;
    fs.paths = {"/some/path"};
    fs.format = "orc";
    PlanNode plan{std::move(fs)};

    MemoryManager mm(1 << 20);
    PlanBuilder pb(&mm);
    SourceCatalog catalog;
    EXPECT_THROW(pb.build(plan, catalog), QuantumError);
}

TEST(FileScanPlan, PartitionedParquetSynthesizesConstantColumns) {
    // Simulate two "partitions": year=2024 has one file, year=2025 has another.
    auto p1 = tempPath("filescan_part_a_");
    auto p2 = tempPath("filescan_part_b_");
    writeKvParquet(p1, {1, 2}, {10, 20});
    writeKvParquet(p2, {3, 4}, {30, 40});

    FileScanNode fs;
    fs.paths  = {p1, p2};
    fs.format = "parquet";
    fs.partition_columns = Schema(std::vector<Field>{{"year", Int64}});
    fs.partition_values  = {
        { LiteralExpr{int64_t{2024}} },
        { LiteralExpr{int64_t{2025}} }
    };

    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = PlanNode{std::move(fs)};
    // GROUP BY the synthesized partition col `year` (positionally col 2 — after
    // file's k (col 0) and v (col 1)).
    agg->group_keys.push_back(ColumnRefExpr{2});
    AggregateSpec sumSpec;
    sumSpec.kind      = AggregateKind::Sum;
    sumSpec.inputType = Int64;
    sumSpec.input     = ColumnRefExpr{1};
    agg->aggregates.push_back(std::move(sumSpec));
    agg->output_names = {"year", "sum_v"};

    PlanNode plan{std::move(agg)};

    MemoryManager mm(1 << 20);
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

    // year=2024 -> 10+20=30; year=2025 -> 30+40=70.
    EXPECT_EQ(got, (std::vector<std::pair<int64_t, int64_t>>{
        {2024, 30}, {2025, 70}}));

    std::filesystem::remove(p1);
    std::filesystem::remove(p2);
}
