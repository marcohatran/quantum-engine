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

#include "io/ParquetReader.hpp"
#include "memory/MemoryManager.hpp"
#include "plan/ColumnPruning.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "plan/PredicatePushdown.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

// (k:int64, v:int64), v = row index ascending, written in `chunk`-row row
// groups so min/max statistics partition cleanly: row group g covers
// v in [g*chunk, (g+1)*chunk).
std::string writeChunkedParquet(int rows, int chunk) {
    arrow::Int64Builder k, v;
    for (int i = 0; i < rows; ++i) {
        k.Append(i % 7).ok();
        v.Append(i).ok();
    }
    std::shared_ptr<arrow::Array> kArr, vArr;
    k.Finish(&kArr).ok();
    v.Finish(&vArr).ok();
    auto schema = arrow::schema({arrow::field("k", arrow::int64()),
                                 arrow::field("v", arrow::int64())});
    auto table = arrow::Table::Make(schema, {kArr, vArr});

    auto dir  = std::filesystem::temp_directory_path();
    auto path = (dir / ("statskip" + std::to_string(std::random_device{}()) +
                        ".parquet")).string();
    auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out,
                               static_cast<int64_t>(chunk)).ok();
    return path;
}

ScalarExpr cmp(BinaryOpType op, int32_t col, int64_t lit, bool litFirst = false) {
    auto e = std::make_unique<BinaryOpExpr>();
    e->op = op;
    if (litFirst) {
        e->lhs = LiteralExpr{int64_t{lit}};
        e->rhs = ColumnRefExpr{col};
    } else {
        e->lhs = ColumnRefExpr{col};
        e->rhs = LiteralExpr{int64_t{lit}};
    }
    return e;
}

ScalarExpr andExpr(ScalarExpr l, ScalarExpr r) {
    auto e = std::make_unique<BinaryOpExpr>();
    e->op  = BinaryOpType::And;
    e->lhs = std::move(l);
    e->rhs = std::move(r);
    return e;
}

void rm(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // namespace

TEST(PredicatePushdown, ExtractsConjunctsAndKeepsFilter) {
    FileScanNode fs;
    fs.format = "parquet";
    fs.paths  = {"/nonexistent.parquet"};

    auto filter       = std::make_unique<FilterNode>();
    filter->child     = PlanNode{std::move(fs)};
    // (v > 10 AND 100 > v) AND (k LIKE-ish ineligible: k+1 > 3)
    auto kPlus = std::make_unique<BinaryOpExpr>();
    kPlus->op  = BinaryOpType::Add;
    kPlus->lhs = ColumnRefExpr{0};
    kPlus->rhs = LiteralExpr{int64_t{1}};
    auto inel  = std::make_unique<BinaryOpExpr>();
    inel->op   = BinaryOpType::Gt;
    inel->lhs  = std::move(kPlus);
    inel->rhs  = LiteralExpr{int64_t{3}};
    filter->predicate = andExpr(
        andExpr(cmp(BinaryOpType::Gt, 1, 10),
                cmp(BinaryOpType::Gt, 1, 100, /*litFirst=*/true)),
        ScalarExpr{std::move(inel)});
    PlanNode plan{std::move(filter)};

    pushScanPredicates(plan);

    auto& f  = *std::get<std::unique_ptr<FilterNode>>(plan);
    auto& sc = std::get<FileScanNode>(f.child);
    ASSERT_EQ(sc.stat_predicates.size(), 2u);
    EXPECT_EQ(sc.stat_predicates[0].column, 1);
    EXPECT_EQ(sc.stat_predicates[0].op, BinaryOpType::Gt);
    EXPECT_EQ(std::get<int64_t>(sc.stat_predicates[0].literal.value), 10);
    // `100 > v` flips to `v < 100`.
    EXPECT_EQ(sc.stat_predicates[1].column, 1);
    EXPECT_EQ(sc.stat_predicates[1].op, BinaryOpType::Lt);
    EXPECT_EQ(std::get<int64_t>(sc.stat_predicates[1].literal.value), 100);
}

TEST(ParquetReader, StatsPredicateSkipsRowGroups) {
    // 10 row groups of 1000 rows; v ascending => only the last 2 groups
    // can contain v >= 8000. Reading WITHOUT a row filter must surface
    // exactly the kept groups' rows — proof the others were skipped.
    const std::string path = writeChunkedParquet(10000, 1000);
    MemoryManager mm(64 << 20);
    FileFormatOptions options;
    options.statsPredicates.push_back(
        StatsPredicate{1, StatsPredicate::Op::Gte, int64_t{8000}});

    ParquetReader reader(path, &mm, options);
    reader.open();
    size_t rows = 0;
    int64_t minV = INT64_MAX;
    while (auto batch = reader.nextBatch()) {
        rows += batch->numRows();
        for (size_t i = 0; i < batch->numRows(); ++i) {
            minV = std::min(minV, batch->column(1).typed<int64_t>()[i]);
        }
    }
    reader.close();
    rm(path);
    EXPECT_EQ(rows, 2000u);
    EXPECT_EQ(minV, 8000);
}

TEST(ParquetReader, StatsPredicateAllGroupsExcluded) {
    const std::string path = writeChunkedParquet(5000, 1000);
    MemoryManager mm(64 << 20);
    FileFormatOptions options;
    options.statsPredicates.push_back(
        StatsPredicate{1, StatsPredicate::Op::Gt, int64_t{999999}});
    ParquetReader reader(path, &mm, options);
    reader.open();
    EXPECT_FALSE(reader.nextBatch().has_value());
    reader.close();
    rm(path);
}

TEST(ParquetReader, StatsPredicateComposesWithProjection) {
    // Project v only (file index 1 -> output 0); predicate in output space.
    const std::string path = writeChunkedParquet(10000, 1000);
    MemoryManager mm(64 << 20);
    FileFormatOptions options;
    options.columns = {1};
    options.statsPredicates.push_back(
        StatsPredicate{0, StatsPredicate::Op::Lt, int64_t{2000}});
    ParquetReader reader(path, &mm, options);
    ASSERT_EQ(reader.schema().numFields(), 1u);
    reader.open();
    size_t rows = 0;
    while (auto batch = reader.nextBatch()) rows += batch->numRows();
    reader.close();
    rm(path);
    EXPECT_EQ(rows, 2000u);
}

TEST(PredicatePushdown, EndToEndAfterPruningSameResults) {
    // SELECT k, SUM(v) FROM t WHERE v >= 8000 GROUP BY k, with and
    // without the passes — identical results, both passes composed.
    const std::string path = writeChunkedParquet(10000, 1000);

    auto makePlan = [&]() {
        FileScanNode fs;
        fs.format = "parquet";
        fs.paths  = {path};
        auto filter       = std::make_unique<FilterNode>();
        filter->child     = PlanNode{std::move(fs)};
        filter->predicate = cmp(BinaryOpType::Gte, 1, 8000);
        auto agg   = std::make_unique<HashAggregateNode>();
        agg->child = PlanNode{std::move(filter)};
        agg->group_keys.push_back(ColumnRefExpr{0});
        AggregateSpec sum;
        sum.kind      = AggregateKind::Sum;
        sum.inputType = Int64;
        sum.input     = ScalarExpr{ColumnRefExpr{1}};
        agg->aggregates.push_back(std::move(sum));
        agg->output_names = {"k", "s"};
        return PlanNode{std::move(agg)};
    };

    auto run = [](PlanNode& plan) {
        MemoryManager mm(64 << 20);
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
    };

    PlanNode plain = makePlan();
    auto expected  = run(plain);
    ASSERT_EQ(expected.size(), 7u);

    PlanNode optimized = makePlan();
    pruneScanColumns(optimized);
    pushScanPredicates(optimized);
    {
        auto& agg    = *std::get<std::unique_ptr<HashAggregateNode>>(optimized);
        auto& filter = *std::get<std::unique_ptr<FilterNode>>(agg.child);
        auto& scan   = std::get<FileScanNode>(filter.child);
        ASSERT_EQ(scan.projected_columns, (std::vector<int32_t>{0, 1}));
        ASSERT_EQ(scan.stat_predicates.size(), 1u);
        EXPECT_EQ(scan.stat_predicates[0].column, 1);  // post-pruning space
    }
    EXPECT_EQ(run(optimized), expected);

    rm(path);
}
