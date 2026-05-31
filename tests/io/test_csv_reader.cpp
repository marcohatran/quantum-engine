#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "io/CsvReader.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/FileScanOperator.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/ProjectOperator.hpp"
#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"

using namespace quantum;

namespace {

// Helper: write `content` to a fresh tempfile and return its path. The file
// is unlinked when `cleanup` is called.
std::string makeTempCsv(const std::string& content, const std::string& nameHint) {
    auto dir  = std::filesystem::temp_directory_path();
    auto path = (dir / (nameHint + std::to_string(std::rand()) + ".csv")).string();
    std::ofstream out(path);
    out << content;
    out.close();
    return path;
}

void cleanup(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // namespace

TEST(CsvReader, ReadsBasicIntStringRows) {
    MemoryManager mm(1 << 20);
    auto path = makeTempCsv(
        "1,alice,10.5\n"
        "2,bob,20.25\n"
        "3,carol,30.0\n",
        "csv_basic_");

    Schema schema({{"k", Int64}, {"name", DataType{TypeId::STRING}}, {"v", Float64}});
    CsvReader reader(path, schema, &mm);

    reader.open();
    auto batch = reader.nextBatch();
    reader.close();
    cleanup(path);

    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);

    const auto* k = batch->column(0).typed<int64_t>();
    EXPECT_EQ(k[0], 1); EXPECT_EQ(k[1], 2); EXPECT_EQ(k[2], 3);

    EXPECT_EQ(batch->column(1).stringAt(0), "alice");
    EXPECT_EQ(batch->column(1).stringAt(1), "bob");
    EXPECT_EQ(batch->column(1).stringAt(2), "carol");

    const auto* v = batch->column(2).typed<double>();
    EXPECT_DOUBLE_EQ(v[0], 10.5);
    EXPECT_DOUBLE_EQ(v[1], 20.25);
    EXPECT_DOUBLE_EQ(v[2], 30.0);
}

TEST(CsvReader, EmptyFieldsBecomeNull) {
    MemoryManager mm(1 << 20);
    auto path = makeTempCsv(
        "1,alice\n"
        "2,\n"        // empty name -> NULL
        ",bob\n",     // empty k -> NULL
        "csv_nulls_");

    Schema schema({{"k", Int64}, {"name", DataType{TypeId::STRING}}});
    CsvReader reader(path, schema, &mm);

    reader.open();
    auto batch = reader.nextBatch();
    reader.close();
    cleanup(path);

    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);
    EXPECT_TRUE (batch->column(0).validity().isValid(0));
    EXPECT_TRUE (batch->column(0).validity().isValid(1));
    EXPECT_FALSE(batch->column(0).validity().isValid(2));
    EXPECT_TRUE (batch->column(1).validity().isValid(0));
    EXPECT_FALSE(batch->column(1).validity().isValid(1));
    EXPECT_TRUE (batch->column(1).validity().isValid(2));
}

TEST(CsvReader, HandlesQuotedFieldsWithCommas) {
    MemoryManager mm(1 << 20);
    auto path = makeTempCsv(
        "1,\"Smith, John\"\n"
        "2,\"O\"\"Brien\"\n",   // O"Brien via embedded ""
        "csv_quoted_");

    Schema schema({{"k", Int64}, {"name", DataType{TypeId::STRING}}});
    CsvReader reader(path, schema, &mm);

    reader.open();
    auto batch = reader.nextBatch();
    reader.close();
    cleanup(path);

    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->column(1).stringAt(0), "Smith, John");
    EXPECT_EQ(batch->column(1).stringAt(1), "O\"Brien");
}

TEST(CsvReader, MultipleBatchesViaBatchSizeOption) {
    MemoryManager mm(1 << 20);
    std::string content;
    for (int i = 0; i < 10; ++i) content += std::to_string(i) + "\n";
    auto path = makeTempCsv(content, "csv_batched_");

    Schema schema({{"k", Int64}});
    FileFormatOptions opts; opts.batchSize = 4;
    CsvReader reader(path, schema, &mm, opts);

    reader.open();
    std::vector<int64_t> all;
    while (auto batch = reader.nextBatch()) {
        for (size_t i = 0; i < batch->numRows(); ++i) {
            all.push_back(batch->column(0).typed<int64_t>()[i]);
        }
    }
    reader.close();
    cleanup(path);

    EXPECT_EQ(all.size(), 10u);
    for (int i = 0; i < 10; ++i) EXPECT_EQ(all[i], i);
}

TEST(FileScan, CsvThroughFilterAndProject) {
    MemoryManager mm(1 << 20);
    auto path = makeTempCsv(
        "1,10\n"
        "2,20\n"
        "3,30\n"
        "4,40\n",
        "csv_scan_");

    Schema schema({{"k", Int64}, {"v", Int64}});
    auto reader = std::make_unique<CsvReader>(path, schema, &mm);
    auto scan   = std::make_unique<FileScanOperator>(&mm, std::move(reader));

    // WHERE v > 15 → drops first row
    auto pred = std::make_unique<BinaryExpr>(
        BinaryOp::Gt,
        std::make_unique<ColumnRef>(1),
        std::make_unique<Literal>(int64_t{15}));
    auto filter = std::make_unique<FilterOperator>(&mm, std::move(scan), std::move(pred));

    // SELECT k, v + 100
    std::vector<ExpressionPtr> exprs;
    exprs.push_back(std::make_unique<ColumnRef>(0));
    exprs.push_back(std::make_unique<BinaryExpr>(
        BinaryOp::Add,
        std::make_unique<ColumnRef>(1),
        std::make_unique<Literal>(int64_t{100})));
    ProjectOperator op(&mm, std::move(filter), std::move(exprs), {"k", "v_plus_100"});

    op.open();
    std::vector<std::pair<int64_t,int64_t>> got;
    while (auto batch = op.getNext()) {
        auto dense = batch->selectActive(&mm);
        for (size_t i = 0; i < dense.numRows(); ++i) {
            got.emplace_back(dense.column(0).typed<int64_t>()[i],
                             dense.column(1).typed<int64_t>()[i]);
        }
    }
    op.close();
    cleanup(path);

    EXPECT_EQ(got, (std::vector<std::pair<int64_t,int64_t>>{
        {2, 120}, {3, 130}, {4, 140}}));
}
