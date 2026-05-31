#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "io/ParquetReader.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/FileScanOperator.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/ProjectOperator.hpp"

using namespace quantum;

namespace {

std::string tempPath(const std::string& nameHint) {
    auto dir  = std::filesystem::temp_directory_path();
    auto path = (dir / (nameHint + std::to_string(std::random_device{}()) + ".parquet")).string();
    return path;
}

void cleanup(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Build an Arrow Table of (k:int64, v:int64, name:utf8) and write it as
// Parquet. Returns the file path.
std::string writeSampleParquet(const std::vector<int64_t>&     ks,
                               const std::vector<int64_t>&     vs,
                               const std::vector<std::string>& names,
                               const std::string&              nameHint) {
    arrow::Int64Builder kBuilder, vBuilder;
    arrow::StringBuilder sBuilder;
    for (auto v : ks) kBuilder.Append(v).ok();
    for (auto v : vs) vBuilder.Append(v).ok();
    for (const auto& s : names) sBuilder.Append(s).ok();

    std::shared_ptr<arrow::Array> kArr, vArr, sArr;
    kBuilder.Finish(&kArr).ok();
    vBuilder.Finish(&vArr).ok();
    sBuilder.Finish(&sArr).ok();

    auto schema = arrow::schema({
        arrow::field("k",    arrow::int64()),
        arrow::field("v",    arrow::int64()),
        arrow::field("name", arrow::utf8())});
    auto table = arrow::Table::Make(schema, {kArr, vArr, sArr});

    auto path = tempPath(nameHint);
    auto outfile_result = arrow::io::FileOutputStream::Open(path);
    EXPECT_TRUE(outfile_result.ok());
    auto outfile = outfile_result.ValueOrDie();

    auto write_status = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile);
    EXPECT_TRUE(write_status.ok())
        << "WriteTable failed: " << write_status.ToString();
    return path;
}

} // namespace

TEST(ParquetReader, RoundtripIntsAndStrings) {
    MemoryManager mm(1 << 20);
    auto path = writeSampleParquet(
        {1, 2, 3, 4},
        {10, 20, 30, 40},
        {"alice", "bob", "carol", "dan"},
        "parquet_basic_");

    ParquetReader reader(path, &mm);
    reader.open();

    auto batch = reader.nextBatch();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->numRows(), 4u);

    const auto* k = batch->column(0).typed<int64_t>();
    EXPECT_EQ(k[0], 1); EXPECT_EQ(k[3], 4);

    const auto* v = batch->column(1).typed<int64_t>();
    EXPECT_EQ(v[0], 10); EXPECT_EQ(v[3], 40);

    EXPECT_EQ(batch->column(2).stringAt(0), "alice");
    EXPECT_EQ(batch->column(2).stringAt(3), "dan");

    EXPECT_FALSE(reader.nextBatch().has_value());
    reader.close();
    cleanup(path);
}

TEST(ParquetReader, BatchSizeOptionSplitsLargeFile) {
    MemoryManager mm(1 << 20);
    std::vector<int64_t>     ks, vs;
    std::vector<std::string> names;
    for (int i = 0; i < 10; ++i) {
        ks.push_back(i);
        vs.push_back(i * 100);
        names.push_back("name_" + std::to_string(i));
    }
    auto path = writeSampleParquet(ks, vs, names, "parquet_batched_");

    FileFormatOptions opts; opts.batchSize = 3;
    ParquetReader reader(path, &mm, opts);
    reader.open();
    size_t totalRows = 0;
    size_t batches   = 0;
    while (auto batch = reader.nextBatch()) {
        totalRows += batch->numRows();
        ++batches;
    }
    reader.close();
    cleanup(path);

    EXPECT_EQ(totalRows, 10u);
    EXPECT_EQ(batches, 4u);   // 3 + 3 + 3 + 1
}

TEST(ParquetReader, ReadsViaFileScheme) {
    MemoryManager mm(1 << 20);
    auto path = writeSampleParquet(
        {7, 8, 9},
        {70, 80, 90},
        {"x", "y", "z"},
        "parquet_uri_");

    // Pass a file:// URI through the same constructor — exercises the
    // io::resolveUri / openReadable path used for s3:// in production.
    ParquetReader reader("file://" + path, &mm);
    reader.open();
    auto batch = reader.nextBatch();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->numRows(), 3u);
    EXPECT_EQ(batch->column(0).typed<int64_t>()[2], 9);
    EXPECT_EQ(batch->column(2).stringAt(0), "x");
    reader.close();
    cleanup(path);
}

TEST(FileScan, ParquetThroughFilterAndProject) {
    MemoryManager mm(1 << 20);
    auto path = writeSampleParquet(
        {1, 2, 3, 4},
        {10, 20, 30, 40},
        {"alice", "bob", "carol", "dan"},
        "parquet_pipeline_");

    auto reader = std::make_unique<ParquetReader>(path, &mm);
    auto scan   = std::make_unique<FileScanOperator>(&mm, std::move(reader));

    // WHERE v > 15 → drops first row
    auto pred = std::make_unique<BinaryExpr>(
        BinaryOp::Gt,
        std::make_unique<ColumnRef>(1),
        std::make_unique<Literal>(int64_t{15}));
    auto filter = std::make_unique<FilterOperator>(&mm, std::move(scan), std::move(pred));

    // SELECT k, v * 10
    std::vector<ExpressionPtr> exprs;
    exprs.push_back(std::make_unique<ColumnRef>(0));
    exprs.push_back(std::make_unique<BinaryExpr>(
        BinaryOp::Mul,
        std::make_unique<ColumnRef>(1),
        std::make_unique<Literal>(int64_t{10})));
    ProjectOperator op(&mm, std::move(filter), std::move(exprs), {"k", "v_times_10"});

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
        {2, 200}, {3, 300}, {4, 400}}));
}
