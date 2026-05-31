#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
#include "io/DeltaTableReader.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/FileScanOperator.hpp"
#include "operator/FilterOperator.hpp"

using namespace quantum;

namespace {

std::string makeTempDir(const std::string& hint) {
    auto root = std::filesystem::temp_directory_path() /
                (hint + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(root);
    std::filesystem::create_directories(root / "_delta_log");
    return root.string();
}

void writeParquetFile(const std::string&          path,
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
    auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out_r.ValueOrDie());
    ASSERT_TRUE(status.ok()) << status.ToString();
    auto close_status = out_r.ValueOrDie()->Close();
    ASSERT_TRUE(close_status.ok());
}

void writeLogFile(const std::string& path, const std::string& body) {
    std::ofstream(path) << body;
}

std::vector<int64_t> drainKs(DeltaTableReader& reader) {
    std::vector<int64_t> ks;
    while (auto b = reader.nextBatch()) {
        const auto* k = b->column(0).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) ks.push_back(k[i]);
    }
    return ks;
}

} // namespace

TEST(DeltaTableReader, SingleAddYieldsAllRows) {
    auto root = makeTempDir("delta_single_");
    writeParquetFile(root + "/part-00000.parquet", {1, 2, 3}, {10, 20, 30});
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        "{\"protocol\":{\"minReaderVersion\":1,\"minWriterVersion\":2}}\n"
        "{\"metaData\":{\"id\":\"abc\",\"format\":{\"provider\":\"parquet\"}}}\n"
        "{\"add\":{\"path\":\"part-00000.parquet\",\"size\":1,\"modificationTime\":0,\"dataChange\":true}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 1u);
    EXPECT_EQ(reader.schema().numFields(), 2u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();

    EXPECT_EQ(ks, (std::vector<int64_t>{1, 2, 3}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, RemoveCancelsEarlierAdd) {
    auto root = makeTempDir("delta_remove_");
    writeParquetFile(root + "/a.parquet", {1, 2}, {10, 20});
    writeParquetFile(root + "/b.parquet", {3, 4}, {30, 40});

    // v0: add a; v1: add b + remove a -> only b is live.
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        "{\"add\":{\"path\":\"a.parquet\"}}\n");
    writeLogFile(root + "/_delta_log/00000000000000000001.json",
        "{\"add\":{\"path\":\"b.parquet\"}}\n"
        "{\"remove\":{\"path\":\"a.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 1u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();

    EXPECT_EQ(ks, (std::vector<int64_t>{3, 4}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, MultipleAddsConcatenate) {
    auto root = makeTempDir("delta_multi_");
    writeParquetFile(root + "/a.parquet", {1}, {10});
    writeParquetFile(root + "/b.parquet", {2}, {20});
    writeParquetFile(root + "/c.parquet", {3}, {30});

    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        "{\"add\":{\"path\":\"a.parquet\"}}\n"
        "{\"add\":{\"path\":\"b.parquet\"}}\n"
        "{\"add\":{\"path\":\"c.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 3u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();

    std::sort(ks.begin(), ks.end());
    EXPECT_EQ(ks, (std::vector<int64_t>{1, 2, 3}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, FileUriAlsoWorks) {
    auto root = makeTempDir("delta_file_uri_");
    writeParquetFile(root + "/part.parquet", {7, 8}, {70, 80});
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        "{\"add\":{\"path\":\"part.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader("file://" + root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 1u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();

    EXPECT_EQ(ks, (std::vector<int64_t>{7, 8}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, EndToEndFilterPipeline) {
    auto root = makeTempDir("delta_pipeline_");
    writeParquetFile(root + "/a.parquet", {1, 2}, {10, 20});
    writeParquetFile(root + "/b.parquet", {3, 4}, {30, 40});
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        "{\"add\":{\"path\":\"a.parquet\"}}\n"
        "{\"add\":{\"path\":\"b.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    auto reader = std::make_unique<DeltaTableReader>(root, &mm);
    auto scan   = std::make_unique<FileScanOperator>(&mm, std::move(reader));

    // WHERE v > 20  ->  keeps {30, 40}
    auto pred = std::make_unique<BinaryExpr>(
        BinaryOp::Gt,
        std::make_unique<ColumnRef>(1),
        std::make_unique<Literal>(int64_t{20}));
    FilterOperator filter(&mm, std::move(scan), std::move(pred));

    filter.open();
    std::vector<int64_t> vs;
    while (auto b = filter.getNext()) {
        auto dense = b->selectActive(&mm);
        for (size_t i = 0; i < dense.numRows(); ++i) {
            vs.push_back(dense.column(1).typed<int64_t>()[i]);
        }
    }
    filter.close();

    std::sort(vs.begin(), vs.end());
    EXPECT_EQ(vs, (std::vector<int64_t>{30, 40}));
    std::filesystem::remove_all(root);
}
