#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <nlohmann/json.hpp>

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

// Write a checkpoint parquet shaped like Delta's: one action per row, with
// an `add` struct column. The first row's add is null — it carries the
// metaData action instead (with `schemaString` when provided), mimicking
// the protocol/metaData action rows real checkpoints carry.
void writeCheckpointParquet(const std::string&              path,
                            const std::vector<std::string>& addPaths,
                            const std::string&              schemaString = "") {
    auto pool     = arrow::default_memory_pool();
    auto addType  = arrow::struct_({arrow::field("path", arrow::utf8())});
    auto metaType = arrow::struct_({arrow::field("schemaString", arrow::utf8())});

    auto pathB = std::make_shared<arrow::StringBuilder>(pool);
    arrow::StructBuilder addB(
        addType, pool, {std::static_pointer_cast<arrow::ArrayBuilder>(pathB)});
    auto ssB = std::make_shared<arrow::StringBuilder>(pool);
    arrow::StructBuilder metaB(
        metaType, pool, {std::static_pointer_cast<arrow::ArrayBuilder>(ssB)});

    // Row 0: the metaData action (add is null).
    ASSERT_TRUE(addB.AppendNull().ok());
    if (schemaString.empty()) {
        ASSERT_TRUE(metaB.AppendNull().ok());
    } else {
        ASSERT_TRUE(metaB.Append().ok());
        ASSERT_TRUE(ssB->Append(schemaString).ok());
    }
    // One row per add action (metaData is null).
    for (const auto& p : addPaths) {
        ASSERT_TRUE(addB.Append().ok());
        ASSERT_TRUE(pathB->Append(p).ok());
        ASSERT_TRUE(metaB.AppendNull().ok());
    }
    std::shared_ptr<arrow::Array> addArr, metaArr;
    ASSERT_TRUE(addB.Finish(&addArr).ok());
    ASSERT_TRUE(metaB.Finish(&metaArr).ok());

    auto schema = arrow::schema({arrow::field("add", addType),
                                 arrow::field("metaData", metaType)});
    auto table  = arrow::Table::Make(schema, {addArr, metaArr});
    auto out_r  = arrow::io::FileOutputStream::Open(path);
    ASSERT_TRUE(out_r.ok());
    auto status = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), out_r.ValueOrDie());
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_TRUE(out_r.ValueOrDie()->Close().ok());
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

TEST(DeltaTableReader, CheckpointSeedsLiveSetAfterLogCleanup) {
    auto root = makeTempDir("delta_ckpt_");
    writeParquetFile(root + "/a.parquet", {1, 2}, {10, 20});
    writeParquetFile(root + "/b.parquet", {3, 4}, {30, 40});
    writeParquetFile(root + "/c.parquet", {5, 6}, {50, 60});

    // Checkpoint at v1 covers adds of a + b. The v0/v1 JSON commits have
    // been cleaned up (Delta's log retention) — only the checkpoint and
    // the v2 commit remain. v2 adds c and removes a -> live = {b, c}.
    writeCheckpointParquet(
        root + "/_delta_log/00000000000000000001.checkpoint.parquet",
        {"a.parquet", "b.parquet"});
    writeLogFile(root + "/_delta_log/_last_checkpoint",
        "{\"version\":1,\"size\":3}\n");
    writeLogFile(root + "/_delta_log/00000000000000000002.json",
        "{\"add\":{\"path\":\"c.parquet\"}}\n"
        "{\"remove\":{\"path\":\"a.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 2u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();

    std::sort(ks.begin(), ks.end());
    EXPECT_EQ(ks, (std::vector<int64_t>{3, 4, 5, 6}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, CommitsAtOrBeforeCheckpointVersionAreIgnored) {
    auto root = makeTempDir("delta_ckpt_overlap_");
    writeParquetFile(root + "/a.parquet", {1}, {10});
    writeParquetFile(root + "/b.parquet", {2}, {20});

    // v0 and v1 JSON commits still exist alongside the checkpoint at v1.
    // Replaying them on top of the checkpoint would duplicate `a`.
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        "{\"add\":{\"path\":\"a.parquet\"}}\n");
    writeLogFile(root + "/_delta_log/00000000000000000001.json",
        "{\"add\":{\"path\":\"b.parquet\"}}\n");
    writeCheckpointParquet(
        root + "/_delta_log/00000000000000000001.checkpoint.parquet",
        {"a.parquet", "b.parquet"});
    writeLogFile(root + "/_delta_log/_last_checkpoint",
        "{\"version\":1,\"size\":3}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 2u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();

    std::sort(ks.begin(), ks.end());
    EXPECT_EQ(ks, (std::vector<int64_t>{1, 2}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, MultiPartCheckpoint) {
    auto root = makeTempDir("delta_ckpt_parts_");
    writeParquetFile(root + "/a.parquet", {1}, {10});
    writeParquetFile(root + "/b.parquet", {2}, {20});

    // Two-part checkpoint at v3: part 1 adds a, part 2 adds b.
    writeCheckpointParquet(
        root + "/_delta_log/00000000000000000003.checkpoint.0000000001.0000000002.parquet",
        {"a.parquet"});
    writeCheckpointParquet(
        root + "/_delta_log/00000000000000000003.checkpoint.0000000002.0000000002.parquet",
        {"b.parquet"});
    writeLogFile(root + "/_delta_log/_last_checkpoint",
        "{\"version\":3,\"size\":4,\"parts\":2}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 2u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();

    std::sort(ks.begin(), ks.end());
    EXPECT_EQ(ks, (std::vector<int64_t>{1, 2}));
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

// --- Column mapping + schema evolution (Phase 5.43) ---

namespace {

// Parquet file with arbitrary int64 column names/values.
void writeParquetCols(
        const std::string& path,
        const std::vector<std::pair<std::string, std::vector<int64_t>>>& cols) {
    std::vector<std::shared_ptr<arrow::Field>>  fields;
    std::vector<std::shared_ptr<arrow::Array>>  arrays;
    for (const auto& [name, vals] : cols) {
        arrow::Int64Builder b;
        for (auto v : vals) ASSERT_TRUE(b.Append(v).ok());
        std::shared_ptr<arrow::Array> a;
        ASSERT_TRUE(b.Finish(&a).ok());
        fields.push_back(arrow::field(name, arrow::int64()));
        arrays.push_back(std::move(a));
    }
    auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    auto out_r = arrow::io::FileOutputStream::Open(path);
    ASSERT_TRUE(out_r.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), out_r.ValueOrDie()).ok());
    ASSERT_TRUE(out_r.ValueOrDie()->Close().ok());
}

// Spark-style schemaString: long fields with optional column-mapping
// physical names.
std::string schemaStringFor(
        const std::vector<std::pair<std::string, std::string>>& namePhys) {
    nlohmann::json fields = nlohmann::json::array();
    for (const auto& [name, phys] : namePhys) {
        nlohmann::json f = {
            {"name", name}, {"type", "long"}, {"nullable", true},
            {"metadata", nlohmann::json::object()},
        };
        if (!phys.empty()) {
            f["metadata"]["delta.columnMapping.physicalName"] = phys;
        }
        fields.push_back(std::move(f));
    }
    return nlohmann::json{{"type", "struct"}, {"fields", fields}}.dump();
}

std::string metaDataLine(const std::string& schemaString) {
    nlohmann::json j = {{"metaData", {{"id", "m1"}, {"schemaString", schemaString}}}};
    return j.dump() + "\n";
}

} // namespace

TEST(DeltaTableReader, ColumnMappingPhysicalNames) {
    auto root = makeTempDir("delta_colmap_");
    writeParquetCols(root + "/part.parquet",
                     {{"col-aaa", {1, 2}}, {"col-bbb", {10, 20}}});
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        metaDataLine(schemaStringFor({{"k", "col-aaa"}, {"v", "col-bbb"}})) +
        "{\"add\":{\"path\":\"part.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    ASSERT_EQ(reader.schema().numFields(), 2u);
    EXPECT_EQ(reader.schema().field(0).name, "k");
    EXPECT_EQ(reader.schema().field(1).name, "v");

    reader.open();
    std::vector<std::pair<int64_t, int64_t>> rows;
    while (auto b = reader.nextBatch()) {
        for (size_t i = 0; i < b->numRows(); ++i) {
            rows.emplace_back(b->column(0).typed<int64_t>()[i],
                              b->column(1).typed<int64_t>()[i]);
        }
    }
    reader.close();
    EXPECT_EQ(rows, (std::vector<std::pair<int64_t, int64_t>>{{1, 10}, {2, 20}}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, SchemaEvolutionAddColumnNullFills) {
    auto root = makeTempDir("delta_evolve_");
    writeParquetCols(root + "/a.parquet", {{"k", {1, 2}}});
    writeParquetCols(root + "/b.parquet", {{"k", {3}}, {"v", {30}}});

    // v0: table is (k); v1: column v added, new file has both columns.
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        metaDataLine(schemaStringFor({{"k", ""}})) +
        "{\"add\":{\"path\":\"a.parquet\"}}\n");
    writeLogFile(root + "/_delta_log/00000000000000000001.json",
        metaDataLine(schemaStringFor({{"k", ""}, {"v", ""}})) +
        "{\"add\":{\"path\":\"b.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    ASSERT_EQ(reader.schema().numFields(), 2u);
    EXPECT_EQ(reader.schema().field(1).name, "v");

    reader.open();
    std::vector<std::pair<int64_t, std::optional<int64_t>>> rows;
    while (auto b = reader.nextBatch()) {
        for (size_t i = 0; i < b->numRows(); ++i) {
            std::optional<int64_t> v;
            if (b->column(1).validity().isValid(i)) {
                v = b->column(1).typed<int64_t>()[i];
            }
            rows.emplace_back(b->column(0).typed<int64_t>()[i], v);
        }
    }
    reader.close();
    EXPECT_EQ(rows,
              (std::vector<std::pair<int64_t, std::optional<int64_t>>>{
                  {1, std::nullopt}, {2, std::nullopt}, {3, 30}}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, FileColumnOrderDiffersFromSchema) {
    auto root = makeTempDir("delta_reorder_");
    // File columns physically ordered (v, k); table schema says (k, v).
    writeParquetCols(root + "/part.parquet", {{"v", {10, 20}}, {"k", {1, 2}}});
    writeLogFile(root + "/_delta_log/00000000000000000000.json",
        metaDataLine(schemaStringFor({{"k", ""}, {"v", ""}})) +
        "{\"add\":{\"path\":\"part.parquet\"}}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    reader.open();
    std::vector<std::pair<int64_t, int64_t>> rows;
    while (auto b = reader.nextBatch()) {
        for (size_t i = 0; i < b->numRows(); ++i) {
            rows.emplace_back(b->column(0).typed<int64_t>()[i],
                              b->column(1).typed<int64_t>()[i]);
        }
    }
    reader.close();
    EXPECT_EQ(rows, (std::vector<std::pair<int64_t, int64_t>>{{1, 10}, {2, 20}}));
    std::filesystem::remove_all(root);
}

TEST(DeltaTableReader, CheckpointCarriesSchemaString) {
    auto root = makeTempDir("delta_ckpt_schema_");
    writeParquetCols(root + "/part.parquet", {{"col-xyz", {7, 8}}});

    // Schema (with a column-mapping physical name) lives ONLY in the
    // checkpoint's metaData action — the JSON commits are gone.
    writeCheckpointParquet(
        root + "/_delta_log/00000000000000000001.checkpoint.parquet",
        {"part.parquet"},
        schemaStringFor({{"k", "col-xyz"}}));
    writeLogFile(root + "/_delta_log/_last_checkpoint",
        "{\"version\":1,\"size\":2}\n");

    MemoryManager mm(1 << 20);
    DeltaTableReader reader(root, &mm);
    ASSERT_EQ(reader.schema().numFields(), 1u);
    EXPECT_EQ(reader.schema().field(0).name, "k");

    reader.open();
    auto ks = drainKs(reader);
    reader.close();
    EXPECT_EQ(ks, (std::vector<int64_t>{7, 8}));
    std::filesystem::remove_all(root);
}
