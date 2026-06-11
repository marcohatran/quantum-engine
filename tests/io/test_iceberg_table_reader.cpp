#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Generic.hh>
#include <avro/GenericDatum.hh>
#include <avro/ValidSchema.hh>

#include "io/IcebergTableReader.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

namespace {

std::string makeTempDir(const std::string& hint) {
    auto root = std::filesystem::temp_directory_path() /
                (hint + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(root / "metadata");
    std::filesystem::create_directories(root / "data");
    return root.string();
}

void writeParquetFile(const std::string&          path,
                      const std::vector<int64_t>& ks,
                      const std::vector<int64_t>& vs) {
    arrow::Int64Builder kb, vb;
    for (auto x : ks) ASSERT_TRUE(kb.Append(x).ok());
    for (auto x : vs) ASSERT_TRUE(vb.Append(x).ok());
    std::shared_ptr<arrow::Array> ka, va;
    ASSERT_TRUE(kb.Finish(&ka).ok());
    ASSERT_TRUE(vb.Finish(&va).ok());
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("k", arrow::int64()),
                       arrow::field("v", arrow::int64())}),
        {ka, va});
    auto out_r = arrow::io::FileOutputStream::Open(path);
    ASSERT_TRUE(out_r.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), out_r.ValueOrDie()).ok());
    ASSERT_TRUE(out_r.ValueOrDie()->Close().ok());
}

// Minimal manifest schema — the reader looks fields up by name, so the
// many extra fields real Iceberg writers add are simply absent here.
const char* kManifestSchema = R"({
  "type": "record", "name": "manifest_entry", "fields": [
    {"name": "status", "type": "int"},
    {"name": "data_file", "type": {
      "type": "record", "name": "data_file_r", "fields": [
        {"name": "file_path",   "type": "string"},
        {"name": "file_format", "type": "string"}
    ]}}
]})";

const char* kManifestListSchema = R"({
  "type": "record", "name": "manifest_file", "fields": [
    {"name": "manifest_path", "type": "string"}
]})";

// One manifest entry: (status, data file path).
struct Entry {
    int32_t     status;
    std::string path;
};

void writeManifest(const std::string& path, const std::vector<Entry>& entries) {
    const avro::ValidSchema schema =
        avro::compileJsonSchemaFromString(kManifestSchema);
    avro::DataFileWriter<avro::GenericDatum> writer(path.c_str(), schema);
    for (const auto& e : entries) {
        avro::GenericDatum datum(schema);
        auto& rec = datum.value<avro::GenericRecord>();
        rec.field("status").value<int32_t>() = e.status;
        auto& df = rec.field("data_file").value<avro::GenericRecord>();
        df.field("file_path").value<std::string>()   = e.path;
        df.field("file_format").value<std::string>() = "PARQUET";
        writer.write(datum);
    }
    writer.close();
}

void writeManifestList(const std::string&              path,
                       const std::vector<std::string>& manifestPaths) {
    const avro::ValidSchema schema =
        avro::compileJsonSchemaFromString(kManifestListSchema);
    avro::DataFileWriter<avro::GenericDatum> writer(path.c_str(), schema);
    for (const auto& m : manifestPaths) {
        avro::GenericDatum datum(schema);
        datum.value<avro::GenericRecord>()
            .field("manifest_path").value<std::string>() = m;
        writer.write(datum);
    }
    writer.close();
}

void writeFile(const std::string& path, const std::string& body) {
    std::ofstream(path) << body;
}

std::string metadataJson(int64_t snapshotId, const std::string& manifestList) {
    return std::string(R"({"format-version":1,"current-snapshot-id":)") +
           std::to_string(snapshotId) +
           R"(,"snapshots":[{"snapshot-id":)" + std::to_string(snapshotId) +
           R"(,"manifest-list":")" + manifestList + R"("}]})";
}

std::vector<int64_t> drainKs(IcebergTableReader& reader) {
    std::vector<int64_t> ks;
    while (auto b = reader.nextBatch()) {
        const auto* k = b->column(0).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) ks.push_back(k[i]);
    }
    return ks;
}

} // namespace

TEST(IcebergTableReader, CurrentSnapshotFilesAreRead) {
    auto root = makeTempDir("iceberg_basic_");
    writeParquetFile(root + "/data/a.parquet", {1, 2}, {10, 20});
    writeParquetFile(root + "/data/b.parquet", {3},    {30});

    writeManifest(root + "/metadata/m0.avro",
                  {{1, root + "/data/a.parquet"},
                   {1, root + "/data/b.parquet"}});
    writeManifestList(root + "/metadata/snap-7.avro", {root + "/metadata/m0.avro"});
    writeFile(root + "/metadata/v1.metadata.json",
              metadataJson(7, root + "/metadata/snap-7.avro"));
    writeFile(root + "/metadata/version-hint.text", "1\n");

    MemoryManager mm(1 << 20);
    IcebergTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 2u);
    EXPECT_EQ(reader.schema().numFields(), 2u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();
    std::sort(ks.begin(), ks.end());
    EXPECT_EQ(ks, (std::vector<int64_t>{1, 2, 3}));
    std::filesystem::remove_all(root);
}

TEST(IcebergTableReader, DeletedManifestEntriesAreSkipped) {
    auto root = makeTempDir("iceberg_delete_");
    writeParquetFile(root + "/data/a.parquet", {1, 2}, {10, 20});
    writeParquetFile(root + "/data/b.parquet", {3, 4}, {30, 40});

    // Snapshot 8's manifest marks a.parquet deleted (status=2): only b
    // is live, and existing (status=0) entries count as live.
    writeManifest(root + "/metadata/m1.avro",
                  {{2, root + "/data/a.parquet"},
                   {0, root + "/data/b.parquet"}});
    writeManifestList(root + "/metadata/snap-8.avro", {root + "/metadata/m1.avro"});
    writeFile(root + "/metadata/v1.metadata.json",
              metadataJson(8, root + "/metadata/snap-8.avro"));
    writeFile(root + "/metadata/version-hint.text", "1");

    MemoryManager mm(1 << 20);
    IcebergTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 1u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();
    EXPECT_EQ(ks, (std::vector<int64_t>{3, 4}));
    std::filesystem::remove_all(root);
}

TEST(IcebergTableReader, NoVersionHintUsesLatestMetadataJson) {
    auto root = makeTempDir("iceberg_nohint_");
    writeParquetFile(root + "/data/a.parquet", {1}, {10});
    writeParquetFile(root + "/data/b.parquet", {2}, {20});

    // v1 points at a; v2 points at b. Without version-hint.text the
    // lexicographically-last metadata json (v2) wins.
    writeManifest(root + "/metadata/m1.avro", {{1, root + "/data/a.parquet"}});
    writeManifest(root + "/metadata/m2.avro", {{1, root + "/data/b.parquet"}});
    writeManifestList(root + "/metadata/snap-1.avro", {root + "/metadata/m1.avro"});
    writeManifestList(root + "/metadata/snap-2.avro", {root + "/metadata/m2.avro"});
    writeFile(root + "/metadata/v1.metadata.json",
              metadataJson(1, root + "/metadata/snap-1.avro"));
    writeFile(root + "/metadata/v2.metadata.json",
              metadataJson(2, root + "/metadata/snap-2.avro"));

    MemoryManager mm(1 << 20);
    IcebergTableReader reader(root, &mm);
    ASSERT_EQ(reader.activeFiles().size(), 1u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();
    EXPECT_EQ(ks, (std::vector<int64_t>{2}));
    std::filesystem::remove_all(root);
}

TEST(IcebergTableReader, MultipleManifestsConcatenate) {
    auto root = makeTempDir("iceberg_multi_");
    writeParquetFile(root + "/data/a.parquet", {1}, {10});
    writeParquetFile(root + "/data/b.parquet", {2}, {20});

    writeManifest(root + "/metadata/m1.avro", {{1, root + "/data/a.parquet"}});
    writeManifest(root + "/metadata/m2.avro", {{1, root + "/data/b.parquet"}});
    writeManifestList(root + "/metadata/snap-3.avro",
                      {root + "/metadata/m1.avro", root + "/metadata/m2.avro"});
    writeFile(root + "/metadata/v1.metadata.json",
              metadataJson(3, root + "/metadata/snap-3.avro"));
    writeFile(root + "/metadata/version-hint.text", "1");

    MemoryManager mm(1 << 20);
    IcebergTableReader reader(root, &mm);
    EXPECT_EQ(reader.activeFiles().size(), 2u);

    reader.open();
    auto ks = drainKs(reader);
    reader.close();
    std::sort(ks.begin(), ks.end());
    EXPECT_EQ(ks, (std::vector<int64_t>{1, 2}));
    std::filesystem::remove_all(root);
}
