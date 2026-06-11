#include "io/IcebergTableReader.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>
#include <utility>

#include <arrow/filesystem/api.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>

#include <avro/DataFile.hh>
#include <avro/Generic.hh>
#include <avro/GenericDatum.hh>
#include <avro/Stream.hh>

#include <nlohmann/json.hpp>

#include "common/Status.hpp"
#include "io/FileSystem.hpp"
#include "io/ParquetReader.hpp"

namespace quantum {

namespace {

std::string trimTrailingSlash(std::string s) {
    while (!s.empty() && s.back() == '/') s.pop_back();
    return s;
}

std::string slurp(const std::shared_ptr<arrow::io::RandomAccessFile>& f,
                  const std::string& diagPath) {
    auto size_r = f->GetSize();
    if (!size_r.ok()) {
        raise(ErrorCode::Internal,
              "IcebergTableReader: GetSize('{}') failed: {}",
              diagPath, size_r.status().ToString());
    }
    const int64_t size = size_r.ValueOrDie();
    std::string buf(static_cast<size_t>(size), '\0');
    auto read_r = f->ReadAt(0, size, buf.data());
    if (!read_r.ok()) {
        raise(ErrorCode::Internal,
              "IcebergTableReader: ReadAt('{}') failed: {}",
              diagPath, read_r.status().ToString());
    }
    return buf;
}

std::string trimWhitespace(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.pop_back();
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))    ++b;
    return s.substr(b);
}

bool endsWith(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const avro::GenericDatum& fieldOf(const avro::GenericRecord& rec,
                                  const std::string&         name,
                                  const std::string&         diag) {
    if (!rec.hasField(name)) {
        raise(ErrorCode::InvalidArgument,
              "IcebergTableReader: Avro record in {} has no field '{}'",
              diag, name);
    }
    return rec.field(name);
}

// Read every record of an Avro object-container file at `uri`.
std::vector<avro::GenericDatum> readAvroRows(const std::string& uri) {
    auto file = io::openReadable(uri);
    // memoryInputStream does not copy — keep `bytes` alive while reading.
    const std::string bytes = slurp(file, uri);
    auto in = avro::memoryInputStream(
        reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    avro::DataFileReader<avro::GenericDatum> reader(std::move(in));

    std::vector<avro::GenericDatum> rows;
    avro::GenericDatum datum(reader.dataSchema());
    while (reader.read(datum)) rows.push_back(datum);
    return rows;
}

} // namespace

IcebergTableReader::IcebergTableReader(std::string       tableUri,
                                       MemoryManager*    mm,
                                       FileFormatOptions options)
    : tableUri_(trimTrailingSlash(std::move(tableUri))),
      mm_(mm),
      options_(options) {
    readTableMetadata();

    // Schema from the first surviving file's parquet footer — must be
    // available before open() (downstream operators compute their output
    // schema at construction time).
    if (!activeFiles_.empty()) {
        ParquetReader probe(activeFiles_.front(), mm_, options_);
        schema_ = probe.schema();
    }
}

IcebergTableReader::~IcebergTableReader() = default;

void IcebergTableReader::readTableMetadata() {
    auto [fs, basePath] = io::resolveUri(tableUri_);
    const std::string metaDir = basePath + "/metadata";

    // Locate the current metadata JSON: version-hint.text first, else the
    // lexicographically-last *.metadata.json (versions are zero-padded by
    // writers that omit the hint; vN ordering is best-effort).
    std::string metadataPath;
    if (auto hint_r = fs->OpenInputFile(metaDir + "/version-hint.text");
        hint_r.ok()) {
        const std::string v =
            trimWhitespace(slurp(hint_r.ValueOrDie(), metaDir + "/version-hint.text"));
        metadataPath = metaDir + "/v" + v + ".metadata.json";
    } else {
        arrow::fs::FileSelector sel;
        sel.base_dir  = metaDir;
        sel.recursive = false;
        auto infos_r = fs->GetFileInfo(sel);
        if (!infos_r.ok()) {
            raise(ErrorCode::InvalidArgument,
                  "IcebergTableReader: list '{}/metadata' failed: {}",
                  tableUri_, infos_r.status().ToString());
        }
        auto infos = infos_r.ValueOrDie();
        std::sort(infos.begin(), infos.end(),
                  [](const auto& a, const auto& b) { return a.path() < b.path(); });
        for (const auto& info : infos) {
            if (info.type() != arrow::fs::FileType::File) continue;
            if (endsWith(info.path(), ".metadata.json")) metadataPath = info.path();
        }
        if (metadataPath.empty()) {
            raise(ErrorCode::InvalidArgument,
                  "IcebergTableReader: no metadata json under '{}/metadata'",
                  tableUri_);
        }
    }

    auto meta_r = fs->OpenInputFile(metadataPath);
    if (!meta_r.ok()) {
        raise(ErrorCode::InvalidArgument,
              "IcebergTableReader: open '{}' failed: {}",
              metadataPath, meta_r.status().ToString());
    }
    auto meta = nlohmann::json::parse(
        slurp(meta_r.ValueOrDie(), metadataPath), nullptr,
        /*allow_exceptions=*/false);
    if (meta.is_discarded() || !meta.is_object()) {
        raise(ErrorCode::InvalidArgument,
              "IcebergTableReader: '{}' is not valid JSON", metadataPath);
    }

    if (!meta.contains("current-snapshot-id") ||
        !meta["current-snapshot-id"].is_number()) {
        return;  // no snapshot yet — empty table
    }
    const int64_t currentId = meta["current-snapshot-id"].get<int64_t>();
    if (currentId < 0 || !meta.contains("snapshots")) return;

    nlohmann::json snapshot;
    for (const auto& s : meta["snapshots"]) {
        if (s.is_object() && s.value("snapshot-id", int64_t{-1}) == currentId) {
            snapshot = s;
            break;
        }
    }
    if (snapshot.is_null()) {
        raise(ErrorCode::InvalidArgument,
              "IcebergTableReader: current-snapshot-id {} not in snapshots list",
              currentId);
    }

    // Manifest paths: via the snapshot's manifest-list Avro file, or the
    // older inline `manifests` array.
    std::vector<std::string> manifests;
    if (snapshot.contains("manifest-list") &&
        snapshot["manifest-list"].is_string()) {
        const auto listPath = snapshot["manifest-list"].get<std::string>();
        for (const auto& row : readAvroRows(listPath)) {
            const auto& rec = row.value<avro::GenericRecord>();
            manifests.push_back(
                fieldOf(rec, "manifest_path", listPath).value<std::string>());
        }
    } else if (snapshot.contains("manifests") &&
               snapshot["manifests"].is_array()) {
        for (const auto& m : snapshot["manifests"]) {
            if (m.is_string()) manifests.push_back(m.get<std::string>());
        }
    } else {
        raise(ErrorCode::InvalidArgument,
              "IcebergTableReader: snapshot {} has neither manifest-list "
              "nor manifests", currentId);
    }

    // Manifest entries: status 2 = deleted; everything else is live in
    // this snapshot.
    for (const auto& manifestPath : manifests) {
        for (const auto& row : readAvroRows(manifestPath)) {
            const auto& rec    = row.value<avro::GenericRecord>();
            const auto  status =
                fieldOf(rec, "status", manifestPath).value<int32_t>();
            if (status == 2) continue;
            const auto& df = fieldOf(rec, "data_file", manifestPath)
                                 .value<avro::GenericRecord>();
            std::string format =
                fieldOf(df, "file_format", manifestPath).value<std::string>();
            std::transform(format.begin(), format.end(), format.begin(),
                           [](unsigned char c) { return std::toupper(c); });
            if (format != "PARQUET") {
                raise(ErrorCode::NotImplemented,
                      "IcebergTableReader: data file format '{}' not supported",
                      format);
            }
            activeFiles_.push_back(
                fieldOf(df, "file_path", manifestPath).value<std::string>());
        }
    }
}

void IcebergTableReader::open() {
    cursor_ = 0;
    current_.reset();
    open_   = true;
    advanceReader();
}

void IcebergTableReader::close() {
    if (current_) { current_->close(); current_.reset(); }
    open_ = false;
}

void IcebergTableReader::advanceReader() {
    if (cursor_ >= activeFiles_.size()) {
        current_.reset();
        return;
    }
    current_ = std::make_unique<ParquetReader>(activeFiles_[cursor_++], mm_, options_);
    current_->open();
}

std::optional<ColumnBatch> IcebergTableReader::nextBatch() {
    if (!open_) raise(ErrorCode::InvalidArgument, "IcebergTableReader: nextBatch before open");

    while (current_) {
        auto batch = current_->nextBatch();
        if (batch.has_value()) return batch;
        current_->close();
        advanceReader();
    }
    return std::nullopt;
}

} // namespace quantum
