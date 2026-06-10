#include "io/DeltaTableReader.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <utility>

#include <arrow/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/reader.h>

#include "common/Status.hpp"
#include "io/FileSystem.hpp"
#include "io/ParquetReader.hpp"

namespace quantum {

namespace {

std::string trimTrailingSlash(std::string s) {
    while (!s.empty() && s.back() == '/') s.pop_back();
    return s;
}

std::string joinUri(const std::string& base, const std::string& rel) {
    return trimTrailingSlash(base) + "/" + rel;
}

std::string slurp(const std::shared_ptr<arrow::io::RandomAccessFile>& f,
                  const std::string& diagPath) {
    auto size_r = f->GetSize();
    if (!size_r.ok()) {
        raise(ErrorCode::Internal,
              "DeltaTableReader: GetSize('{}') failed: {}",
              diagPath, size_r.status().ToString());
    }
    int64_t size = size_r.ValueOrDie();
    std::string buf(static_cast<size_t>(size), '\0');
    auto read_r = f->ReadAt(0, size, buf.data());
    if (!read_r.ok()) {
        raise(ErrorCode::Internal,
              "DeltaTableReader: ReadAt('{}') failed: {}",
              diagPath, read_r.status().ToString());
    }
    return buf;
}

struct Action {
    enum class Kind { Add, Remove, Other };
    Kind        kind = Kind::Other;
    std::string path;  // relative to the table root, for Add / Remove
};

std::vector<Action> parseCommit(const std::string& body) {
    std::vector<Action> out;
    std::stringstream ss(body);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) continue;

        Action a;
        if (j.contains("add") && j["add"].is_object()) {
            a.kind = Action::Kind::Add;
            a.path = j["add"].value("path", "");
        } else if (j.contains("remove") && j["remove"].is_object()) {
            a.kind = Action::Kind::Remove;
            a.path = j["remove"].value("path", "");
        }
        if (a.kind != Action::Kind::Other && !a.path.empty()) {
            out.push_back(std::move(a));
        }
    }
    return out;
}

bool endsWith(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string baseName(std::string_view path) {
    const auto slash = path.find_last_of('/');
    return std::string(slash == std::string_view::npos ? path
                                                       : path.substr(slash + 1));
}

// Commit version from a `_delta_log` entry like
// `00000000000000000005.json`. Returns -1 for non-commit names.
int64_t commitVersionOf(const std::string& name) {
    const auto dot = name.find('.');
    if (dot == std::string::npos || dot == 0) return -1;
    for (size_t i = 0; i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') return -1;
    }
    return std::stoll(name.substr(0, dot));
}

std::string zeroPad(int64_t v, int width) {
    std::ostringstream os;
    os << std::setw(width) << std::setfill('0') << v;
    return os.str();
}

// `_last_checkpoint` contents: the version the latest checkpoint covers
// and, for multi-part checkpoints, the part count.
struct CheckpointInfo {
    int64_t version = -1;
    int64_t parts   = 0;  // 0 or 1 = single-file checkpoint
};

std::optional<CheckpointInfo> readLastCheckpoint(
        const std::shared_ptr<arrow::fs::FileSystem>& fs,
        const std::string&                            logDir) {
    auto file_r = fs->OpenInputFile(logDir + "/_last_checkpoint");
    if (!file_r.ok()) return std::nullopt;  // no checkpoint yet
    auto body = slurp(file_r.ValueOrDie(), logDir + "/_last_checkpoint");
    auto j = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object() ||
        !j.contains("version") || !j["version"].is_number()) {
        return std::nullopt;
    }
    CheckpointInfo ci;
    ci.version = j["version"].get<int64_t>();
    if (j.contains("parts") && j["parts"].is_number()) {
        ci.parts = j["parts"].get<int64_t>();
    }
    return ci;
}

// Append the `add.path` entries of one checkpoint parquet to `live`. A
// checkpoint row is one action: rows where the `add` struct is null are
// protocol / metaData / remove-tombstone actions and don't contribute.
void appendCheckpointAdds(
        const std::shared_ptr<arrow::fs::FileSystem>& fs,
        const std::string&                            path,
        std::vector<std::string>&                     live) {
    auto file_r = fs->OpenInputFile(path);
    if (!file_r.ok()) {
        raise(ErrorCode::Internal,
              "DeltaTableReader: open checkpoint '{}' failed: {}",
              path, file_r.status().ToString());
    }
    auto reader_r = parquet::arrow::OpenFile(file_r.ValueOrDie(),
                                             arrow::default_memory_pool());
    if (!reader_r.ok()) {
        raise(ErrorCode::Internal,
              "DeltaTableReader: parquet open checkpoint '{}' failed: {}",
              path, reader_r.status().ToString());
    }
    auto reader = std::move(reader_r).ValueOrDie();

    // Only materialize the `add` column — checkpoints also carry remove /
    // metaData / protocol / stats, none of which we need.
    std::shared_ptr<arrow::Schema> fileSchema;
    auto schema_st = reader->GetSchema(&fileSchema);
    if (!schema_st.ok()) {
        raise(ErrorCode::Internal,
              "DeltaTableReader: checkpoint schema '{}' failed: {}",
              path, schema_st.ToString());
    }
    const int addIdx = fileSchema->GetFieldIndex("add");
    if (addIdx < 0) return;  // no adds in this part

    std::shared_ptr<arrow::Table> table;
    auto read_st = reader->ReadTable({addIdx}, &table);
    if (!read_st.ok()) {
        raise(ErrorCode::Internal,
              "DeltaTableReader: checkpoint ReadTable '{}' failed: {}",
              path, read_st.ToString());
    }

    auto addCol = table->column(0);
    for (const auto& chunk : addCol->chunks()) {
        const auto& sa = static_cast<const arrow::StructArray&>(*chunk);
        auto pathField = sa.GetFieldByName("path");
        if (!pathField) {
            raise(ErrorCode::Internal,
                  "DeltaTableReader: checkpoint '{}' add struct has no path", path);
        }
        const auto& pathArr = static_cast<const arrow::StringArray&>(*pathField);
        for (int64_t i = 0; i < sa.length(); ++i) {
            if (sa.IsNull(i) || pathArr.IsNull(i)) continue;
            live.push_back(pathArr.GetString(i));
        }
    }
}

} // namespace

DeltaTableReader::DeltaTableReader(std::string       tableUri,
                                   MemoryManager*    mm,
                                   FileFormatOptions options)
    : tableUri_(trimTrailingSlash(std::move(tableUri))),
      mm_(mm),
      options_(options) {
    readDeltaLog();

    // Probe the first surviving file for the schema. Downstream operators
    // (ProjectOperator) compute their output schema at construction time,
    // so the schema must be available before open() is called.
    if (!activeFiles_.empty()) {
        ParquetReader probe(activeFiles_.front(), mm_, options_);
        schema_ = probe.schema();
    }
}

DeltaTableReader::~DeltaTableReader() = default;

void DeltaTableReader::readDeltaLog() {
    auto [fs, basePath] = io::resolveUri(tableUri_);
    const std::string logDir = basePath + "/_delta_log";

    arrow::fs::FileSelector sel;
    sel.base_dir       = logDir;
    sel.recursive      = false;
    sel.allow_not_found = false;

    auto infos_r = fs->GetFileInfo(sel);
    if (!infos_r.ok()) {
        raise(ErrorCode::InvalidArgument,
              "DeltaTableReader: list '{}/_delta_log' failed: {}",
              tableUri_, infos_r.status().ToString());
    }
    auto infos = infos_r.ValueOrDie();

    // Delta numbers commits with 20-digit zero-padded versions, so byte
    // ordering on the path is equivalent to numeric ordering on the version.
    std::sort(infos.begin(), infos.end(),
              [](const auto& a, const auto& b) { return a.path() < b.path(); });

    std::vector<std::string> live;  // relative paths, in insertion order

    // Seed the live set from the latest checkpoint, if any. The checkpoint
    // parquet holds the complete table state at its version (one action per
    // row); JSON commits at or before that version are superseded — Delta's
    // log cleanup may already have deleted them.
    int64_t checkpointVersion = -1;
    if (auto ckpt = readLastCheckpoint(fs, logDir)) {
        checkpointVersion = ckpt->version;
        const std::string ver = zeroPad(ckpt->version, 20);
        if (ckpt->parts <= 1) {
            appendCheckpointAdds(fs, logDir + "/" + ver + ".checkpoint.parquet", live);
        } else {
            for (int64_t p = 1; p <= ckpt->parts; ++p) {
                appendCheckpointAdds(
                    fs,
                    logDir + "/" + ver + ".checkpoint." + zeroPad(p, 10) + "." +
                        zeroPad(ckpt->parts, 10) + ".parquet",
                    live);
            }
        }
    }

    for (const auto& info : infos) {
        if (info.type() != arrow::fs::FileType::File) continue;
        if (!endsWith(info.path(), ".json")) continue;  // skip checkpoint parquets
        if (commitVersionOf(baseName(info.path())) <= checkpointVersion) continue;

        auto file_r = fs->OpenInputFile(info.path());
        if (!file_r.ok()) {
            raise(ErrorCode::Internal,
                  "DeltaTableReader: open log '{}' failed: {}",
                  info.path(), file_r.status().ToString());
        }
        auto body = slurp(file_r.ValueOrDie(), info.path());

        for (const auto& a : parseCommit(body)) {
            if (a.kind == Action::Kind::Add) {
                live.push_back(a.path);
            } else if (a.kind == Action::Kind::Remove) {
                auto it = std::find(live.begin(), live.end(), a.path);
                if (it != live.end()) live.erase(it);
            }
        }
    }

    activeFiles_.clear();
    activeFiles_.reserve(live.size());
    for (auto& rel : live) {
        activeFiles_.push_back(joinUri(tableUri_, rel));
    }
}

void DeltaTableReader::open() {
    cursor_ = 0;
    current_.reset();
    open_   = true;
    advanceReader();
}

void DeltaTableReader::close() {
    if (current_) { current_->close(); current_.reset(); }
    open_ = false;
}

void DeltaTableReader::advanceReader() {
    if (cursor_ >= activeFiles_.size()) {
        current_.reset();
        return;
    }
    current_ = std::make_unique<ParquetReader>(activeFiles_[cursor_++], mm_, options_);
    current_->open();
}

std::optional<ColumnBatch> DeltaTableReader::nextBatch() {
    if (!open_) raise(ErrorCode::InvalidArgument, "DeltaTableReader: nextBatch before open");

    while (current_) {
        auto batch = current_->nextBatch();
        if (batch.has_value()) return batch;
        current_->close();
        advanceReader();
    }
    return std::nullopt;
}

} // namespace quantum
