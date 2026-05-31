#include "io/DeltaTableReader.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include <arrow/filesystem/api.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
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

    arrow::fs::FileSelector sel;
    sel.base_dir       = basePath + "/_delta_log";
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

    for (const auto& info : infos) {
        if (info.type() != arrow::fs::FileType::File) continue;
        if (!endsWith(info.path(), ".json")) continue;  // skip checkpoint parquets

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
