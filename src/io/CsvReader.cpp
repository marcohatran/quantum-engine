#include "io/CsvReader.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

namespace {

// Parse one CSV row into fields. Supports double-quoted fields with embedded
// `""` as a quote. No escape characters beyond that. Phase 5.1 doesn't try
// to be RFC-4180 perfect — just enough to roundtrip typical data.
std::vector<std::string> splitCsvRow(std::string_view line) {
    std::vector<std::string> fields;
    std::string              cur;
    bool                     inQuotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (inQuotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                } else {
                    inQuotes = false;
                }
            } else {
                cur.push_back(c);
            }
        } else {
            if (c == ',') {
                fields.push_back(std::move(cur));
                cur.clear();
            } else if (c == '"' && cur.empty()) {
                inQuotes = true;
            } else {
                cur.push_back(c);
            }
        }
    }
    fields.push_back(std::move(cur));
    return fields;
}

bool parseBool(const std::string& s) {
    return s == "true" || s == "1" || s == "TRUE" || s == "True";
}

} // namespace

CsvReader::CsvReader(std::string       path,
                     Schema            schema,
                     MemoryManager*    mm,
                     FileFormatOptions options)
    : path_(std::move(path)),
      schema_(std::move(schema)),
      mm_(mm),
      options_(options) {}

void CsvReader::open() {
    in_.open(path_);
    if (!in_) {
        raise(ErrorCode::InvalidArgument,
              "CsvReader: failed to open '{}' for reading", path_);
    }
    open_ = true;
}

void CsvReader::close() {
    if (in_.is_open()) in_.close();
    open_ = false;
}

std::optional<ColumnBatch> CsvReader::consumeRows(
        const std::vector<std::vector<std::string>>& rawCols,
        const std::vector<std::vector<bool>>&        validity,
        size_t                                       n) {
    if (n == 0) return std::nullopt;

    std::vector<ColumnVector> cols;
    cols.reserve(schema_.numFields());

    for (size_t c = 0; c < schema_.numFields(); ++c) {
        const auto& type = schema_.field(c).type;
        const auto& vals = rawCols[c];
        const auto& vCol = validity[c];

        switch (type.id) {
            case TypeId::BOOL: {
                auto col = ColumnVector::make(Bool, n, mm_);
                col.validity().setAllValid();
                for (size_t i = 0; i < n; ++i) {
                    col.typed<uint8_t>()[i] = vCol[i] ? (parseBool(vals[i]) ? 1U : 0U) : 0U;
                    col.validity().setValid(i, vCol[i]);
                }
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::INT32: {
                auto col = ColumnVector::make(Int32, n, mm_);
                col.validity().setAllValid();
                for (size_t i = 0; i < n; ++i) {
                    if (vCol[i]) col.typed<int32_t>()[i] = std::stoi(vals[i]);
                    else { col.typed<int32_t>()[i] = 0; col.validity().setValid(i, false); }
                }
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::INT64: {
                auto col = ColumnVector::make(Int64, n, mm_);
                col.validity().setAllValid();
                for (size_t i = 0; i < n; ++i) {
                    if (vCol[i]) col.typed<int64_t>()[i] = std::stoll(vals[i]);
                    else { col.typed<int64_t>()[i] = 0; col.validity().setValid(i, false); }
                }
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::FLOAT64: {
                auto col = ColumnVector::make(Float64, n, mm_);
                col.validity().setAllValid();
                for (size_t i = 0; i < n; ++i) {
                    if (vCol[i]) col.typed<double>()[i] = std::stod(vals[i]);
                    else { col.typed<double>()[i] = 0; col.validity().setValid(i, false); }
                }
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::STRING: {
                StringBuilder sb(mm_);
                for (size_t i = 0; i < n; ++i) {
                    if (vCol[i]) sb.append(vals[i]);
                    else         sb.appendNull();
                }
                cols.push_back(sb.build());
                break;
            }
        }
    }
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

std::optional<ColumnBatch> CsvReader::nextBatch() {
    if (!open_) raise(ErrorCode::InvalidArgument, "CsvReader: nextBatch before open");

    const size_t numCols = schema_.numFields();
    std::vector<std::vector<std::string>> rawCols(numCols);
    std::vector<std::vector<bool>>        validity(numCols);

    size_t rows = 0;
    std::string line;
    while (rows < options_.batchSize && std::getline(in_, line)) {
        // Skip blank lines.
        if (line.empty()) continue;
        // Strip a trailing CR (Windows line endings).
        if (line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        auto fields = splitCsvRow(line);
        if (fields.size() != numCols) {
            raise(ErrorCode::InvalidArgument,
                  "CsvReader: row has {} fields, expected {} for path {}",
                  fields.size(), numCols, path_);
        }
        for (size_t c = 0; c < numCols; ++c) {
            const bool isNull = fields[c].empty();
            rawCols[c].push_back(isNull ? std::string{} : std::move(fields[c]));
            validity[c].push_back(!isNull);
        }
        ++rows;
    }

    return consumeRows(rawCols, validity, rows);
}

} // namespace quantum
