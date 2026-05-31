#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "common/Macros.hpp"
#include "io/FileFormat.hpp"

namespace quantum {

class MemoryManager;

// Line-based CSV reader. Each row is one line; columns are comma-separated.
// Empty fields become NULLs. Supported types per the supplied schema:
//   BOOL    — accepts "true" / "false" / "1" / "0"
//   INT32   — decimal int
//   INT64   — decimal int
//   FLOAT64 — decimal float
//   STRING  — bytes between commas; double-quoted strings supported
//             (with "" as an embedded quote)
//
// Phase 5.1 scope: no header row (caller supplies the schema), no compression,
// no chunked file splits, no parallel reads. The reader yields ColumnBatches
// of options.batchSize rows.
class CsvReader final : public FileFormatReader {
public:
    CsvReader(std::string       path,
              Schema            schema,
              MemoryManager*    mm,
              FileFormatOptions options = {});

    QUANTUM_DISALLOW_COPY(CsvReader);

    void                       open() override;
    std::optional<ColumnBatch> nextBatch() override;
    void                       close() override;

    const Schema& schema() const override { return schema_; }

private:
    std::optional<ColumnBatch> consumeRows(const std::vector<std::vector<std::string>>& rawCols,
                                           const std::vector<std::vector<bool>>&        validity,
                                           size_t                                       n);

    std::string       path_;
    Schema            schema_;
    MemoryManager*    mm_;
    FileFormatOptions options_;
    std::ifstream     in_;
    bool              open_ = false;
};

} // namespace quantum
