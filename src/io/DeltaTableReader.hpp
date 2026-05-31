#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/Macros.hpp"
#include "io/FileFormat.hpp"

namespace quantum {

class MemoryManager;
class ParquetReader;

// Minimum-viable Delta Lake table reader.
//
// Replays the JSON commit log in `<root>/_delta_log/*.json` to build the
// active file set, then streams batches from each surviving Parquet file
// in turn via the existing ParquetReader. The schema is derived from the
// first surviving file (Delta records the schema in the metaData action,
// but reading it from the parquet footer is just as accurate for this
// MVP and avoids parsing Spark's nested struct-schema JSON).
//
// Limitations (Phase 5.4):
//  - JSON commit log only (no checkpoint .parquet parsing yet)
//  - No partition pruning, no schema evolution, no column mapping
//  - All data files are read in full; no row-group skipping
//
// The table root is a URI accepted by io::openReadable: local path,
// `file://...`, `s3://...`, `gs://...`, `abfs[s]://...`.
class DeltaTableReader final : public FileFormatReader {
public:
    DeltaTableReader(std::string       tableUri,
                     MemoryManager*    mm,
                     FileFormatOptions options = {});
    ~DeltaTableReader() override;

    QUANTUM_DISALLOW_COPY(DeltaTableReader);

    void                       open() override;
    std::optional<ColumnBatch> nextBatch() override;
    void                       close() override;

    const Schema& schema() const override { return schema_; }

    // Visible for tests / introspection.
    const std::vector<std::string>& activeFiles() const noexcept { return activeFiles_; }

private:
    void readDeltaLog();
    void advanceReader();

    std::string              tableUri_;
    MemoryManager*           mm_;
    FileFormatOptions        options_;
    Schema                   schema_;
    std::vector<std::string> activeFiles_;

    std::unique_ptr<ParquetReader> current_;
    size_t                         cursor_ = 0;
    bool                           open_   = false;
};

} // namespace quantum
