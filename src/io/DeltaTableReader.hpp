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
// Builds the active file set from the Delta log: if
// `<root>/_delta_log/_last_checkpoint` exists, the live set is seeded from
// the checkpoint parquet's `add` entries (single- and multi-part
// checkpoints), then only JSON commits newer than the checkpoint version
// are replayed — older commits may already be deleted by Delta's log
// cleanup. Batches stream from each surviving Parquet file in turn via the
// existing ParquetReader. The schema is derived from the first surviving
// file (Delta records the schema in the metaData action, but reading it
// from the parquet footer is just as accurate for this MVP and avoids
// parsing Spark's nested struct-schema JSON).
//
// Limitations:
//  - No partition pruning, no schema evolution, no column mapping
//  - All data files are read in full; no row-group skipping
//  - `_last_checkpoint` is trusted; no fallback scan for newer complete
//    checkpoints if it's stale (readers are allowed to do better)
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
