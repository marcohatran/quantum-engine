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

// Minimum-viable Apache Iceberg (format v1) table reader.
//
// Resolves the current table state from `<root>/metadata`:
//   - `version-hint.text` names the current metadata version (vN); without
//     it, the lexicographically-last `*.metadata.json` is used.
//   - the metadata JSON's `current-snapshot-id` selects a snapshot, whose
//     `manifest-list` Avro file lists the manifest files (older v1 layouts
//     with an inline `manifests` array are also supported).
//   - each manifest's Avro entries carry `status` (0 existing / 1 added /
//     2 deleted) and a `data_file` record; live files are the non-deleted
//     PARQUET entries.
// Surviving data files stream through ParquetReader, schema from the first
// file's footer.
//
// Limitations:
//  - No partition pruning, no schema evolution / field-id mapping, no
//    delete files (format v2), no statistics-based skipping
//  - File paths in manifests are used verbatim (Iceberg stores absolute
//    URIs); they resolve through io::openReadable like any other URI
class IcebergTableReader final : public FileFormatReader {
public:
    IcebergTableReader(std::string       tableUri,
                       MemoryManager*    mm,
                       FileFormatOptions options = {});
    ~IcebergTableReader() override;

    QUANTUM_DISALLOW_COPY(IcebergTableReader);

    void                       open() override;
    std::optional<ColumnBatch> nextBatch() override;
    void                       close() override;

    const Schema& schema() const override { return schema_; }

    // Visible for tests / introspection.
    const std::vector<std::string>& activeFiles() const noexcept { return activeFiles_; }

private:
    void readTableMetadata();
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
