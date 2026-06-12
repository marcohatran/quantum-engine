#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/DataType.hpp"
#include "vector/ColumnBatch.hpp"

namespace quantum {

class MemoryManager;

// Pluggable file-format reader. One reader instance corresponds to one open
// file; nextBatch is called repeatedly until it returns nullopt.
//
// Concrete impls (Phase 5+): CsvReader, ParquetReader (eventually),
// IcebergReader, DeltaReader, ...
class FileFormatReader {
public:
    virtual ~FileFormatReader() = default;
    virtual void                        open()    = 0;
    virtual std::optional<ColumnBatch>  nextBatch() = 0;
    virtual void                        close()   = 0;

    virtual const Schema& schema() const = 0;
};

// Reader options shared across concrete formats. Phase 5.1 introduces only
// the bare minimum; backends will grow their own option structs as they land.
struct FileFormatOptions {
    size_t batchSize = 4096;
    // File-schema column indices to read, ascending (empty = all).
    // Honored by ParquetReader; other formats decode everything.
    std::vector<int32_t> columns;
};

using FileFormatReaderPtr = std::unique_ptr<FileFormatReader>;

} // namespace quantum
