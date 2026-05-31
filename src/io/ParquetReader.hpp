#pragma once

#include <memory>
#include <string>

#include "common/Macros.hpp"
#include "io/FileFormat.hpp"

// Forward-declare Apache Arrow types so callers don't pull arrow headers.
namespace arrow {
class Table;
namespace io { class RandomAccessFile; }
}
namespace parquet { namespace arrow { class FileReader; } }

namespace quantum {

class MemoryManager;

// Parquet reader on top of Apache Arrow's parquet-arrow API. Reads the
// entire file into an Arrow Table at open() time, then yields each row
// group / batch as a quantum ColumnBatch.
//
// Phase 5.2 supports BOOL / INT32 / INT64 / FLOAT64 / STRING columns.
// Dictionary-encoded STRING columns are read back as plain STRING for now
// (re-dict-encoding into the engine's DictStringBuilder layout is a
// follow-up).
class ParquetReader final : public FileFormatReader {
public:
    ParquetReader(std::string       path,
                  MemoryManager*    mm,
                  FileFormatOptions options = {});
    ~ParquetReader() override;

    QUANTUM_DISALLOW_COPY(ParquetReader);

    void                       open() override;
    std::optional<ColumnBatch> nextBatch() override;
    void                       close() override;

    const Schema& schema() const override { return schema_; }

private:
    std::string       path_;
    MemoryManager*    mm_;
    FileFormatOptions options_;
    Schema            schema_;

    // Arrow handles. Held as pointers to keep the header light.
    std::shared_ptr<arrow::Table>                table_;
    size_t                                       cursor_ = 0;   // row offset
    bool                                         open_   = false;
};

} // namespace quantum
