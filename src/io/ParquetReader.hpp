#pragma once

#include <memory>
#include <string>

#include "common/Macros.hpp"
#include "io/FileFormat.hpp"

// Forward-declare Apache Arrow types so callers don't pull arrow headers.
namespace arrow {
class RecordBatchReader;
namespace io { class RandomAccessFile; }
}
namespace parquet { namespace arrow { class FileReader; } }

namespace quantum {

class MemoryManager;

// Streaming Parquet reader on top of Apache Arrow's parquet-arrow API.
// Construction reads only the file footer (schema available immediately,
// no data decoded); open() starts a RecordBatchReader that decodes
// batchSize-row batches on demand, each converted once into engine
// ColumnVectors. Peak memory is O(batch), not O(file).
//
// Supports BOOL / INT32 / INT64 / FLOAT64 / STRING columns.
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
    std::unique_ptr<parquet::arrow::FileReader> reader_;
    std::shared_ptr<arrow::RecordBatchReader>   batchReader_;
    bool                                        open_ = false;
};

} // namespace quantum
