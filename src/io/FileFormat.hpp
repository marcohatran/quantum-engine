#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
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
// One `column <op> literal` row-group skipping condition (see
// plan::ScanPredicate — this is the io-layer mirror so io/ doesn't depend
// on the plan IR). `column` indexes the scan output, i.e. positions after
// FileFormatOptions::columns projection.
struct StatsPredicate {
    enum class Op { Eq, Lt, Gt, Lte, Gte };
    int32_t                       column = 0;
    Op                            op     = Op::Eq;
    std::variant<int64_t, double> literal;
};

struct FileFormatOptions {
    size_t batchSize = 4096;
    // File-schema column indices to read, ascending (empty = all).
    // Honored by ParquetReader; other formats decode everything.
    std::vector<int32_t> columns;
    // Conjunctive row-group skipping predicates (ParquetReader only):
    // row groups whose column-chunk min/max statistics cannot satisfy
    // every predicate are never decoded. Conservative — callers must
    // still filter rows.
    std::vector<StatsPredicate> statsPredicates;
};

using FileFormatReaderPtr = std::unique_ptr<FileFormatReader>;

} // namespace quantum
