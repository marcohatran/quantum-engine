#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "common/DataType.hpp"
#include "io/FileFormat.hpp"

namespace quantum {

class MemoryManager;

// A scalar partition value. The native side stores partition column values
// as one of these types; the plan IR's LiteralExpr maps onto this set.
using PartitionValue = std::variant<int64_t, double, bool, std::string>;

// Wraps an inner FileFormatReader and synthesizes one or more constant
// partition columns at the end of each batch. For Hive-style partitioning
// (`year=2024/month=03/file.parquet`), the partition column values are
// constant per-file — the strategy decodes them from the directory path
// and passes them in; this reader just broadcasts them onto every row.
//
// Output schema: [inner schema fields] ++ [partition fields].
class PartitionedFileReader final : public FileFormatReader {
public:
    PartitionedFileReader(FileFormatReaderPtr           inner,
                          std::vector<Field>            partitionFields,
                          std::vector<PartitionValue>   partitionValues,
                          MemoryManager*                mm);
    ~PartitionedFileReader() override = default;

    void                       open() override;
    std::optional<ColumnBatch> nextBatch() override;
    void                       close() override;
    const Schema&              schema() const override { return schema_; }

private:
    FileFormatReaderPtr           inner_;
    std::vector<Field>            partitionFields_;
    std::vector<PartitionValue>   partitionValues_;
    MemoryManager*                mm_;
    Schema                        schema_;
};

} // namespace quantum
