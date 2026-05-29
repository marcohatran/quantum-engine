#pragma once

#include <vector>

#include "operator/Operator.hpp"

namespace quantum {

// Phase 1 mock source: stores a pre-built vector of ColumnBatch and pops one
// per getNext() call. Real Phase 5 scan will read Parquet/Iceberg/Delta.
class ScanOperator final : public Operator {
public:
    ScanOperator(MemoryManager* mm, Schema schema, std::vector<ColumnBatch> batches)
        : Operator(mm),
          schema_(std::move(schema)),
          batches_(std::move(batches)) {}

    void                       open() override  {}
    std::optional<ColumnBatch> getNext() override;
    void                       close() override {}
    const Schema&              outputSchema() const override { return schema_; }

private:
    Schema                   schema_;
    std::vector<ColumnBatch> batches_;
    size_t                   cursor_ = 0;
};

} // namespace quantum
