#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "expression/Expression.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Hash repartition (the in-process exchange). Drains the child once and
// routes every row to one of `numPartitions` partitions by the hash of
// its key expressions (nulls hash to a fixed seed, so all-null keys
// cluster too). Used two ways:
//
//   - As a plain pull operator: getNext() streams partition 0's batches,
//     then partition 1's, … — output is hash-CLUSTERED: all rows with
//     equal keys appear in exactly one contiguous partition run.
//   - By parallel consumers (ParallelHashAggregateOperator): call
//     takePartitions() instead of getNext() to claim the per-partition
//     batch vectors and fan them out across the TaskScheduler.
class ExchangeOperator final : public Operator {
public:
    ExchangeOperator(MemoryManager*             mm,
                     OperatorPtr                child,
                     std::vector<ExpressionPtr> keys,
                     size_t                     numPartitions);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return child_->outputSchema(); }

    size_t numPartitions() const noexcept { return numPartitions_; }

    // Drain the child and hand over every partition's batches. Mutually
    // exclusive with getNext(); call between open() and close().
    std::vector<std::vector<ColumnBatch>> takePartitions();

private:
    void materialize();

    OperatorPtr                child_;
    std::vector<ExpressionPtr> keys_;
    size_t                     numPartitions_;

    bool                                  materialized_ = false;
    std::vector<std::vector<ColumnBatch>> partitions_;
    size_t                                emitPartition_ = 0;
    size_t                                emitBatch_     = 0;
};

} // namespace quantum
