#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "expression/Expression.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/ExchangeOperator.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Parallel partitioned aggregation over an exchange: the child is hash-
// partitioned by the GROUP BY keys, so groups are disjoint across
// partitions and P independent HashAggregateOperators — one per
// partition, fanned across the TaskScheduler — produce the global result
// by simple concatenation (no merge phase).
//
// Each partition gets its own pre-built key expressions and aggregate
// function instances (`PartitionAggSpec`): operators stay single-threaded;
// only the thread-safe MemoryManager is shared between workers.
class ParallelHashAggregateOperator final : public Operator {
public:
    struct PartitionAggSpec {
        std::vector<ExpressionPtr>        keys;
        std::vector<AggregateFunctionPtr> aggs;
        std::vector<std::string>          outputNames;
    };

    // `perPartition.size()` defines the parallelism; it must match the
    // exchange's partition count.
    ParallelHashAggregateOperator(MemoryManager*                    mm,
                                  std::unique_ptr<ExchangeOperator> exchange,
                                  std::vector<PartitionAggSpec>     perPartition);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    std::unique_ptr<ExchangeOperator> exchange_;
    std::vector<PartitionAggSpec>     perPartition_;
    Schema                            outSchema_;

    bool                     ran_ = false;
    std::vector<ColumnBatch> results_;
    size_t                   emit_ = 0;
};

} // namespace quantum
