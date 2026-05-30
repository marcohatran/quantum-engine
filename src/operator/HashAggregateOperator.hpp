#pragma once

#include <memory>
#include <string>
#include <vector>

#include "expression/Expression.hpp"
#include "hash/HashTable.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// GROUP BY <int64 key> with one or more aggregate functions. Two-phase
// execution: drain the child during the first getNext() to populate the hash
// table, then emit one batch holding (key, agg...) for every distinct group
// (including a row for the NULL key group when present). Subsequent
// getNext() calls return nullopt.
//
// Phase 2 first slice: single int64 grouping column; no spill, no batched
// emission. Multi-column / string keys and spill come later.
class HashAggregateOperator final : public Operator {
public:
    HashAggregateOperator(MemoryManager*                    mm,
                          OperatorPtr                       child,
                          ExpressionPtr                     groupKey,
                          std::vector<AggregateFunctionPtr> aggs,
                          std::vector<std::string>          outputNames);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    void build();
    ColumnBatch emit();

    OperatorPtr                       child_;
    ExpressionPtr                     groupKey_;
    std::vector<AggregateFunctionPtr> aggs_;
    Schema                            outSchema_;

    std::vector<size_t> aggOffsets_;   // byte offset into per-group state slab
    size_t              stateBytes_ = 0;

    std::unique_ptr<HashTable> table_;
    bool                       built_    = false;
    bool                       emitted_  = false;
};

} // namespace quantum
