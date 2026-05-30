#pragma once

#include <memory>
#include <string>
#include <vector>

#include "expression/Expression.hpp"
#include "hash/BytesHashTable.hpp"
#include "hash/KeyCodec.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// GROUP BY <key columns> with one or more aggregate functions. Two-phase
// execution: drain the child during the first getNext() to populate the hash
// table, then emit one batch holding (key cols..., agg cols...) for every
// distinct group. Subsequent getNext() calls return nullopt.
//
// NULL semantics: each key column's validity is encoded into the composite
// key bytes, so e.g. (NULL, 1) and (NULL, 2) form distinct groups, matching
// SQL GROUP BY behaviour.
//
// Phase 2 scope: any number of fixed-width key columns (BOOL/INT32/INT64/
// FLOAT64), all aggregates listed in AggregateFunction.hpp. No spill, no
// batched emission yet.
class HashAggregateOperator final : public Operator {
public:
    HashAggregateOperator(MemoryManager*                    mm,
                          OperatorPtr                       child,
                          std::vector<ExpressionPtr>        groupKeys,
                          std::vector<AggregateFunctionPtr> aggs,
                          std::vector<std::string>          outputNames);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    void        build();
    ColumnBatch emit();

    OperatorPtr                       child_;
    std::vector<ExpressionPtr>        groupKeys_;
    std::vector<AggregateFunctionPtr> aggs_;
    Schema                            outSchema_;
    std::vector<DataType>             keyTypes_;
    std::unique_ptr<KeyCodec>         codec_;

    std::vector<size_t> aggOffsets_;
    size_t              stateBytes_ = 0;

    std::unique_ptr<BytesHashTable> table_;
    bool                            built_   = false;
    bool                            emitted_ = false;
};

} // namespace quantum
