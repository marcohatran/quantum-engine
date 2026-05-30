#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "expression/Expression.hpp"
#include "hash/BytesHashTable.hpp"
#include "hash/KeyCodec.hpp"
#include "hash/VarBytesHashTable.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// GROUP BY <key columns> with one or more aggregate functions. Two-phase
// execution: drain the child during the first getNext() to populate the hash
// table, then emit one batch holding (key cols..., agg cols...) for every
// distinct group. Subsequent getNext() calls return nullopt.
//
// Internally uses BytesHashTable for all-fixed-width keys (cache-friendly
// inline keys) and VarBytesHashTable when any key column is variable-width
// (e.g. STRING). KeyCodec chooses the mode.
//
// NULL semantics: each key column's validity is encoded into the key, so e.g.
// (NULL, 1) and (NULL, 2) form distinct groups.
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
    ColumnBatch emitFixed();
    ColumnBatch emitVar();

    OperatorPtr                       child_;
    std::vector<ExpressionPtr>        groupKeys_;
    std::vector<AggregateFunctionPtr> aggs_;
    Schema                            outSchema_;
    std::vector<DataType>             keyTypes_;
    std::unique_ptr<KeyCodec>         codec_;

    std::vector<size_t> aggOffsets_;
    size_t              stateBytes_ = 0;

    std::unique_ptr<BytesHashTable>    tableFixed_;
    std::unique_ptr<VarBytesHashTable> tableVar_;
    bool                               built_   = false;
    bool                               emitted_ = false;
};

} // namespace quantum
