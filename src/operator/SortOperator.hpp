#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "expression/Expression.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Sort by N key expressions. Materializes all input rows, sorts a row-index
// permutation by the key columns in declared order, then emits one densely
// packed batch in sorted order. Memory cost: O(N) row indices + the
// underlying input batches; time: O(N log N) comparator calls.
//
// `ascending` and `nullsFirst` are parallel arrays to `keys`. Each key
// expression is evaluated once per input batch and the resulting column is
// used by the comparator. Supported key types: INT32 / INT64 / FLOAT64 /
// BOOL / STRING. Mixed-type keys are supported (each key has its own type).
class SortOperator final : public Operator {
public:
    SortOperator(MemoryManager*               mm,
                 OperatorPtr                  child,
                 std::vector<ExpressionPtr>   keys,
                 std::vector<bool>            ascending,
                 std::vector<bool>            nullsFirst);

    void                       open()  override;
    void                       close() override;
    const Schema&              outputSchema() const override { return child_->outputSchema(); }
    std::optional<ColumnBatch> getNext() override;

private:
    void materializeAndSort();

    OperatorPtr                child_;
    std::vector<ExpressionPtr> keys_;
    std::vector<bool>          ascending_;
    std::vector<bool>          nullsFirst_;

    bool        emitted_  = false;
    ColumnBatch sorted_;
};

} // namespace quantum
