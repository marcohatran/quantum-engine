#pragma once

#include "expression/Expression.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Evaluates a BOOL predicate against the child's batch and updates the
// position list (intersected with the existing one). Column data is moved
// through unchanged — the filter's speed comes from index updates only.
class FilterOperator final : public Operator {
public:
    FilterOperator(MemoryManager* mm, OperatorPtr child, ExpressionPtr predicate)
        : Operator(mm),
          child_(std::move(child)),
          predicate_(std::move(predicate)) {}

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return child_->outputSchema(); }

private:
    OperatorPtr   child_;
    ExpressionPtr predicate_;
};

} // namespace quantum
