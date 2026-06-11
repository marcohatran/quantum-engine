#pragma once

#include <cstdint>
#include <optional>

#include "expression/BinaryExpr.hpp"
#include "expression/Expression.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Evaluates a BOOL predicate against the child's batch and updates the
// position list (intersected with the existing one). Column data is moved
// through unchanged — the filter's speed comes from index updates only.
class FilterOperator final : public Operator {
public:
    FilterOperator(MemoryManager* mm, OperatorPtr child, ExpressionPtr predicate);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return child_->outputSchema(); }

private:
    // Fused fast path for the dominant predicate shape
    // `ColumnRef <cmp> Literal(int64)` over an INT64 column: positions are
    // computed straight from the column values — no intermediate BOOL
    // column materialization, no expression dispatch per batch.
    struct FusedI64Compare {
        size_t   col;
        BinaryOp op;
        int64_t  lit;
    };

    OperatorPtr                    child_;
    ExpressionPtr                  predicate_;
    std::optional<FusedI64Compare> fused_;
};

} // namespace quantum
