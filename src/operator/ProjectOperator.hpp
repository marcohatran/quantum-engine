#pragma once

#include <vector>

#include "expression/Expression.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Evaluates one expression per output column. The new batch carries the
// child's position list unchanged (Phase 1 keeps the filter optimization
// alive end-to-end).
class ProjectOperator final : public Operator {
public:
    ProjectOperator(MemoryManager* mm,
                    OperatorPtr child,
                    std::vector<ExpressionPtr> exprs,
                    std::vector<std::string>   outputNames);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    OperatorPtr                child_;
    std::vector<ExpressionPtr> exprs_;
    Schema                     outSchema_;
};

} // namespace quantum
