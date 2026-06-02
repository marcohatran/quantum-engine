#pragma once

#include <optional>
#include <vector>

#include "operator/Operator.hpp"

namespace quantum {

// UNION ALL over N children. All children must share the same output
// schema (the union's output schema is the first child's). Drains each
// child in declared order and yields its batches; no dedup, no sort.
class UnionAllOperator final : public Operator {
public:
    UnionAllOperator(MemoryManager* mm, std::vector<OperatorPtr> children);

    void                       open()  override;
    void                       close() override;
    const Schema&              outputSchema() const override;
    std::optional<ColumnBatch> getNext() override;

private:
    std::vector<OperatorPtr> children_;
    size_t                   cursor_ = 0;
    bool                     opened_ = false;
};

} // namespace quantum
