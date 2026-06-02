#pragma once

#include <cstdint>
#include <optional>

#include "operator/Operator.hpp"

namespace quantum {

// Emits at most `limit` active rows from its child. The first N <= limit
// active rows pass through; when the running count reaches `limit`, getNext
// returns nullopt. If a batch from the child would push the running count
// past the limit, it's truncated by setting a new PositionList that keeps
// just enough indices to reach the limit.
class LimitOperator final : public Operator {
public:
    LimitOperator(MemoryManager* mm, OperatorPtr child, int64_t limit)
        : Operator(mm), child_(std::move(child)), limit_(limit) {}

    void                       open()  override { child_->open(); }
    void                       close() override { child_->close(); }
    const Schema&              outputSchema() const override { return child_->outputSchema(); }
    std::optional<ColumnBatch> getNext() override;

private:
    OperatorPtr child_;
    int64_t     limit_   = 0;
    int64_t     emitted_ = 0;
};

} // namespace quantum
