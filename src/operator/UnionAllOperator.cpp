#include "operator/UnionAllOperator.hpp"

#include <utility>

#include "common/Status.hpp"

namespace quantum {

UnionAllOperator::UnionAllOperator(MemoryManager* mm,
                                   std::vector<OperatorPtr> children)
    : Operator(mm), children_(std::move(children)) {
    if (children_.empty()) {
        raise(ErrorCode::InvalidArgument, "UnionAllOperator: requires at least one child");
    }
}

void UnionAllOperator::open() {
    cursor_ = 0;
    opened_ = true;
    children_[0]->open();
}

void UnionAllOperator::close() {
    if (opened_ && cursor_ < children_.size()) {
        children_[cursor_]->close();
    }
    opened_ = false;
}

const Schema& UnionAllOperator::outputSchema() const {
    return children_.front()->outputSchema();
}

std::optional<ColumnBatch> UnionAllOperator::getNext() {
    while (cursor_ < children_.size()) {
        auto batch = children_[cursor_]->getNext();
        if (batch.has_value()) return batch;
        children_[cursor_]->close();
        ++cursor_;
        if (cursor_ < children_.size()) children_[cursor_]->open();
    }
    return std::nullopt;
}

} // namespace quantum
