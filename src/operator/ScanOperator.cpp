#include "operator/ScanOperator.hpp"

namespace quantum {

std::optional<ColumnBatch> ScanOperator::getNext() {
    if (cursor_ >= batches_.size()) return std::nullopt;
    return std::optional<ColumnBatch>(std::move(batches_[cursor_++]));
}

} // namespace quantum
