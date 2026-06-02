#include "operator/LimitOperator.hpp"

#include <utility>
#include <vector>

namespace quantum {

std::optional<ColumnBatch> LimitOperator::getNext() {
    if (emitted_ >= limit_) return std::nullopt;
    auto batchOpt = child_->getNext();
    if (!batchOpt.has_value()) return std::nullopt;

    ColumnBatch batch = std::move(*batchOpt);
    const auto   batchActive = static_cast<int64_t>(batch.activeRows());
    const int64_t remaining   = limit_ - emitted_;

    if (batchActive <= remaining) {
        emitted_ += batchActive;
        return batch;
    }

    // Truncate the position list to exactly `remaining` indices, preserving
    // the order the child produced.
    const PositionList& pl = batch.positions();
    std::vector<int32_t> indices;
    indices.reserve(static_cast<size_t>(remaining));
    for (int64_t i = 0; i < remaining; ++i) {
        indices.push_back(pl.at(static_cast<size_t>(i)));
    }
    batch.setPositions(PositionList::fromIndices(std::move(indices)));
    emitted_ = limit_;
    return batch;
}

} // namespace quantum
