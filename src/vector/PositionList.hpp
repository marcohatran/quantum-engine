#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <absl/types/span.h>

namespace quantum {

// Tracks the indices of currently-active rows in a ColumnBatch. The fast path
// is "identity": no indices stored, callers iterate 0..n-1 directly. Once a
// filter removes any row, the list materializes into an explicit index vector.
class PositionList {
public:
    PositionList() noexcept : identity_(true), n_(0) {}

    static PositionList identity(size_t n) {
        PositionList p;
        p.identity_ = true;
        p.n_        = n;
        return p;
    }

    static PositionList fromIndices(std::vector<int32_t> indices) {
        PositionList p;
        p.identity_ = false;
        p.n_        = indices.size();
        p.indices_  = std::move(indices);
        return p;
    }

    bool   isIdentity() const noexcept { return identity_; }
    size_t size()       const noexcept { return n_; }
    int32_t at(size_t i) const noexcept {
        return identity_ ? static_cast<int32_t>(i) : indices_[i];
    }
    absl::Span<const int32_t> indices() const noexcept {
        return identity_ ? absl::Span<const int32_t>{} : absl::MakeConstSpan(indices_);
    }

private:
    bool                 identity_;
    size_t               n_;
    std::vector<int32_t> indices_;
};

} // namespace quantum
