#include "vector/ColumnBatch.hpp"

#include <cassert>
#include <cstring>

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

ColumnBatch::ColumnBatch(std::vector<ColumnVector> cols, PositionList pos) noexcept
    : cols_(std::move(cols)), positions_(std::move(pos)) {}

size_t ColumnBatch::numRows() const noexcept {
    return cols_.empty() ? 0 : cols_[0].numRows();
}

namespace {

template <typename T>
void gatherTyped(const ColumnVector& src, ColumnVector& dst, const PositionList& pos) {
    const T* in  = src.typed<T>();
    T*       out = dst.typed<T>();
    for (size_t i = 0; i < pos.size(); ++i) {
        out[i] = in[pos.at(i)];
    }
}

void gatherValidity(const ValidityBitmask& src, ValidityBitmask& dst, const PositionList& pos) {
    for (size_t i = 0; i < pos.size(); ++i) {
        dst.setValid(i, src.isValid(static_cast<size_t>(pos.at(i))));
    }
}

} // namespace

ColumnBatch ColumnBatch::selectActive(MemoryManager* mm) const {
    if (positions_.isIdentity() && positions_.size() == numRows()) {
        // Already dense — but we still need to deep-copy since ColumnVector is move-only.
        std::vector<ColumnVector> out;
        out.reserve(cols_.size());
        for (const auto& src : cols_) {
            ColumnVector dst = ColumnVector::make(src.type(), src.numRows(), mm);
            std::memcpy(dst.values().data(), src.values().data(), src.values().size());
            std::memcpy(dst.validity().data(), src.validity().data(), src.validity().sizeBytes());
            out.push_back(std::move(dst));
        }
        return ColumnBatch(std::move(out), PositionList::identity(numRows()));
    }

    const size_t            nActive = positions_.size();
    std::vector<ColumnVector> out;
    out.reserve(cols_.size());

    for (const auto& src : cols_) {
        ColumnVector dst = ColumnVector::make(src.type(), nActive, mm);

        switch (src.type().id) {
            case TypeId::INT32:   gatherTyped<int32_t>(src, dst, positions_); break;
            case TypeId::INT64:   gatherTyped<int64_t>(src, dst, positions_); break;
            case TypeId::FLOAT64: gatherTyped<double>(src, dst, positions_);  break;
            case TypeId::BOOL:    gatherTyped<uint8_t>(src, dst, positions_); break;
            case TypeId::STRING:
                raise(ErrorCode::NotImplemented,
                      "selectActive: STRING not supported in Phase 1");
        }
        gatherValidity(src.validity(), dst.validity(), positions_);
        out.push_back(std::move(dst));
    }

    return ColumnBatch(std::move(out), PositionList::identity(nActive));
}

} // namespace quantum
