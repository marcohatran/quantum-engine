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

ColumnVector cloneFixedWidth(const ColumnVector& src, MemoryManager* mm) {
    ColumnVector dst = ColumnVector::make(src.type(), src.numRows(), mm);
    std::memcpy(dst.values().data(),   src.values().data(),   src.values().size());
    std::memcpy(dst.validity().data(), src.validity().data(), src.validity().sizeBytes());
    return dst;
}

ColumnVector cloneVariableWidth(const ColumnVector& src, MemoryManager* mm) {
    // ColumnVector::clone already does this — and handles dict-encoded too —
    // so just delegate. Existed before to keep the variable-width branch
    // separate from fixed-width inside selectActive().
    return src.clone(mm);
}

ColumnVector gatherVariableWidth(const ColumnVector& src,
                                 const PositionList& pos,
                                 MemoryManager*      mm) {
    const size_t nOut = pos.size();

    if (src.isDictEncoded()) {
        // Gather codes (int32 fixed-width gather) and share the dictionary
        // via clone-copy. Pruning unused dict entries is a follow-up.
        Buffer codes = mm->allocate(roundUp(nOut * sizeof(int32_t), kCacheLineBytes));
        const int32_t* srcCodes = src.values().as<int32_t>();
        int32_t*       outCodes = codes.as<int32_t>();
        for (size_t k = 0; k < nOut; ++k) {
            outCodes[k] = srcCodes[pos.at(k)];
        }
        Buffer newDictOff = mm->allocate(roundUp(src.dictOffsets().size(), kCacheLineBytes));
        Buffer newDictVal = mm->allocate(roundUp(src.dictValues().size(),  kCacheLineBytes));
        std::memcpy(newDictOff.data(), src.dictOffsets().data(), src.dictOffsets().size());
        if (src.dictValues().size() > 0) {
            std::memcpy(newDictVal.data(), src.dictValues().data(), src.dictValues().size());
        }
        ColumnVector dst = ColumnVector::makeDictEncoded(
            src.type(), nOut, src.dictSize(), std::move(codes),
            std::move(newDictOff), std::move(newDictVal), mm);
        gatherValidity(src.validity(), dst.validity(), pos);
        return dst;
    }

    const int32_t* srcOff = src.offsets().as<int32_t>();

    const size_t outOffBytes = roundUp((nOut + 1) * sizeof(int32_t), kCacheLineBytes);
    Buffer  newOffsets = mm->allocate(outOffBytes);
    int32_t* outOff    = newOffsets.as<int32_t>();
    int64_t running = 0;
    for (size_t k = 0; k < nOut; ++k) {
        outOff[k] = static_cast<int32_t>(running);
        const auto i = static_cast<size_t>(pos.at(k));
        running += srcOff[i + 1] - srcOff[i];
    }
    outOff[nOut] = static_cast<int32_t>(running);

    Buffer values = mm->allocate(roundUp(static_cast<size_t>(running), kCacheLineBytes));
    char*       outVals = values.as<char>();
    const char* srcVals = src.values().as<char>();
    for (size_t k = 0; k < nOut; ++k) {
        const auto i   = static_cast<size_t>(pos.at(k));
        const auto beg = static_cast<size_t>(srcOff[i]);
        const auto len = static_cast<size_t>(srcOff[i + 1] - srcOff[i]);
        if (len > 0) std::memcpy(outVals + outOff[k], srcVals + beg, len);
    }

    ColumnVector dst = ColumnVector::makeVariableWidth(
        src.type(), nOut, std::move(newOffsets), std::move(values), mm);
    gatherValidity(src.validity(), dst.validity(), pos);
    return dst;
}

} // namespace

ColumnBatch ColumnBatch::selectActive(MemoryManager* mm) const {
    if (positions_.isIdentity() && positions_.size() == numRows()) {
        std::vector<ColumnVector> out;
        out.reserve(cols_.size());
        for (const auto& src : cols_) {
            out.push_back(src.isVariableWidth()
                          ? cloneVariableWidth(src, mm)
                          : cloneFixedWidth(src, mm));
        }
        return ColumnBatch(std::move(out), PositionList::identity(numRows()));
    }

    const size_t              nActive = positions_.size();
    std::vector<ColumnVector> out;
    out.reserve(cols_.size());

    for (const auto& src : cols_) {
        if (src.isVariableWidth()) {
            out.push_back(gatherVariableWidth(src, positions_, mm));
            continue;
        }

        ColumnVector dst = ColumnVector::make(src.type(), nActive, mm);
        switch (src.type().id) {
            case TypeId::INT32:   gatherTyped<int32_t>(src, dst, positions_); break;
            case TypeId::INT64:   gatherTyped<int64_t>(src, dst, positions_); break;
            case TypeId::FLOAT64: gatherTyped<double>(src, dst, positions_);  break;
            case TypeId::BOOL:    gatherTyped<uint8_t>(src, dst, positions_); break;
            case TypeId::STRING:  /* unreachable: handled above */            break;
        }
        gatherValidity(src.validity(), dst.validity(), positions_);
        out.push_back(std::move(dst));
    }

    return ColumnBatch(std::move(out), PositionList::identity(nActive));
}

} // namespace quantum
