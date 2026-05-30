#include "vector/ColumnVector.hpp"

#include <cstring>

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

ColumnVector::ColumnVector(DataType type, size_t numRows, MemoryManager* mm)
    : type_(type), numRows_(numRows) {
    if (!type.isFixedWidth()) {
        raise(ErrorCode::InvalidArgument,
              "ColumnVector: fixed-width constructor cannot build {} columns; "
              "use makeVariableWidth", type.toString());
    }
    const size_t valuesBytes = roundUp(
        static_cast<size_t>(type.fixedWidth()) * numRows,
        kCacheLineBytes);
    values_   = mm->allocate(valuesBytes);
    validity_ = ValidityBitmask(mm, numRows);
}

ColumnVector ColumnVector::make(DataType type, size_t numRows, MemoryManager* mm) {
    return ColumnVector(type, numRows, mm);
}

ColumnVector ColumnVector::makeVariableWidth(DataType       type,
                                             size_t         numRows,
                                             Buffer         offsets,
                                             Buffer         values,
                                             MemoryManager* mm) {
    if (type.isFixedWidth()) {
        raise(ErrorCode::InvalidArgument,
              "makeVariableWidth: {} is not a variable-width type",
              type.toString());
    }
    ColumnVector col;
    col.type_     = type;
    col.numRows_  = numRows;
    col.offsets_  = std::move(offsets);
    col.values_   = std::move(values);
    col.validity_ = ValidityBitmask(mm, numRows);
    return col;
}

ColumnVector ColumnVector::clone(MemoryManager* mm) const {
    if (isVariableWidth()) {
        const size_t offBytes = roundUp((numRows_ + 1) * sizeof(int32_t), kCacheLineBytes);
        const size_t valBytes = roundUp(values_.size(), kCacheLineBytes);
        Buffer newOffsets = mm->allocate(offBytes);
        Buffer newValues  = mm->allocate(valBytes);
        std::memcpy(newOffsets.data(), offsets_.data(), (numRows_ + 1) * sizeof(int32_t));
        if (values_.size() > 0) std::memcpy(newValues.data(), values_.data(), values_.size());
        ColumnVector out = makeVariableWidth(
            type_, numRows_, std::move(newOffsets), std::move(newValues), mm);
        std::memcpy(out.validity().data(), validity_.data(), validity_.sizeBytes());
        return out;
    }
    ColumnVector out = make(type_, numRows_, mm);
    std::memcpy(out.values().data(),   values_.data(),   values_.size());
    std::memcpy(out.validity().data(), validity_.data(), validity_.sizeBytes());
    return out;
}

} // namespace quantum
