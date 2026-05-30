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
              "use makeVariableWidth or makeDictEncoded", type.toString());
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

ColumnVector ColumnVector::makeDictEncoded(DataType       type,
                                           size_t         numRows,
                                           size_t         dictSize,
                                           Buffer         codes,
                                           Buffer         dictOffsets,
                                           Buffer         dictValues,
                                           MemoryManager* mm) {
    if (type.isFixedWidth()) {
        raise(ErrorCode::InvalidArgument,
              "makeDictEncoded: {} is not a variable-width type",
              type.toString());
    }
    if (dictOffsets.size() < sizeof(int32_t)) {
        raise(ErrorCode::InvalidArgument,
              "makeDictEncoded: dictOffsets must hold at least one int32");
    }
    ColumnVector col;
    col.type_        = type;
    col.numRows_     = numRows;
    col.dictSize_    = dictSize;
    col.values_      = std::move(codes);
    col.dictOffsets_ = std::move(dictOffsets);
    col.dictValues_  = std::move(dictValues);
    col.validity_    = ValidityBitmask(mm, numRows);
    return col;
}

ColumnVector ColumnVector::clone(MemoryManager* mm) const {
    if (isDictEncoded()) {
        const size_t codesBytes = roundUp(numRows_ * sizeof(int32_t), kCacheLineBytes);
        const size_t dOffBytes  = roundUp(dictOffsets_.size(), kCacheLineBytes);
        const size_t dValBytes  = roundUp(dictValues_.size(),  kCacheLineBytes);
        Buffer newCodes   = mm->allocate(codesBytes);
        Buffer newDictOff = mm->allocate(dOffBytes);
        Buffer newDictVal = mm->allocate(dValBytes);
        if (numRows_ > 0) {
            std::memcpy(newCodes.data(), values_.data(), numRows_ * sizeof(int32_t));
        }
        std::memcpy(newDictOff.data(), dictOffsets_.data(), dictOffsets_.size());
        if (dictValues_.size() > 0) {
            std::memcpy(newDictVal.data(), dictValues_.data(), dictValues_.size());
        }
        ColumnVector out = makeDictEncoded(
            type_, numRows_, dictSize_, std::move(newCodes),
            std::move(newDictOff), std::move(newDictVal), mm);
        std::memcpy(out.validity().data(), validity_.data(), validity_.sizeBytes());
        return out;
    }
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
