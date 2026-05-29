#include "vector/ColumnVector.hpp"

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

ColumnVector::ColumnVector(DataType type, size_t numRows, MemoryManager* mm)
    : type_(type), numRows_(numRows) {
    if (!type.isFixedWidth()) {
        raise(ErrorCode::NotImplemented,
              "ColumnVector: variable-width type {} not supported in Phase 1",
              type.toString());
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

} // namespace quantum
