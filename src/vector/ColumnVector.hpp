#pragma once

#include <cstddef>

#include "common/DataType.hpp"
#include "common/Macros.hpp"
#include "memory/Buffer.hpp"
#include "vector/ValidityBitmask.hpp"

namespace quantum {

class MemoryManager;

// A typed column of `numRows` values, plus a per-row validity bitmask.
// Phase 1 supports BOOL, INT32, INT64, FLOAT64. STRING is reserved.
class ColumnVector {
public:
    ColumnVector() noexcept = default;
    ColumnVector(DataType type, size_t numRows, MemoryManager* mm);

    QUANTUM_DISALLOW_COPY(ColumnVector);
    ColumnVector(ColumnVector&&) noexcept            = default;
    ColumnVector& operator=(ColumnVector&&) noexcept = default;

    DataType type()    const noexcept { return type_; }
    size_t   numRows() const noexcept { return numRows_; }

    Buffer&             values()        noexcept { return values_; }
    const Buffer&       values()  const noexcept { return values_; }
    ValidityBitmask&    validity()       noexcept { return validity_; }
    const ValidityBitmask& validity() const noexcept { return validity_; }

    template <typename T>       T* typed()       noexcept { return values_.as<T>(); }
    template <typename T> const T* typed() const noexcept { return values_.as<T>(); }

    // Factory helpers — each allocates a values buffer sized for `numRows` slots
    // (all-valid by default).
    static ColumnVector make(DataType type, size_t numRows, MemoryManager* mm);

private:
    DataType        type_{TypeId::INT64};
    size_t          numRows_ = 0;
    Buffer          values_;
    ValidityBitmask validity_;
};

} // namespace quantum
