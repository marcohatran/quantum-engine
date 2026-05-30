#pragma once

#include <cstddef>
#include <string_view>

#include "common/DataType.hpp"
#include "common/Macros.hpp"
#include "memory/Buffer.hpp"
#include "vector/ValidityBitmask.hpp"

namespace quantum {

class MemoryManager;

// A typed column of `numRows` values, plus a per-row validity bitmask.
//
// Two layouts:
//   - Fixed-width (BOOL, INT32, INT64, FLOAT64): values_ holds numRows slots
//     of fixedWidth() bytes each. offsets_ is unused (empty).
//   - Variable-width (STRING): values_ holds the concatenated bytes for every
//     row; offsets_ holds int32_t[numRows + 1] where row i occupies values_
//     bytes [offsets_[i], offsets_[i+1]). NULL rows have equal consecutive
//     offsets (length 0) and validity bit cleared.
class ColumnVector {
public:
    ColumnVector() noexcept = default;

    // Fixed-width constructor. Throws if `type` is variable-width.
    ColumnVector(DataType type, size_t numRows, MemoryManager* mm);

    QUANTUM_DISALLOW_COPY(ColumnVector);
    ColumnVector(ColumnVector&&) noexcept            = default;
    ColumnVector& operator=(ColumnVector&&) noexcept = default;

    DataType type()    const noexcept { return type_; }
    size_t   numRows() const noexcept { return numRows_; }

    Buffer&             values()         noexcept { return values_; }
    const Buffer&       values()   const noexcept { return values_; }
    Buffer&             offsets()        noexcept { return offsets_; }
    const Buffer&       offsets()  const noexcept { return offsets_; }
    ValidityBitmask&    validity()       noexcept { return validity_; }
    const ValidityBitmask& validity() const noexcept { return validity_; }

    template <typename T>       T* typed()       noexcept { return values_.as<T>(); }
    template <typename T> const T* typed() const noexcept { return values_.as<T>(); }

    bool isVariableWidth() const noexcept { return !type_.isFixedWidth(); }

    // Variable-width row access (STRING). Caller must ensure type is STRING and
    // row index is in bounds; result is invalid if validity bit is cleared.
    std::string_view stringAt(size_t row) const noexcept {
        const int32_t* off = offsets_.as<int32_t>();
        const auto     beg = static_cast<size_t>(off[row]);
        const auto     end = static_cast<size_t>(off[row + 1]);
        return std::string_view(values_.as<char>() + beg, end - beg);
    }

    // Fixed-width factory.
    static ColumnVector make(DataType type, size_t numRows, MemoryManager* mm);

    // Deep-copy: returns a new ColumnVector with identical contents owned by
    // `mm`. Handles both fixed-width and variable-width layouts.
    ColumnVector clone(MemoryManager* mm) const;

    // Variable-width factory. `offsets` must already hold int32_t[numRows + 1]
    // and `values` must hold offsets[numRows] bytes. Validity is allocated and
    // defaults to all-valid; callers update it after the fact.
    static ColumnVector makeVariableWidth(DataType       type,
                                          size_t         numRows,
                                          Buffer         offsets,
                                          Buffer         values,
                                          MemoryManager* mm);

private:
    DataType        type_{TypeId::INT64};
    size_t          numRows_ = 0;
    Buffer          values_;
    Buffer          offsets_;     // empty unless variable-width
    ValidityBitmask validity_;
};

} // namespace quantum
