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
// Three layouts (all reported as type().id):
//   - Fixed-width (BOOL, INT32, INT64, FLOAT64): values_ holds numRows slots
//     of fixedWidth() bytes each. offsets_, dictOffsets_, dictValues_ are
//     unused (empty).
//   - Variable-width unencoded (STRING): values_ holds the concatenated bytes
//     for every row; offsets_ holds int32_t[numRows + 1] where row i occupies
//     values_ bytes [offsets_[i], offsets_[i+1]). NULL rows have equal
//     consecutive offsets (length 0) and validity bit cleared.
//   - Dictionary-encoded (STRING): values_ holds int32_t[numRows] codes;
//     dictOffsets_ holds int32_t[dictSize + 1]; dictValues_ holds the dict
//     bytes. offsets_ is empty. NULL rows have validity bit cleared; their
//     code value is undefined.
//
// Type stays TypeId::STRING for both variable-width and dict-encoded layouts —
// dict encoding is a layout variant, not a different type. Callers use
// stringAt(row) which dispatches internally.
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

    Buffer&             values()           noexcept { return values_; }
    const Buffer&       values()     const noexcept { return values_; }
    Buffer&             offsets()          noexcept { return offsets_; }
    const Buffer&       offsets()    const noexcept { return offsets_; }
    Buffer&             dictOffsets()      noexcept { return dictOffsets_; }
    const Buffer&       dictOffsets() const noexcept { return dictOffsets_; }
    Buffer&             dictValues()       noexcept { return dictValues_; }
    const Buffer&       dictValues()  const noexcept { return dictValues_; }
    ValidityBitmask&    validity()         noexcept { return validity_; }
    const ValidityBitmask& validity() const noexcept { return validity_; }

    template <typename T>       T* typed()       noexcept { return values_.as<T>(); }
    template <typename T> const T* typed() const noexcept { return values_.as<T>(); }

    bool isVariableWidth() const noexcept { return !type_.isFixedWidth(); }
    bool isDictEncoded()   const noexcept { return dictOffsets_.size() > 0; }

    // dictSize: number of unique entries in the dictionary (0 when not dict).
    size_t dictSize() const noexcept { return dictSize_; }

    // Variable-width row access (STRING). Caller must ensure type is STRING
    // and row index is in bounds; result is undefined if validity bit is clear.
    // Dispatches internally between dict-encoded and unencoded layouts.
    std::string_view stringAt(size_t row) const noexcept {
        if (isDictEncoded()) {
            const int32_t  code = values_.as<int32_t>()[row];
            const int32_t* doff = dictOffsets_.as<int32_t>();
            const auto     beg  = static_cast<size_t>(doff[code]);
            const auto     end  = static_cast<size_t>(doff[code + 1]);
            return std::string_view(dictValues_.as<char>() + beg, end - beg);
        }
        const int32_t* off = offsets_.as<int32_t>();
        const auto     beg = static_cast<size_t>(off[row]);
        const auto     end = static_cast<size_t>(off[row + 1]);
        return std::string_view(values_.as<char>() + beg, end - beg);
    }

    // Non-owning view aliasing src's buffers (validity included). The
    // source must outlive the view; mutating through a view mutates the
    // source. ColumnRef::eval returns views to avoid per-batch column
    // clones — operators that retain eval results beyond the input batch
    // (ProjectOperator) must call ensureOwned first.
    static ColumnVector view(const ColumnVector& src) noexcept {
        ColumnVector v;
        v.type_        = src.type_;
        v.numRows_     = src.numRows_;
        v.values_      = Buffer::view(src.values_);
        v.offsets_     = Buffer::view(src.offsets_);
        v.dictOffsets_ = Buffer::view(src.dictOffsets_);
        v.dictValues_  = Buffer::view(src.dictValues_);
        v.dictSize_    = src.dictSize_;
        v.validity_    = ValidityBitmask::view(src.validity_);
        v.isView_      = true;
        return v;
    }

    bool isView() const noexcept { return isView_; }

    // An owning column: *this moved out when already owned, a deep clone
    // when this is a view.
    ColumnVector ensureOwned(MemoryManager* mm) && {
        if (!isView_) return std::move(*this);
        return clone(mm);
    }

    // Fixed-width factory.
    static ColumnVector make(DataType type, size_t numRows, MemoryManager* mm);

    // Deep-copy: returns a new ColumnVector with identical contents owned by
    // `mm`. Handles all three layouts.
    ColumnVector clone(MemoryManager* mm) const;

    // Variable-width unencoded factory. `offsets` must already hold
    // int32_t[numRows + 1] and `values` must hold offsets[numRows] bytes.
    static ColumnVector makeVariableWidth(DataType       type,
                                          size_t         numRows,
                                          Buffer         offsets,
                                          Buffer         values,
                                          MemoryManager* mm);

    // Dictionary-encoded factory. `codes` must hold int32_t[numRows];
    // `dictOffsets` must hold int32_t[dictSize + 1]; `dictValues` must hold
    // dictOffsets[dictSize] bytes. `dictSize` is the number of dict entries.
    static ColumnVector makeDictEncoded(DataType       type,
                                        size_t         numRows,
                                        size_t         dictSize,
                                        Buffer         codes,
                                        Buffer         dictOffsets,
                                        Buffer         dictValues,
                                        MemoryManager* mm);

private:
    DataType        type_{TypeId::INT64};
    size_t          numRows_ = 0;
    Buffer          values_;       // bytes (var-width) or codes int32 (dict)
    Buffer          offsets_;      // empty for fixed-width and for dict
    Buffer          dictOffsets_;  // empty unless dict-encoded
    Buffer          dictValues_;   // empty unless dict-encoded
    size_t          dictSize_ = 0; // logical dict entries (0 when not dict)
    bool            isView_   = false;
    ValidityBitmask validity_;
};

} // namespace quantum
