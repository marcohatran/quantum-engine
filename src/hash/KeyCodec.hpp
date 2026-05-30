#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "vector/ColumnVector.hpp"

namespace quantum {

// Composite-key codec. Two modes selected at construction by inspecting the
// key column types:
//
//   isFixedWidth() == true  — every column is fixed-width. Encoded keys have
//     a constant size keyBytes() and live in a pre-allocated buffer:
//       [validity bits][col0 value][col1 value]...
//
//   isFixedWidth() == false — at least one column is variable-width (STRING).
//     Encoded keys are variable-length; callers pass a std::vector<uint8_t>
//     buffer to encodeVar(). Layout:
//       [validity bits]
//       per column in order:
//         fixed-width col: exactly `w` bytes (zero-filled when NULL)
//         var-width col:   [int32 length][bytes] (length=0 when NULL)
//
// Both modes encode each column's validity bit into the header so two rows
// with different NULL patterns map to different keys (matching SQL GROUP BY
// semantics that NULL is its own group).
class KeyCodec {
public:
    explicit KeyCodec(const std::vector<DataType>& keyTypes);

    size_t numColumns() const noexcept { return types_.size(); }
    bool   isFixedWidth() const noexcept { return fixedWidth_; }

    // --- Fixed-width-only API (valid iff isFixedWidth()) ---

    size_t keyBytes()      const noexcept { return keyBytes_; }
    size_t validityBytes() const noexcept { return validityBytes_; }

    void encode(const std::vector<const ColumnVector*>& keyCols,
                size_t                                  row,
                uint8_t*                                out) const;

    bool hasAnyNull(const uint8_t* key) const noexcept;

    void decode(const uint8_t* key,
                size_t         colIdx,
                ColumnVector&  outCol,
                size_t         outRow) const;

    // --- Variable-width API (works in both modes) ---

    void encodeVar(const std::vector<const ColumnVector*>& keyCols,
                   size_t                                  row,
                   std::vector<uint8_t>&                   out) const;

    bool hasAnyNullVar(const uint8_t* key, size_t len) const noexcept;

    // Walks the encoded variable-width key. For each column in order,
    // invokes visit(colIdx, isValid, bytes, bytesLen). For fixed-width
    // columns, bytes points at `fixedWidth()` bytes regardless of validity.
    // For variable-width columns, bytes/length describe the value bytes;
    // bytes is non-null but length may be 0.
    using VarVisitor = std::function<void(size_t  colIdx,
                                          bool    isValid,
                                          const uint8_t* bytes,
                                          size_t  bytesLen)>;
    void decodeVar(const uint8_t* key, size_t len, const VarVisitor& visit) const;

private:
    std::vector<DataType> types_;
    bool                  fixedWidth_;

    // Fixed-width-only fields:
    std::vector<size_t>   offsets_;
    size_t                validityBytes_ = 0;
    size_t                keyBytes_      = 0;
};

} // namespace quantum
