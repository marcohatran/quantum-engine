#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vector/ColumnVector.hpp"

namespace quantum {

// Composite-key codec. Encodes a set of key columns (one row at a time) into
// a flat fixed-width byte buffer suitable for BytesHashTable lookup.
//
// Layout per key:
//   [validity bits: ceil(N/8) bytes, rounded to 1][col0 value][col1 value]...
//
// The validity bits guarantee that two rows with different NULL columns map to
// different keys (matching SQL GROUP BY semantics that NULL is its own group).
class KeyCodec {
public:
    explicit KeyCodec(const std::vector<DataType>& keyTypes);

    size_t numColumns() const noexcept { return offsets_.size(); }
    size_t keyBytes()   const noexcept { return keyBytes_; }
    size_t validityBytes() const noexcept { return validityBytes_; }

    // Encode the row at index `row` of `keyCols` into `out`. `out` must point
    // to at least `keyBytes()` writable bytes. `keyCols.size()` must match
    // numColumns().
    void encode(const std::vector<const ColumnVector*>& keyCols,
                size_t                                  row,
                uint8_t*                                out) const;

    // True if any of the encoded columns had a NULL bit in this key buffer.
    // Used by HashJoin to skip probe rows where any key column is NULL.
    bool hasAnyNull(const uint8_t* key) const noexcept;

    // Decode a single column value from a key buffer back into the slot at
    // index `row` of `outCol`. Validity bit is also copied. Used when emitting
    // the group-key columns from HashAggregate's output.
    void decode(const uint8_t*       key,
                size_t               colIdx,
                ColumnVector&        outCol,
                size_t               row) const;

private:
    std::vector<DataType> types_;
    std::vector<size_t>   offsets_;  // byte offset of each column's value
    size_t                validityBytes_ = 0;
    size_t                keyBytes_      = 0;
};

} // namespace quantum
