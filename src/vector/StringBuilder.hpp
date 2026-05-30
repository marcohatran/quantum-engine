#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "common/Macros.hpp"
#include "vector/ColumnVector.hpp"

namespace quantum {

class MemoryManager;

// Helper for constructing a STRING ColumnVector row-by-row.
//
// Buffers grow in std::vector while building; build() copies the accumulated
// bytes and offsets into MemoryManager-backed off-heap Buffers and returns
// the finalized vector. The builder may be reused after build() (it resets
// to empty).
//
// This isn't optimal — a one-pass off-heap builder would be faster — but it
// is correct, simple, and fine for the Phase 3 first slice.
class StringBuilder {
public:
    explicit StringBuilder(MemoryManager* mm);

    QUANTUM_DISALLOW_COPY(StringBuilder);

    void append(std::string_view s);
    void appendNull();

    size_t numRows()  const noexcept { return validity_.size(); }
    size_t numBytes() const noexcept { return bytes_.size(); }

    ColumnVector build();

private:
    MemoryManager*       mm_;
    std::vector<char>    bytes_;
    std::vector<int32_t> offsets_;   // size N+1; offsets_[0] == 0
    std::vector<bool>    validity_;  // size N
};

} // namespace quantum
