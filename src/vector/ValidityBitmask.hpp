#pragma once

#include <cstddef>
#include <cstdint>

#include "common/Macros.hpp"
#include "memory/Buffer.hpp"

namespace quantum {

class MemoryManager;

// One-bit-per-row null-tracking buffer. Backing buffer is rounded up to a
// 64-byte multiple so SIMD operations need no tail handling.
class ValidityBitmask {
public:
    ValidityBitmask() noexcept = default;
    ValidityBitmask(MemoryManager* mm, size_t numRows);

    QUANTUM_DISALLOW_COPY(ValidityBitmask);
    ValidityBitmask(ValidityBitmask&&) noexcept            = default;
    ValidityBitmask& operator=(ValidityBitmask&&) noexcept = default;

    static size_t bytesFor(size_t numRows) noexcept;

    // Non-owning view aliasing src's bits; src must outlive the view.
    static ValidityBitmask view(const ValidityBitmask& src) noexcept {
        ValidityBitmask v;
        v.buf_     = Buffer::view(src.buf_);
        v.numRows_ = src.numRows_;
        return v;
    }

    size_t         numRows() const noexcept { return numRows_; }
    uint8_t*       data()         noexcept { return buf_.data(); }
    const uint8_t* data() const   noexcept { return buf_.data(); }
    size_t         sizeBytes() const noexcept { return buf_.size(); }

    bool isValid(size_t i) const noexcept;
    void setValid(size_t i, bool v) noexcept;
    void setAllValid() noexcept;
    void setAllNull()  noexcept;

    size_t nullCount() const noexcept;

    // this[i] = this[i] & other[i]. Requires other.numRows() == this->numRows().
    void andInPlace(const ValidityBitmask& other) noexcept;

private:
    Buffer buf_;
    size_t numRows_ = 0;
};

} // namespace quantum
