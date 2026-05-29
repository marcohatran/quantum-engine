#include "vector/ValidityBitmask.hpp"

#include <cassert>
#include <cstring>

#include "common/Bits.hpp"
#include "memory/MemoryManager.hpp"

#if defined(QUANTUM_HAS_AVX2)
    #include <immintrin.h>
#endif

namespace quantum {

size_t ValidityBitmask::bytesFor(size_t numRows) noexcept {
    return bytesForBits(numRows);
}

ValidityBitmask::ValidityBitmask(MemoryManager* mm, size_t numRows)
    : buf_(mm->allocate(bytesFor(numRows))), numRows_(numRows) {
    setAllValid();
}

bool ValidityBitmask::isValid(size_t i) const noexcept {
    return getBit(buf_.data(), i);
}

void ValidityBitmask::setValid(size_t i, bool v) noexcept {
    assignBit(buf_.data(), i, v);
}

void ValidityBitmask::setAllValid() noexcept {
    std::memset(buf_.data(), 0xFF, buf_.size());
}

void ValidityBitmask::setAllNull() noexcept {
    std::memset(buf_.data(), 0x00, buf_.size());
}

size_t ValidityBitmask::nullCount() const noexcept {
    if (numRows_ == 0) return 0;

    const size_t  nFullWords = numRows_ / 64;
    const auto*   words      = reinterpret_cast<const uint64_t*>(buf_.data());

    size_t setBits = 0;
    for (size_t i = 0; i < nFullWords; ++i) {
        setBits += static_cast<size_t>(popcount64(words[i]));
    }

    // Tail bits — those rows in [nFullWords*64, numRows_). The buffer is
    // padded to a 64-byte multiple, so reading words[nFullWords] is always safe.
    const size_t tailBits = numRows_ - nFullWords * 64;
    if (tailBits > 0) {
        const uint64_t mask = (uint64_t{1} << tailBits) - 1U;
        setBits += static_cast<size_t>(popcount64(words[nFullWords] & mask));
    }

    return numRows_ - setBits;
}

void ValidityBitmask::andInPlace(const ValidityBitmask& other) noexcept {
    assert(other.numRows_ == numRows_);
    uint8_t*       dst = buf_.data();
    const uint8_t* src = other.buf_.data();
    size_t         n   = buf_.size();

#if defined(QUANTUM_HAS_AVX2)
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_and_si256(a, b));
    }
    for (; i < n; ++i) dst[i] = static_cast<uint8_t>(dst[i] & src[i]);
#else
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint8_t>(dst[i] & src[i]);
#endif
}

} // namespace quantum
