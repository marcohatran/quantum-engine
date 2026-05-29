#pragma once

#include <cstddef>
#include <cstdint>

namespace quantum {

constexpr size_t kCacheLineBytes = 64;

constexpr size_t roundUp(size_t value, size_t multiple) noexcept {
    return ((value + multiple - 1) / multiple) * multiple;
}

constexpr size_t bytesForBits(size_t nBits) noexcept {
    return roundUp((nBits + 7) / 8, kCacheLineBytes);
}

inline int popcount64(uint64_t v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(v);
#else
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return static_cast<int>((v * 0x0101010101010101ULL) >> 56);
#endif
}

inline bool getBit(const uint8_t* bits, size_t i) noexcept {
    return (bits[i >> 3] >> (i & 7)) & 1U;
}

inline void setBit(uint8_t* bits, size_t i) noexcept {
    bits[i >> 3] |= static_cast<uint8_t>(1U << (i & 7));
}

inline void clearBit(uint8_t* bits, size_t i) noexcept {
    bits[i >> 3] &= static_cast<uint8_t>(~(1U << (i & 7)));
}

inline void assignBit(uint8_t* bits, size_t i, bool v) noexcept {
    if (v) setBit(bits, i);
    else   clearBit(bits, i);
}

} // namespace quantum
