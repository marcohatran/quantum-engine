#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/Bits.hpp"
#include "common/Macros.hpp"

namespace quantum {

// Bump-pointer arena. Allocates 64-byte-aligned chunks of memory; freeing is
// only supported via reset(). Backs MemoryManager for the lifetime of a query.
class Arena {
public:
    explicit Arena(size_t chunkBytes = 1U << 20);  // 1 MiB default chunk
    ~Arena();

    QUANTUM_DISALLOW_COPY(Arena);
    Arena(Arena&&) noexcept            = default;
    Arena& operator=(Arena&&) noexcept = default;

    // Returns a 64-byte-aligned pointer to `bytes` of memory, or nullptr on overflow.
    uint8_t* allocate(size_t bytes, size_t align = kCacheLineBytes) noexcept;

    // Releases every chunk (calls operator delete[]).
    void reset() noexcept;

    size_t bytesReserved() const noexcept { return reserved_; }
    size_t bytesAllocated() const noexcept { return allocated_; }

private:
    struct Chunk {
        uint8_t* base;
        size_t   size;
        size_t   used;
    };

    Chunk& growChunk(size_t needed);

    std::vector<Chunk> chunks_;
    size_t             chunkBytes_;
    size_t             reserved_  = 0;  // total bytes mapped
    size_t             allocated_ = 0;  // bytes handed to callers
};

} // namespace quantum
