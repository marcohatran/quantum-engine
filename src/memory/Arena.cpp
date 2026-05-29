#include "memory/Arena.hpp"

#include <cstdlib>
#include <new>

namespace quantum {

namespace {
uint8_t* allocAligned(size_t size, size_t align) noexcept {
    void* p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc(size, align);
#else
    if (posix_memalign(&p, align, size) != 0) {
        p = nullptr;
    }
#endif
    return static_cast<uint8_t*>(p);
}

void freeAligned(void* p) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
} // namespace

Arena::Arena(size_t chunkBytes) : chunkBytes_(chunkBytes) {}

Arena::~Arena() {
    reset();
}

Arena::Chunk& Arena::growChunk(size_t needed) {
    size_t   size = (needed > chunkBytes_) ? needed : chunkBytes_;
    size          = roundUp(size, kCacheLineBytes);
    uint8_t* base = allocAligned(size, kCacheLineBytes);
    chunks_.push_back(Chunk{base, size, 0});
    reserved_ += size;
    return chunks_.back();
}

uint8_t* Arena::allocate(size_t bytes, size_t align) noexcept {
    if (bytes == 0) return nullptr;
    bytes = roundUp(bytes, align);

    Chunk* chunk = chunks_.empty() ? nullptr : &chunks_.back();
    if (!chunk || (chunk->used + bytes) > chunk->size || chunk->base == nullptr) {
        try {
            chunk = &growChunk(bytes);
        } catch (const std::bad_alloc&) {
            return nullptr;
        }
        if (chunk->base == nullptr) return nullptr;
    }

    uint8_t* p = chunk->base + chunk->used;
    chunk->used += bytes;
    allocated_  += bytes;
    return p;
}

void Arena::reset() noexcept {
    for (auto& c : chunks_) {
        freeAligned(c.base);
    }
    chunks_.clear();
    reserved_  = 0;
    allocated_ = 0;
}

} // namespace quantum
