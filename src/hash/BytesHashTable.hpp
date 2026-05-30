#pragma once

#include <cstddef>
#include <cstdint>

#include "common/Macros.hpp"
#include "memory/Buffer.hpp"

namespace quantum {

class MemoryManager;

// Open-addressing hash table keyed by an opaque fixed-width byte string of
// length `keyBytes`. Equality is memcmp; hash is a 64-bit FNV-1a + finalizer
// over the key bytes. Each slot also stores `stateBytes` of caller-managed
// state (initialized via the init callback on first insertion).
//
// This is the workhorse for composite group/join keys: callers encode each
// row's group/join key (including NULL-validity bits) into a flat buffer and
// hand the byte pointer to find / findOrInsert.
class BytesHashTable {
public:
    using InitFn = void (*)(uint8_t* state);

    BytesHashTable(size_t keyBytes,
                   size_t stateBytes,
                   MemoryManager* mm,
                   size_t initialCapacity = 1024);

    QUANTUM_DISALLOW_COPY(BytesHashTable);

    uint8_t*       findOrInsert(const uint8_t* key, InitFn init);
    const uint8_t* find(const uint8_t* key) const noexcept;

    class Cursor {
    public:
        bool                 valid() const noexcept { return idx_ < cap_; }
        const uint8_t*       key()   const noexcept { return keys_  + idx_ * keyStride_; }
        const uint8_t*       state() const noexcept { return states_ + idx_ * stateStride_; }
        void                 next() noexcept;

    private:
        friend class BytesHashTable;
        Cursor(const uint8_t* keys, const uint8_t* occ,
               const uint8_t* states, size_t cap,
               size_t keyStride, size_t stateStride);
        const uint8_t* keys_;
        const uint8_t* occ_;
        const uint8_t* states_;
        size_t         cap_;
        size_t         keyStride_;
        size_t         stateStride_;
        size_t         idx_;
    };

    Cursor cursor() const noexcept;

    size_t numGroups()  const noexcept { return numGroups_; }
    size_t keyBytes()   const noexcept { return keyBytes_; }
    size_t stateBytes() const noexcept { return stateBytes_; }

private:
    static uint64_t hash(const uint8_t* key, size_t len) noexcept;

    void   grow();
    size_t probe(const uint8_t* key) const noexcept;

    MemoryManager* mm_;
    size_t         keyBytes_;
    size_t         stateBytes_;
    size_t         capacity_;
    size_t         mask_;
    size_t         numGroups_ = 0;
    size_t         growAt_;

    Buffer keys_;       // capacity_ * keyBytes_
    Buffer occupied_;   // bitmask
    Buffer states_;     // capacity_ * stateBytes_
};

} // namespace quantum
