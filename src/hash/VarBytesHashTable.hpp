#pragma once

#include <cstddef>
#include <cstdint>

#include "common/Macros.hpp"
#include "memory/Buffer.hpp"

namespace quantum {

class MemoryManager;

// Open-addressing hash table keyed by **variable-length** byte strings.
// Each slot holds an offset and length into a separate key arena plus
// `stateBytes` of inline state.
//
// Used for GROUP BY / JOIN on STRING (or any mix of fixed/variable) columns.
// For fixed-width-only keys, prefer BytesHashTable — the inline-key layout
// is more cache-friendly. KeyCodec exposes which one is needed.
class VarBytesHashTable {
public:
    using InitFn = void (*)(uint8_t* state);

    VarBytesHashTable(size_t stateBytes,
                      MemoryManager* mm,
                      size_t initialCapacity = 1024);

    QUANTUM_DISALLOW_COPY(VarBytesHashTable);

    uint8_t*       findOrInsert(const uint8_t* key, size_t len, InitFn init);

    // Pre-grow so that up to `additional` NEW groups can be inserted
    // without rehashing (state pointers stay valid). See BytesHashTable.
    void reserve(size_t additional);
    const uint8_t* find(const uint8_t* key, size_t len) const noexcept;

    class Cursor {
    public:
        bool                 valid()   const noexcept { return idx_ < cap_; }
        const uint8_t*       key()     const noexcept;
        size_t               keyLen()  const noexcept;
        const uint8_t*       state()   const noexcept;
        void                 next() noexcept;

    private:
        friend class VarBytesHashTable;
        Cursor(const VarBytesHashTable* t);
        const VarBytesHashTable* t_;
        size_t                   cap_;
        size_t                   idx_;
    };

    Cursor cursor() const noexcept { return Cursor(this); }

    size_t numGroups()  const noexcept { return numGroups_; }
    size_t stateBytes() const noexcept { return stateBytes_; }

private:
    static uint64_t hash(const uint8_t* key, size_t len) noexcept;

    // Per-slot layout: [int32 keyOff][int32 keyLen][state bytes...]
    static constexpr size_t kSlotHeaderBytes = 8;

    size_t   slotStride()  const noexcept { return kSlotHeaderBytes + stateBytes_; }
    uint8_t* slotPtr(size_t idx)       noexcept;
    const uint8_t* slotPtr(size_t idx) const noexcept;

    // Append a key into the key arena, growing if needed. Returns the offset.
    int32_t appendKey(const uint8_t* key, size_t len);

    void   grow();
    bool   keysEqual(size_t slotIdx, const uint8_t* key, size_t len) const noexcept;
    size_t probe(const uint8_t* key, size_t len) const noexcept;

    MemoryManager* mm_;
    size_t         stateBytes_;
    size_t         capacity_;
    size_t         mask_;
    size_t         numGroups_  = 0;
    size_t         growAt_;

    Buffer slots_;      // capacity_ * slotStride()
    Buffer occupied_;   // bitmask

    // Growing key arena.
    Buffer keyArena_;
    size_t keyArenaSize_     = 0;   // capacity in bytes
    size_t keyArenaUsed_     = 0;   // bytes appended so far
};

} // namespace quantum
