#pragma once

#include <cstddef>
#include <cstdint>

#include "common/Macros.hpp"
#include "memory/Buffer.hpp"

namespace quantum {

class MemoryManager;

// Open-addressing hash table for aggregation. Keys are int64; each slot holds
// `stateBytes` of per-group state owned by the caller (initialized via the
// init callback on first insertion). A separate dedicated slot represents the
// NULL group so the main probing loop never has to special-case it.
//
// Phase 2 scope: int64 single-column keys only. Multi-column / string keys are
// a follow-up.
class HashTable {
public:
    using InitFn = void (*)(uint8_t* state);

    HashTable(size_t stateBytes, MemoryManager* mm, size_t initialCapacity = 1024);

    QUANTUM_DISALLOW_COPY(HashTable);

    // Returns a pointer to the per-group state. On first sight, the slot is
    // initialized via `init`. Grows the table when load factor exceeds 0.7.
    uint8_t* findOrInsert(int64_t key, InitFn init);

    // The dedicated NULL-group slot. Lazily initialized on first call.
    uint8_t* findOrInsertNull(InitFn init);

    // Forward cursor over occupied (non-null) slots.
    class Cursor {
    public:
        bool                 valid() const noexcept { return idx_ < cap_; }
        int64_t              key()   const noexcept { return keys_[idx_]; }
        const uint8_t*       state() const noexcept { return states_ + idx_ * stride_; }
        void                 next() noexcept;

    private:
        friend class HashTable;
        Cursor(const int64_t* keys, const uint8_t* occ,
               const uint8_t* states, size_t cap, size_t stride);
        const int64_t* keys_;
        const uint8_t* occ_;
        const uint8_t* states_;
        size_t         cap_;
        size_t         stride_;
        size_t         idx_;
    };

    Cursor cursor() const noexcept;

    size_t numGroups()      const noexcept { return numGroups_; }
    bool   hasNullGroup()   const noexcept { return nullInit_; }
    const uint8_t* nullGroupState() const noexcept { return nullState_.data(); }
    size_t stateBytes()     const noexcept { return stateBytes_; }

private:
    static uint64_t hash(int64_t k) noexcept;

    void   grow();
    size_t probe(int64_t key) const noexcept;

    MemoryManager*  mm_;
    size_t          stateBytes_;
    size_t          capacity_;       // power of two
    size_t          mask_;
    size_t          numGroups_  = 0; // non-null groups only
    size_t          growAt_;         // capacity * 7 / 10

    Buffer keys_;       // int64_t[capacity]
    Buffer occupied_;   // bitmask, capacity bits, rounded
    Buffer states_;     // capacity * stateBytes

    Buffer nullState_;  // single state for the NULL group
    bool   nullInit_ = false;
};

} // namespace quantum
