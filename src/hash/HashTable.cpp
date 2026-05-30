#include "hash/HashTable.hpp"

#include <cassert>
#include <cstring>

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

namespace {

inline bool isPow2(size_t n) noexcept { return n && (n & (n - 1)) == 0; }

size_t nextPow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

} // namespace

uint64_t HashTable::hash(int64_t k) noexcept {
    auto x = static_cast<uint64_t>(k);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

HashTable::HashTable(size_t stateBytes, MemoryManager* mm, size_t initialCapacity)
    : mm_(mm),
      stateBytes_(stateBytes),
      capacity_(nextPow2(initialCapacity < 16 ? 16 : initialCapacity)),
      mask_(capacity_ - 1),
      growAt_(capacity_ * 7 / 10) {
    assert(isPow2(capacity_));
    keys_     = mm_->allocate(capacity_ * sizeof(int64_t));
    occupied_ = mm_->allocate(bytesForBits(capacity_));
    states_   = mm_->allocate(capacity_ * stateBytes_);
    nullState_ = mm_->allocate(stateBytes_);
    std::memset(occupied_.data(), 0, occupied_.size());
}

size_t HashTable::probe(int64_t key) const noexcept {
    const uint64_t       h        = hash(key);
    const int64_t*       keys     = keys_.as<int64_t>();
    const uint8_t*       occ      = occupied_.data();
    size_t               idx      = h & mask_;
    for (;;) {
        if (!getBit(occ, idx)) return idx;          // empty
        if (keys[idx] == key)  return idx;          // hit
        idx = (idx + 1) & mask_;                    // linear probe
    }
}

uint8_t* HashTable::findOrInsert(int64_t key, InitFn init) {
    if (numGroups_ >= growAt_) grow();

    const size_t idx = probe(key);
    uint8_t*     occ = occupied_.data();
    int64_t*     ks  = keys_.as<int64_t>();
    uint8_t*     st  = states_.data() + idx * stateBytes_;

    if (!getBit(occ, idx)) {
        setBit(occ, idx);
        ks[idx] = key;
        init(st);
        ++numGroups_;
    }
    return st;
}

uint8_t* HashTable::findOrInsertNull(InitFn init) {
    if (!nullInit_) {
        init(nullState_.data());
        nullInit_ = true;
    }
    return nullState_.data();
}

void HashTable::grow() {
    const size_t   oldCap   = capacity_;
    const int64_t* oldKeys  = keys_.as<int64_t>();
    const uint8_t* oldOcc   = occupied_.data();
    const uint8_t* oldStates = states_.data();

    // Allocate new larger arrays. We hold the old buffers in locals until the
    // rehash is done so their MemoryManager lifetime overlaps the new ones.
    Buffer oldKeysBuf    = std::move(keys_);
    Buffer oldOccBuf     = std::move(occupied_);
    Buffer oldStatesBuf  = std::move(states_);

    capacity_ = oldCap * 2;
    mask_     = capacity_ - 1;
    growAt_   = capacity_ * 7 / 10;
    numGroups_ = 0;

    keys_     = mm_->allocate(capacity_ * sizeof(int64_t));
    occupied_ = mm_->allocate(bytesForBits(capacity_));
    states_   = mm_->allocate(capacity_ * stateBytes_);
    std::memset(occupied_.data(), 0, occupied_.size());

    int64_t* newKeys = keys_.as<int64_t>();
    uint8_t* newOcc  = occupied_.data();
    uint8_t* newSt   = states_.data();

    for (size_t i = 0; i < oldCap; ++i) {
        if (!getBit(oldOcc, i)) continue;
        const int64_t k    = oldKeys[i];
        uint64_t      h    = hash(k);
        size_t        slot = h & mask_;
        while (getBit(newOcc, slot)) slot = (slot + 1) & mask_;
        setBit(newOcc, slot);
        newKeys[slot] = k;
        std::memcpy(newSt + slot * stateBytes_,
                    oldStates + i * stateBytes_,
                    stateBytes_);
        ++numGroups_;
    }
}

HashTable::Cursor::Cursor(const int64_t* keys, const uint8_t* occ,
                          const uint8_t* states, size_t cap, size_t stride)
    : keys_(keys), occ_(occ), states_(states), cap_(cap), stride_(stride), idx_(0) {
    while (idx_ < cap_ && !getBit(occ_, idx_)) ++idx_;
}

void HashTable::Cursor::next() noexcept {
    ++idx_;
    while (idx_ < cap_ && !getBit(occ_, idx_)) ++idx_;
}

HashTable::Cursor HashTable::cursor() const noexcept {
    return Cursor(keys_.as<int64_t>(), occupied_.data(),
                  states_.data(), capacity_, stateBytes_);
}

} // namespace quantum
