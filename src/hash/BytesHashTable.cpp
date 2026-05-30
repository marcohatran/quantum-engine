#include "hash/BytesHashTable.hpp"

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

uint64_t BytesHashTable::hash(const uint8_t* key, size_t len) noexcept {
    // FNV-1a 64-bit, plus a final scramble for better avalanche on small keys.
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= key[i];
        h *= 0x100000001b3ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

BytesHashTable::BytesHashTable(size_t keyBytes, size_t stateBytes,
                               MemoryManager* mm, size_t initialCapacity)
    : mm_(mm),
      keyBytes_(keyBytes),
      stateBytes_(stateBytes),
      capacity_(nextPow2(initialCapacity < 16 ? 16 : initialCapacity)),
      mask_(capacity_ - 1),
      growAt_(capacity_ * 7 / 10) {
    assert(isPow2(capacity_));
    if (keyBytes_ == 0) raise(ErrorCode::InvalidArgument, "BytesHashTable: keyBytes must be > 0");
    keys_     = mm_->allocate(capacity_ * keyBytes_);
    occupied_ = mm_->allocate(bytesForBits(capacity_));
    states_   = mm_->allocate(capacity_ * stateBytes_);
    std::memset(occupied_.data(), 0, occupied_.size());
}

size_t BytesHashTable::probe(const uint8_t* key) const noexcept {
    const uint64_t h    = hash(key, keyBytes_);
    const uint8_t* keys = keys_.data();
    const uint8_t* occ  = occupied_.data();
    size_t         idx  = h & mask_;
    for (;;) {
        if (!getBit(occ, idx)) return idx;
        if (std::memcmp(keys + idx * keyBytes_, key, keyBytes_) == 0) return idx;
        idx = (idx + 1) & mask_;
    }
}

uint8_t* BytesHashTable::findOrInsert(const uint8_t* key, InitFn init) {
    if (numGroups_ >= growAt_) grow();

    const size_t idx = probe(key);
    uint8_t*     occ = occupied_.data();
    uint8_t*     ks  = keys_.data();
    uint8_t*     st  = states_.data() + idx * stateBytes_;

    if (!getBit(occ, idx)) {
        setBit(occ, idx);
        std::memcpy(ks + idx * keyBytes_, key, keyBytes_);
        init(st);
        ++numGroups_;
    }
    return st;
}

const uint8_t* BytesHashTable::find(const uint8_t* key) const noexcept {
    const uint64_t h    = hash(key, keyBytes_);
    const uint8_t* keys = keys_.data();
    const uint8_t* occ  = occupied_.data();
    size_t         idx  = h & mask_;
    for (;;) {
        if (!getBit(occ, idx)) return nullptr;
        if (std::memcmp(keys + idx * keyBytes_, key, keyBytes_) == 0) {
            return states_.data() + idx * stateBytes_;
        }
        idx = (idx + 1) & mask_;
    }
}

void BytesHashTable::grow() {
    const size_t   oldCap   = capacity_;
    const uint8_t* oldKeys  = keys_.data();
    const uint8_t* oldOcc   = occupied_.data();
    const uint8_t* oldStates = states_.data();

    Buffer oldKeysBuf   = std::move(keys_);
    Buffer oldOccBuf    = std::move(occupied_);
    Buffer oldStatesBuf = std::move(states_);

    capacity_  = oldCap * 2;
    mask_      = capacity_ - 1;
    growAt_    = capacity_ * 7 / 10;
    numGroups_ = 0;

    keys_     = mm_->allocate(capacity_ * keyBytes_);
    occupied_ = mm_->allocate(bytesForBits(capacity_));
    states_   = mm_->allocate(capacity_ * stateBytes_);
    std::memset(occupied_.data(), 0, occupied_.size());

    uint8_t* newKeys = keys_.data();
    uint8_t* newOcc  = occupied_.data();
    uint8_t* newSt   = states_.data();

    for (size_t i = 0; i < oldCap; ++i) {
        if (!getBit(oldOcc, i)) continue;
        const uint8_t* k    = oldKeys + i * keyBytes_;
        uint64_t       h    = hash(k, keyBytes_);
        size_t         slot = h & mask_;
        while (getBit(newOcc, slot)) slot = (slot + 1) & mask_;
        setBit(newOcc, slot);
        std::memcpy(newKeys + slot * keyBytes_, k, keyBytes_);
        std::memcpy(newSt   + slot * stateBytes_,
                    oldStates + i * stateBytes_,
                    stateBytes_);
        ++numGroups_;
    }
}

BytesHashTable::Cursor::Cursor(const uint8_t* keys, const uint8_t* occ,
                               const uint8_t* states, size_t cap,
                               size_t keyStride, size_t stateStride)
    : keys_(keys), occ_(occ), states_(states),
      cap_(cap), keyStride_(keyStride), stateStride_(stateStride), idx_(0) {
    while (idx_ < cap_ && !getBit(occ_, idx_)) ++idx_;
}

void BytesHashTable::Cursor::next() noexcept {
    ++idx_;
    while (idx_ < cap_ && !getBit(occ_, idx_)) ++idx_;
}

BytesHashTable::Cursor BytesHashTable::cursor() const noexcept {
    return Cursor(keys_.data(), occupied_.data(), states_.data(),
                  capacity_, keyBytes_, stateBytes_);
}

} // namespace quantum
