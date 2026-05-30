#include "hash/VarBytesHashTable.hpp"

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

uint64_t VarBytesHashTable::hash(const uint8_t* key, size_t len) noexcept {
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

VarBytesHashTable::VarBytesHashTable(size_t stateBytes,
                                     MemoryManager* mm,
                                     size_t initialCapacity)
    : mm_(mm),
      stateBytes_(stateBytes),
      capacity_(nextPow2(initialCapacity < 16 ? 16 : initialCapacity)),
      mask_(capacity_ - 1),
      growAt_(capacity_ * 7 / 10) {
    assert(isPow2(capacity_));
    slots_    = mm_->allocate(capacity_ * slotStride());
    occupied_ = mm_->allocate(bytesForBits(capacity_));
    std::memset(occupied_.data(), 0, occupied_.size());

    keyArenaSize_ = 64;
    keyArena_     = mm_->allocate(keyArenaSize_);
    keyArenaUsed_ = 0;
}

uint8_t* VarBytesHashTable::slotPtr(size_t idx) noexcept {
    return slots_.data() + idx * slotStride();
}
const uint8_t* VarBytesHashTable::slotPtr(size_t idx) const noexcept {
    return slots_.data() + idx * slotStride();
}

int32_t VarBytesHashTable::appendKey(const uint8_t* key, size_t len) {
    if (keyArenaUsed_ + len > keyArenaSize_) {
        size_t newSize = keyArenaSize_;
        while (newSize < keyArenaUsed_ + len) newSize *= 2;
        Buffer newArena = mm_->allocate(newSize);
        if (keyArenaUsed_ > 0) {
            std::memcpy(newArena.data(), keyArena_.data(), keyArenaUsed_);
        }
        keyArena_     = std::move(newArena);
        keyArenaSize_ = newSize;
    }
    const int32_t off = static_cast<int32_t>(keyArenaUsed_);
    if (len > 0) std::memcpy(keyArena_.data() + keyArenaUsed_, key, len);
    keyArenaUsed_ += len;
    return off;
}

bool VarBytesHashTable::keysEqual(size_t slotIdx, const uint8_t* key, size_t len) const noexcept {
    const uint8_t* s = slotPtr(slotIdx);
    int32_t off, slen;
    std::memcpy(&off,  s,     sizeof(off));
    std::memcpy(&slen, s + 4, sizeof(slen));
    if (static_cast<size_t>(slen) != len) return false;
    if (len == 0) return true;
    return std::memcmp(keyArena_.data() + off, key, len) == 0;
}

size_t VarBytesHashTable::probe(const uint8_t* key, size_t len) const noexcept {
    const uint64_t h    = hash(key, len);
    const uint8_t* occ  = occupied_.data();
    size_t         idx  = h & mask_;
    for (;;) {
        if (!getBit(occ, idx)) return idx;
        if (keysEqual(idx, key, len)) return idx;
        idx = (idx + 1) & mask_;
    }
}

uint8_t* VarBytesHashTable::findOrInsert(const uint8_t* key, size_t len, InitFn init) {
    if (numGroups_ >= growAt_) grow();

    const size_t idx = probe(key, len);
    uint8_t*     occ = occupied_.data();
    uint8_t*     slot = slotPtr(idx);
    uint8_t*     state = slot + kSlotHeaderBytes;

    if (!getBit(occ, idx)) {
        const int32_t off  = appendKey(key, len);
        const int32_t slen = static_cast<int32_t>(len);
        std::memcpy(slot,     &off,  sizeof(off));
        std::memcpy(slot + 4, &slen, sizeof(slen));
        setBit(occ, idx);
        init(state);
        ++numGroups_;
    }
    return state;
}

const uint8_t* VarBytesHashTable::find(const uint8_t* key, size_t len) const noexcept {
    const uint64_t h    = hash(key, len);
    const uint8_t* occ  = occupied_.data();
    size_t         idx  = h & mask_;
    for (;;) {
        if (!getBit(occ, idx)) return nullptr;
        if (keysEqual(idx, key, len)) return slotPtr(idx) + kSlotHeaderBytes;
        idx = (idx + 1) & mask_;
    }
}

void VarBytesHashTable::grow() {
    const size_t oldCap = capacity_;
    Buffer oldSlots    = std::move(slots_);
    Buffer oldOccupied = std::move(occupied_);

    capacity_  = oldCap * 2;
    mask_      = capacity_ - 1;
    growAt_    = capacity_ * 7 / 10;
    numGroups_ = 0;

    slots_    = mm_->allocate(capacity_ * slotStride());
    occupied_ = mm_->allocate(bytesForBits(capacity_));
    std::memset(occupied_.data(), 0, occupied_.size());

    uint8_t*       newOcc   = occupied_.data();
    const uint8_t* oldOcc   = oldOccupied.data();
    const uint8_t* oldArena = keyArena_.data();

    for (size_t i = 0; i < oldCap; ++i) {
        if (!getBit(oldOcc, i)) continue;
        const uint8_t* oldSlot = oldSlots.data() + i * slotStride();
        int32_t off, slen;
        std::memcpy(&off,  oldSlot,     sizeof(off));
        std::memcpy(&slen, oldSlot + 4, sizeof(slen));

        const uint64_t h    = hash(oldArena + off, static_cast<size_t>(slen));
        size_t         slot = h & mask_;
        while (getBit(newOcc, slot)) slot = (slot + 1) & mask_;
        setBit(newOcc, slot);

        std::memcpy(slotPtr(slot), oldSlot, slotStride());
        ++numGroups_;
    }
}

VarBytesHashTable::Cursor::Cursor(const VarBytesHashTable* t)
    : t_(t), cap_(t->capacity_), idx_(0) {
    while (idx_ < cap_ && !getBit(t_->occupied_.data(), idx_)) ++idx_;
}

void VarBytesHashTable::Cursor::next() noexcept {
    ++idx_;
    while (idx_ < cap_ && !getBit(t_->occupied_.data(), idx_)) ++idx_;
}

const uint8_t* VarBytesHashTable::Cursor::key() const noexcept {
    const uint8_t* slot = t_->slotPtr(idx_);
    int32_t off;
    std::memcpy(&off, slot, sizeof(off));
    return t_->keyArena_.data() + off;
}

size_t VarBytesHashTable::Cursor::keyLen() const noexcept {
    const uint8_t* slot = t_->slotPtr(idx_);
    int32_t slen;
    std::memcpy(&slen, slot + 4, sizeof(slen));
    return static_cast<size_t>(slen);
}

const uint8_t* VarBytesHashTable::Cursor::state() const noexcept {
    return t_->slotPtr(idx_) + kSlotHeaderBytes;
}

} // namespace quantum
