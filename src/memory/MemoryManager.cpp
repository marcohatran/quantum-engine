#include "memory/MemoryManager.hpp"

#include "common/Status.hpp"
#include "memory/Buffer.hpp"

#include <fmt/format.h>

namespace quantum {

MemoryManager::MemoryManager(size_t budgetBytes)
    : arena_(1U << 20), budget_(budgetBytes) {}

Buffer MemoryManager::allocate(size_t bytes) {
    if (bytes == 0) return Buffer{};

    if (!reserve(bytes)) {
        throw QuantumOOM(fmt::format(
            "MemoryManager OOM: requested={} used={} budget={}",
            bytes, used(), budget_));
    }

    uint8_t* p = nullptr;
    {
        std::lock_guard<std::mutex> lock(arenaMu_);
        auto it = freeLists_.find(bytes);
        if (it != freeLists_.end() && !it->second.empty()) {
            p = it->second.back();
            it->second.pop_back();
        } else {
            p = arena_.allocate(bytes);
        }
    }

    if (p == nullptr) {
        release(bytes);
        throw QuantumOOM(fmt::format("MemoryManager arena allocation failed: bytes={}", bytes));
    }

    return Buffer{p, bytes, this};
}

bool MemoryManager::reserve(size_t bytes) noexcept {
    size_t prev = used_.load(std::memory_order_relaxed);
    while (true) {
        size_t next = prev + bytes;
        if (next > budget_) return false;
        if (used_.compare_exchange_weak(prev, next,
                                        std::memory_order_acq_rel,
                                        std::memory_order_relaxed)) {
            return true;
        }
    }
}

void MemoryManager::release(size_t bytes) noexcept {
    if (bytes == 0) return;
    used_.fetch_sub(bytes, std::memory_order_acq_rel);
}

void MemoryManager::deallocate(uint8_t* data, size_t bytes) noexcept {
    if (bytes == 0 || data == nullptr) return;
    try {
        std::lock_guard<std::mutex> lock(arenaMu_);
        freeLists_[bytes].push_back(data);
    } catch (...) {
        // Free-list bookkeeping failed (allocation) — drop the block; the
        // arena reclaims it at engine teardown, exactly as before recycling.
    }
    release(bytes);
}

} // namespace quantum
