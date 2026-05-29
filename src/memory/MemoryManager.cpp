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
        p = arena_.allocate(bytes);
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

} // namespace quantum
