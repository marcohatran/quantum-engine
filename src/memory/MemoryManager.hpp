#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "common/Macros.hpp"
#include "memory/Arena.hpp"

namespace quantum {

class Buffer;

// Single source of memory truth for a query. Holds a budget (bytes), backed by
// an Arena. Buffer destructors call release() to decrement used().
class MemoryManager {
public:
    explicit MemoryManager(size_t budgetBytes);
    ~MemoryManager() = default;

    QUANTUM_DISALLOW_COPY(MemoryManager);

    // Allocates `bytes` of 64-byte-aligned memory. Throws QuantumOOM on budget overflow.
    Buffer allocate(size_t bytes);

    // Reserves budget without allocating. Returns false if the reservation would
    // overflow the budget; otherwise increments used() by `bytes`.
    bool reserve(size_t bytes) noexcept;

    // Releases `bytes` back to the budget. Called from Buffer's destructor.
    void release(size_t bytes) noexcept;

    size_t used()   const noexcept { return used_.load(std::memory_order_relaxed); }
    size_t budget() const noexcept { return budget_; }

    // Fraction of the budget currently in use (0.0 .. 1.0). Operators that
    // support spill poll this to decide when to flush partial state.
    double usedFraction() const noexcept {
        return budget_ == 0 ? 0.0
                            : static_cast<double>(used()) / static_cast<double>(budget_);
    }

    // Phase 1 stub: no spillable operators registered yet.
    bool spill() noexcept { return false; }

private:
    Arena               arena_;
    std::mutex          arenaMu_;
    std::atomic<size_t> used_{0};
    size_t              budget_;
};

} // namespace quantum
