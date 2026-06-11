#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "common/Macros.hpp"

namespace quantum {

class MemoryManager;

// RAII move-only off-heap byte range. Guaranteed 64-byte aligned. The destructor
// notifies the owning MemoryManager so the budget is released.
class Buffer {
public:
    Buffer() noexcept = default;

    // Ownership-transferring constructor; used by MemoryManager::allocate.
    Buffer(uint8_t* data, size_t size, MemoryManager* owner) noexcept
        : data_(data), size_(size), owner_(owner) {}

    ~Buffer();

    QUANTUM_DISALLOW_COPY(Buffer);

    Buffer(Buffer&& other) noexcept { swap(other); }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    // Non-owning view over another buffer's bytes: destruction releases
    // nothing (no owner). The viewed Buffer must outlive the view.
    static Buffer view(const Buffer& src) noexcept {
        return Buffer(const_cast<uint8_t*>(src.data()), src.size(), nullptr);
    }

    uint8_t*       data()       noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }
    size_t         size() const noexcept { return size_; }
    bool           empty() const noexcept { return size_ == 0; }

    template <typename T>       T* as()       noexcept { return reinterpret_cast<T*>(data_); }
    template <typename T> const T* as() const noexcept { return reinterpret_cast<const T*>(data_); }

private:
    void release() noexcept;
    void swap(Buffer& other) noexcept {
        std::swap(data_,  other.data_);
        std::swap(size_,  other.size_);
        std::swap(owner_, other.owner_);
    }

    uint8_t*       data_  = nullptr;
    size_t         size_  = 0;
    MemoryManager* owner_ = nullptr;
};

} // namespace quantum
