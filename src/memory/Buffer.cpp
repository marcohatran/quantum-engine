#include "memory/Buffer.hpp"

#include "memory/MemoryManager.hpp"

namespace quantum {

Buffer::~Buffer() {
    release();
}

void Buffer::release() noexcept {
    if (owner_ != nullptr && size_ > 0) {
        owner_->release(size_);
    }
    data_  = nullptr;
    size_  = 0;
    owner_ = nullptr;
}

} // namespace quantum
