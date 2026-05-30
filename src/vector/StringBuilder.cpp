#include "vector/StringBuilder.hpp"

#include <cstring>

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

StringBuilder::StringBuilder(MemoryManager* mm) : mm_(mm) {
    offsets_.push_back(0);
}

void StringBuilder::append(std::string_view s) {
    bytes_.insert(bytes_.end(), s.begin(), s.end());
    offsets_.push_back(static_cast<int32_t>(bytes_.size()));
    validity_.push_back(true);
}

void StringBuilder::appendNull() {
    offsets_.push_back(static_cast<int32_t>(bytes_.size()));  // length 0
    validity_.push_back(false);
}

ColumnVector StringBuilder::build() {
    const size_t n        = validity_.size();
    const size_t valBytes = roundUp(bytes_.size(), kCacheLineBytes);
    const size_t offBytes = roundUp((n + 1) * sizeof(int32_t), kCacheLineBytes);

    Buffer values  = mm_->allocate(valBytes);
    Buffer offsets = mm_->allocate(offBytes);
    if (!bytes_.empty()) {
        std::memcpy(values.data(), bytes_.data(), bytes_.size());
    }
    std::memcpy(offsets.data(), offsets_.data(), (n + 1) * sizeof(int32_t));

    ColumnVector out = ColumnVector::makeVariableWidth(
        DataType{TypeId::STRING}, n, std::move(offsets), std::move(values), mm_);
    out.validity().setAllValid();
    for (size_t i = 0; i < n; ++i) {
        if (!validity_[i]) out.validity().setValid(i, false);
    }

    bytes_.clear();
    offsets_.clear();
    offsets_.push_back(0);
    validity_.clear();
    return out;
}

} // namespace quantum
