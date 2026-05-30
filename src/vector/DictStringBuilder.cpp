#include "vector/DictStringBuilder.hpp"

#include <cstring>

#include "common/Bits.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

DictStringBuilder::DictStringBuilder(MemoryManager* mm) : mm_(mm) {
    dictOffsets_.push_back(0);
}

int32_t DictStringBuilder::internCode(std::string_view s) {
    std::string key(s);
    auto it = dictIndex_.find(key);
    if (it != dictIndex_.end()) return it->second;

    const auto code = static_cast<int32_t>(dictOffsets_.size() - 1);
    bytes_.insert(bytes_.end(), s.begin(), s.end());
    dictOffsets_.push_back(static_cast<int32_t>(bytes_.size()));
    dictIndex_.emplace(std::move(key), code);
    return code;
}

void DictStringBuilder::append(std::string_view s) {
    codes_.push_back(internCode(s));
    validity_.push_back(true);
}

void DictStringBuilder::appendNull() {
    // Code value is irrelevant when validity is false; push 0 as a safe
    // placeholder so the codes array is fully initialized for SIMD scans.
    codes_.push_back(0);
    validity_.push_back(false);
}

ColumnVector DictStringBuilder::build() {
    const size_t n         = codes_.size();
    const size_t codesBytes = roundUp(n * sizeof(int32_t), kCacheLineBytes);
    const size_t offBytes   = roundUp(dictOffsets_.size() * sizeof(int32_t), kCacheLineBytes);
    const size_t valBytes   = roundUp(bytes_.size(), kCacheLineBytes);

    Buffer codesBuf = mm_->allocate(codesBytes);
    Buffer offBuf   = mm_->allocate(offBytes);
    Buffer valBuf   = mm_->allocate(valBytes);

    if (n > 0) std::memcpy(codesBuf.data(), codes_.data(), n * sizeof(int32_t));
    std::memcpy(offBuf.data(), dictOffsets_.data(), dictOffsets_.size() * sizeof(int32_t));
    if (!bytes_.empty()) std::memcpy(valBuf.data(), bytes_.data(), bytes_.size());

    const size_t dictN = dictOffsets_.size() - 1;
    ColumnVector out = ColumnVector::makeDictEncoded(
        DataType{TypeId::STRING}, n, dictN,
        std::move(codesBuf), std::move(offBuf), std::move(valBuf), mm_);

    out.validity().setAllValid();
    for (size_t i = 0; i < n; ++i) {
        if (!validity_[i]) out.validity().setValid(i, false);
    }

    // Reset for reuse (callers may repeatedly build into the same builder).
    codes_.clear();
    validity_.clear();
    bytes_.clear();
    dictOffsets_.clear();
    dictOffsets_.push_back(0);
    dictIndex_.clear();

    return out;
}

} // namespace quantum
