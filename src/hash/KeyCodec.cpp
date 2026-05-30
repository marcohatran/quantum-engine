#include "hash/KeyCodec.hpp"

#include <cstring>

#include "common/Bits.hpp"
#include "common/Status.hpp"

namespace quantum {

KeyCodec::KeyCodec(const std::vector<DataType>& keyTypes)
    : types_(keyTypes) {
    if (keyTypes.empty()) {
        raise(ErrorCode::InvalidArgument, "KeyCodec: at least one key column required");
    }
    validityBytes_ = (keyTypes.size() + 7) / 8;
    if (validityBytes_ == 0) validityBytes_ = 1;
    size_t off = validityBytes_;
    offsets_.reserve(keyTypes.size());
    for (const auto& t : keyTypes) {
        const int32_t w = t.fixedWidth();
        if (w <= 0) {
            raise(ErrorCode::NotImplemented,
                  "KeyCodec: variable-width type {} not supported", t.toString());
        }
        offsets_.push_back(off);
        off += static_cast<size_t>(w);
    }
    keyBytes_ = off;
}

void KeyCodec::encode(const std::vector<const ColumnVector*>& keyCols,
                      size_t                                  row,
                      uint8_t*                                out) const {
    if (keyCols.size() != types_.size()) {
        raise(ErrorCode::InvalidArgument,
              "KeyCodec.encode: {} columns vs {} key types",
              keyCols.size(), types_.size());
    }
    std::memset(out, 0, keyBytes_);
    for (size_t c = 0; c < keyCols.size(); ++c) {
        const ColumnVector* col = keyCols[c];
        const bool          isV = col->validity().isValid(row);
        if (isV) {
            assignBit(out, c, true);  // validity bit c
            const int32_t w   = types_[c].fixedWidth();
            std::memcpy(out + offsets_[c],
                        col->values().data() + row * static_cast<size_t>(w),
                        static_cast<size_t>(w));
        }
        // else: bit stays 0, value bytes stay 0 (irrelevant for equality
        // since two NULLs in this column are equal — same validity bit, same zero bytes).
    }
}

bool KeyCodec::hasAnyNull(const uint8_t* key) const noexcept {
    for (size_t c = 0; c < types_.size(); ++c) {
        if (!getBit(key, c)) return true;
    }
    return false;
}

void KeyCodec::decode(const uint8_t* key,
                      size_t         colIdx,
                      ColumnVector&  outCol,
                      size_t         row) const {
    const bool isV = getBit(key, colIdx);
    outCol.validity().setValid(row, isV);
    if (!isV) return;
    const int32_t w = types_[colIdx].fixedWidth();
    std::memcpy(outCol.values().data() + row * static_cast<size_t>(w),
                key + offsets_[colIdx],
                static_cast<size_t>(w));
}

} // namespace quantum
