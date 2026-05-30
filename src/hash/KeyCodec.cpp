#include "hash/KeyCodec.hpp"

#include <cstring>

#include "common/Bits.hpp"
#include "common/Status.hpp"

namespace quantum {

KeyCodec::KeyCodec(const std::vector<DataType>& keyTypes)
    : types_(keyTypes), fixedWidth_(true) {
    if (keyTypes.empty()) {
        raise(ErrorCode::InvalidArgument, "KeyCodec: at least one key column required");
    }
    for (const auto& t : keyTypes) {
        if (!t.isFixedWidth()) {
            fixedWidth_ = false;
            break;
        }
    }

    validityBytes_ = (keyTypes.size() + 7) / 8;
    if (validityBytes_ == 0) validityBytes_ = 1;

    if (fixedWidth_) {
        size_t off = validityBytes_;
        offsets_.reserve(keyTypes.size());
        for (const auto& t : keyTypes) {
            offsets_.push_back(off);
            off += static_cast<size_t>(t.fixedWidth());
        }
        keyBytes_ = off;
    }
}

// ---------- Fixed-width path ----------

void KeyCodec::encode(const std::vector<const ColumnVector*>& keyCols,
                      size_t                                  row,
                      uint8_t*                                out) const {
    if (!fixedWidth_) {
        raise(ErrorCode::Internal,
              "KeyCodec::encode: codec is not fixed-width; use encodeVar");
    }
    if (keyCols.size() != types_.size()) {
        raise(ErrorCode::InvalidArgument,
              "KeyCodec.encode: {} columns vs {} key types",
              keyCols.size(), types_.size());
    }
    std::memset(out, 0, keyBytes_);
    for (size_t c = 0; c < keyCols.size(); ++c) {
        const ColumnVector* col = keyCols[c];
        if (col->validity().isValid(row)) {
            assignBit(out, c, true);
            const int32_t w = types_[c].fixedWidth();
            std::memcpy(out + offsets_[c],
                        col->values().data() + row * static_cast<size_t>(w),
                        static_cast<size_t>(w));
        }
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
                      size_t         outRow) const {
    if (!fixedWidth_) {
        raise(ErrorCode::Internal,
              "KeyCodec::decode: codec is not fixed-width; use decodeVar");
    }
    const bool isV = getBit(key, colIdx);
    outCol.validity().setValid(outRow, isV);
    if (!isV) return;
    const int32_t w = types_[colIdx].fixedWidth();
    std::memcpy(outCol.values().data() + outRow * static_cast<size_t>(w),
                key + offsets_[colIdx],
                static_cast<size_t>(w));
}

// ---------- Variable-width path ----------

void KeyCodec::encodeVar(const std::vector<const ColumnVector*>& keyCols,
                         size_t                                  row,
                         std::vector<uint8_t>&                   out) const {
    if (keyCols.size() != types_.size()) {
        raise(ErrorCode::InvalidArgument,
              "KeyCodec.encodeVar: {} columns vs {} key types",
              keyCols.size(), types_.size());
    }
    out.clear();
    out.resize(validityBytes_, 0);

    for (size_t c = 0; c < keyCols.size(); ++c) {
        const ColumnVector* col   = keyCols[c];
        const bool          valid = col->validity().isValid(row);
        if (valid) assignBit(out.data(), c, true);

        if (types_[c].isFixedWidth()) {
            const int32_t w = types_[c].fixedWidth();
            const size_t  oldSize = out.size();
            out.resize(oldSize + static_cast<size_t>(w), 0);
            if (valid) {
                std::memcpy(out.data() + oldSize,
                            col->values().data() + row * static_cast<size_t>(w),
                            static_cast<size_t>(w));
            }
        } else {
            // STRING. Length prefix + bytes (length = 0 when NULL).
            std::string_view s;
            if (valid) s = col->stringAt(row);
            const int32_t len = static_cast<int32_t>(s.size());
            const size_t oldSize = out.size();
            out.resize(oldSize + sizeof(int32_t) + s.size());
            std::memcpy(out.data() + oldSize, &len, sizeof(len));
            if (!s.empty()) {
                std::memcpy(out.data() + oldSize + sizeof(int32_t),
                            s.data(), s.size());
            }
        }
    }
}

bool KeyCodec::hasAnyNullVar(const uint8_t* key, size_t /*len*/) const noexcept {
    for (size_t c = 0; c < types_.size(); ++c) {
        if (!getBit(key, c)) return true;
    }
    return false;
}

void KeyCodec::decodeVar(const uint8_t* key, size_t /*len*/, const VarVisitor& visit) const {
    size_t pos = validityBytes_;
    for (size_t c = 0; c < types_.size(); ++c) {
        const bool valid = getBit(key, c);
        if (types_[c].isFixedWidth()) {
            const size_t w = static_cast<size_t>(types_[c].fixedWidth());
            visit(c, valid, key + pos, w);
            pos += w;
        } else {
            int32_t slen;
            std::memcpy(&slen, key + pos, sizeof(slen));
            pos += sizeof(int32_t);
            visit(c, valid, key + pos, static_cast<size_t>(slen));
            pos += static_cast<size_t>(slen);
        }
    }
}

} // namespace quantum
