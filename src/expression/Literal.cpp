#include "expression/Literal.hpp"

#include <cstring>

#include <fmt/format.h>

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

namespace {
template <typename T>
void fillBroadcast(ColumnVector& out, T v) {
    T*           data = out.typed<T>();
    const size_t n    = out.numRows();
    for (size_t i = 0; i < n; ++i) data[i] = v;
}

// Broadcast a string literal as a dict-encoded ColumnVector with a single
// dictionary entry. Codes are all zero, so the column footprint is
// 4 bytes/row regardless of string length, and dict-aware kernels (e.g. the
// dict-vs-dict equality fast path) can detect the broadcast trivially via
// dictSize() == 1.
ColumnVector broadcastString(std::string_view s, size_t n, MemoryManager* mm) {
    Buffer codes = mm->allocate(roundUp(n * sizeof(int32_t), kCacheLineBytes));
    Buffer dictOff = mm->allocate(roundUp(2 * sizeof(int32_t), kCacheLineBytes));
    Buffer dictVal = mm->allocate(roundUp(s.size(), kCacheLineBytes));

    // All codes default-zero; values_ buffer is already zeroed by the allocator
    // contract? Not guaranteed — explicitly memset to be safe.
    std::memset(codes.data(), 0, codes.size());

    int32_t off[2] = {0, static_cast<int32_t>(s.size())};
    std::memcpy(dictOff.data(), off, sizeof(off));
    if (!s.empty()) std::memcpy(dictVal.data(), s.data(), s.size());

    ColumnVector out = ColumnVector::makeDictEncoded(
        DataType{TypeId::STRING}, n, /*dictSize=*/1,
        std::move(codes), std::move(dictOff), std::move(dictVal), mm);
    out.validity().setAllValid();
    return out;
}
} // namespace

ColumnVector Literal::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    if (std::holds_alternative<std::string>(value_)) {
        return broadcastString(std::get<std::string>(value_), batch.numRows(), mm);
    }

    ColumnVector out = ColumnVector::make(type_, batch.numRows(), mm);
    out.validity().setAllValid();

    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
            uint8_t* d = out.typed<uint8_t>();
            for (size_t i = 0; i < out.numRows(); ++i) d[i] = v ? 1U : 0U;
        } else if constexpr (std::is_same_v<T, std::string>) {
            // handled above
        } else {
            fillBroadcast<T>(out, v);
        }
    }, value_);

    return out;
}

std::string Literal::toString() const {
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return fmt::format("Literal(\"{}\")", v);
        } else {
            return fmt::format("Literal({})", v);
        }
    }, value_);
}

} // namespace quantum
