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

ColumnVector broadcastString(std::string_view s, size_t n, MemoryManager* mm) {
    const size_t valBytes = roundUp(s.size() * n, kCacheLineBytes);
    const size_t offBytes = roundUp((n + 1) * sizeof(int32_t), kCacheLineBytes);
    Buffer values  = mm->allocate(valBytes);
    Buffer offsets = mm->allocate(offBytes);

    int32_t* off = offsets.as<int32_t>();
    char*    val = values.as<char>();
    for (size_t i = 0; i < n; ++i) {
        off[i] = static_cast<int32_t>(i * s.size());
        if (!s.empty()) std::memcpy(val + i * s.size(), s.data(), s.size());
    }
    off[n] = static_cast<int32_t>(n * s.size());

    ColumnVector out = ColumnVector::makeVariableWidth(
        DataType{TypeId::STRING}, n, std::move(offsets), std::move(values), mm);
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
