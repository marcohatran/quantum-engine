#include "expression/Literal.hpp"

#include <fmt/format.h>

#include "common/Status.hpp"

namespace quantum {

namespace {
template <typename T>
void fillBroadcast(ColumnVector& out, T v) {
    T*           data = out.typed<T>();
    const size_t n    = out.numRows();
    for (size_t i = 0; i < n; ++i) data[i] = v;
}
} // namespace

ColumnVector Literal::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(type_, batch.numRows(), mm);
    out.validity().setAllValid();

    std::visit([&](auto v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
            uint8_t* d = out.typed<uint8_t>();
            for (size_t i = 0; i < out.numRows(); ++i) d[i] = v ? 1U : 0U;
        } else {
            fillBroadcast<T>(out, v);
        }
    }, value_);

    return out;
}

std::string Literal::toString() const {
    return std::visit([](auto v) -> std::string {
        return fmt::format("Literal({})", v);
    }, value_);
}

} // namespace quantum
