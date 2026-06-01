#include "expression/UnaryExpr.hpp"

#include <cstring>

#include <fmt/format.h>

#include "common/Status.hpp"

namespace quantum {

ColumnVector UnaryExpr::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    ColumnVector inV = input_->eval(batch, mm);
    if (inV.type().id != TypeId::BOOL) {
        raise(ErrorCode::TypeMismatch,
              "UnaryExpr::Not requires BOOL input, got {}", inV.type().toString());
    }

    ColumnVector out = ColumnVector::make(Bool, inV.numRows(), mm);
    // Preserve validity directly — NOT NULL is NULL.
    std::memcpy(out.validity().data(), inV.validity().data(), inV.validity().sizeBytes());

    const uint8_t* in  = inV.typed<uint8_t>();
    uint8_t*       o   = out.typed<uint8_t>();
    const auto&    pos = batch.positions();

    auto step = [&](size_t i) {
        // Flip the value bit. For invalid (NULL) rows the value byte is
        // garbage but the validity bit is already cleared above, so the
        // garbage never escapes a downstream Filter / consumer.
        o[i] = static_cast<uint8_t>((~in[i]) & 1U);
    };

    if (pos.isIdentity()) {
        for (size_t i = 0; i < pos.size(); ++i) step(i);
    } else {
        const auto idx = pos.indices();
        for (size_t k = 0; k < idx.size(); ++k) step(static_cast<size_t>(idx[k]));
    }
    return out;
}

DataType UnaryExpr::resultType(const Schema& /*s*/) const { return Bool; }

std::string UnaryExpr::toString() const {
    switch (op_) {
        case UnaryOp::Not: return fmt::format("(NOT {})", input_->toString());
    }
    return "?";
}

} // namespace quantum
