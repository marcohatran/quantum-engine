#include "expression/UnaryExpr.hpp"

#include <cstring>

#include <fmt/format.h>

#include "common/Status.hpp"

namespace quantum {

ColumnVector UnaryExpr::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    ColumnVector inV = input_->eval(batch, mm);
    ColumnVector out = ColumnVector::make(Bool, inV.numRows(), mm);
    const auto&  pos = batch.positions();

    if (op_ == UnaryOp::Not) {
        if (inV.type().id != TypeId::BOOL) {
            raise(ErrorCode::TypeMismatch,
                  "UnaryExpr::Not requires BOOL input, got {}", inV.type().toString());
        }
        std::memcpy(out.validity().data(), inV.validity().data(), inV.validity().sizeBytes());
        const uint8_t* in = inV.typed<uint8_t>();
        uint8_t*       o  = out.typed<uint8_t>();
        auto step = [&](size_t i) { o[i] = static_cast<uint8_t>((~in[i]) & 1U); };
        if (pos.isIdentity()) {
            for (size_t i = 0; i < pos.size(); ++i) step(i);
        } else {
            const auto idx = pos.indices();
            for (size_t k = 0; k < idx.size(); ++k) step(static_cast<size_t>(idx[k]));
        }
        return out;
    }

    // IsNull: output is BOOL, always valid. Reads only the input validity
    // bitmap; input value bytes are ignored. Works for any input type.
    out.validity().setAllValid();
    uint8_t* o = out.typed<uint8_t>();
    auto step = [&](size_t i) {
        o[i] = inV.validity().isValid(i) ? 0U : 1U;
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
        case UnaryOp::Not:    return fmt::format("(NOT {})",      input_->toString());
        case UnaryOp::IsNull: return fmt::format("({} IS NULL)",  input_->toString());
    }
    return "?";
}

} // namespace quantum
