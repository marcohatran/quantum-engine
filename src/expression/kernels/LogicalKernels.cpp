#include "expression/kernels/LogicalKernels.hpp"

#include "common/Status.hpp"

namespace quantum::kernels {

namespace {

template <LogicalOp Op>
inline void evalOne(uint8_t  lhsValid, uint8_t lhsVal,
                    uint8_t  rhsValid, uint8_t rhsVal,
                    uint8_t& outValue, uint8_t& outValid) {
    const uint8_t lT = lhsValid & (lhsVal & 1U);  // "left is TRUE"
    const uint8_t rT = rhsValid & (rhsVal & 1U);  // "right is TRUE"
    const uint8_t lF = lhsValid & ((~lhsVal) & 1U);
    const uint8_t rF = rhsValid & ((~rhsVal) & 1U);

    if constexpr (Op == LogicalOp::And) {
        outValue = static_cast<uint8_t>(lT & rT);
        // VALID when both sides valid OR either side is definitely false.
        outValid = static_cast<uint8_t>((lhsValid & rhsValid) | lF | rF);
    } else { /* Or */
        outValue = static_cast<uint8_t>(lT | rT);
        // VALID when both sides valid OR either side is definitely true.
        outValid = static_cast<uint8_t>((lhsValid & rhsValid) | lT | rT);
    }
}

template <LogicalOp Op>
void run(const ColumnVector& lhs, const ColumnVector& rhs,
         ColumnVector& out, const PositionList& pos) {
    const uint8_t* a = lhs.typed<uint8_t>();
    const uint8_t* b = rhs.typed<uint8_t>();
    uint8_t*       o = out.typed<uint8_t>();

    out.validity().setAllNull();

    auto step = [&](size_t i) {
        uint8_t lhsValid = lhs.validity().isValid(i) ? 1U : 0U;
        uint8_t rhsValid = rhs.validity().isValid(i) ? 1U : 0U;
        uint8_t value;
        uint8_t valid;
        evalOne<Op>(lhsValid, a[i], rhsValid, b[i], value, valid);
        o[i] = value;
        out.validity().setValid(i, valid != 0U);
    };

    if (pos.isIdentity()) {
        for (size_t i = 0; i < pos.size(); ++i) step(i);
    } else {
        const auto idx = pos.indices();
        for (size_t k = 0; k < idx.size(); ++k) step(static_cast<size_t>(idx[k]));
    }
}

} // namespace

ColumnVector applyLogical(LogicalOp           op,
                          const ColumnVector& lhs,
                          const ColumnVector& rhs,
                          const PositionList& pos,
                          MemoryManager*      mm) {
    if (lhs.type().id != TypeId::BOOL || rhs.type().id != TypeId::BOOL) {
        raise(ErrorCode::TypeMismatch,
              "LogicalKernel: requires BOOL inputs, got lhs={} rhs={}",
              lhs.type().toString(), rhs.type().toString());
    }
    ColumnVector out = ColumnVector::make(Bool, lhs.numRows(), mm);
    switch (op) {
        case LogicalOp::And: run<LogicalOp::And>(lhs, rhs, out, pos); break;
        case LogicalOp::Or:  run<LogicalOp::Or> (lhs, rhs, out, pos); break;
    }
    return out;
}

} // namespace quantum::kernels
