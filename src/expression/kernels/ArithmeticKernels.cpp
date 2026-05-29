#include "expression/kernels/ArithmeticKernels.hpp"

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum::kernels {

namespace {

template <typename T, ArithOp Op>
inline T apply(T a, T b) {
    if constexpr (Op == ArithOp::Add) return static_cast<T>(a + b);
    else if constexpr (Op == ArithOp::Sub) return static_cast<T>(a - b);
    else /* Mul */                       return static_cast<T>(a * b);
}

template <typename T, ArithOp Op>
void run(const ColumnVector& lhs, const ColumnVector& rhs,
         ColumnVector& out, const PositionList& pos) {
    const T* a = lhs.typed<T>();
    const T* b = rhs.typed<T>();
    T*       o = out.typed<T>();

    // Start every output slot invalid; we only validate the active ones below.
    out.validity().setAllNull();

    if (pos.isIdentity()) {
        const size_t n = pos.size();
        for (size_t i = 0; i < n; ++i) {
            o[i] = apply<T, Op>(a[i], b[i]);
        }
    } else {
        const auto idx = pos.indices();
        for (size_t k = 0; k < idx.size(); ++k) {
            const auto i = static_cast<size_t>(idx[k]);
            o[i] = apply<T, Op>(a[i], b[i]);
        }
    }
}

template <ArithOp Op>
ColumnVector dispatch(DataType type,
                      const ColumnVector& lhs, const ColumnVector& rhs,
                      const PositionList& pos, MemoryManager* mm) {
    ColumnVector out = ColumnVector::make(type, lhs.numRows(), mm);
    switch (type.id) {
        case TypeId::INT32:   run<int32_t, Op>(lhs, rhs, out, pos); break;
        case TypeId::INT64:   run<int64_t, Op>(lhs, rhs, out, pos); break;
        case TypeId::FLOAT64: run<double,  Op>(lhs, rhs, out, pos); break;
        default:
            raise(ErrorCode::TypeMismatch,
                  "ArithmeticKernel: unsupported type {}", type.toString());
    }

    // Propagate validity from both inputs into the output's active slots only.
    // We do this by ANDing the input validity buffers into the output (which
    // already has every active slot's bit cleared above). The cheap approach:
    // set output validity to lhs.validity AND rhs.validity, then for any rows
    // outside the position list clear the bit explicitly.
    if (pos.isIdentity()) {
        std::memcpy(out.validity().data(), lhs.validity().data(), lhs.validity().sizeBytes());
        out.validity().andInPlace(rhs.validity());
    } else {
        // Active slots only — bit-by-bit (rare path: post-filter expression).
        for (size_t k = 0; k < pos.size(); ++k) {
            const auto i = static_cast<size_t>(pos.at(k));
            out.validity().setValid(i, lhs.validity().isValid(i) && rhs.validity().isValid(i));
        }
    }

    return out;
}

} // namespace

ColumnVector applyArithmetic(ArithOp op, DataType type,
                             const ColumnVector& lhs, const ColumnVector& rhs,
                             const PositionList& pos, MemoryManager* mm) {
    switch (op) {
        case ArithOp::Add: return dispatch<ArithOp::Add>(type, lhs, rhs, pos, mm);
        case ArithOp::Sub: return dispatch<ArithOp::Sub>(type, lhs, rhs, pos, mm);
        case ArithOp::Mul: return dispatch<ArithOp::Mul>(type, lhs, rhs, pos, mm);
    }
    raise(ErrorCode::Internal, "ArithmeticKernel: unknown op");
}

} // namespace quantum::kernels
