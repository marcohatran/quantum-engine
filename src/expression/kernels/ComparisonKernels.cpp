#include "expression/kernels/ComparisonKernels.hpp"

#include <cstring>
#include <string_view>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum::kernels {

namespace {

template <typename T, CmpOp Op>
inline uint8_t cmp(T a, T b) {
    if constexpr (Op == CmpOp::Eq) return a == b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Lt) return a <  b ? 1U : 0U;
    else /* Gt */                       return a >  b ? 1U : 0U;
}

template <typename T, CmpOp Op>
void run(const ColumnVector& lhs, const ColumnVector& rhs,
         ColumnVector& out, const PositionList& pos) {
    const T* a = lhs.typed<T>();
    const T* b = rhs.typed<T>();
    uint8_t* o = out.typed<uint8_t>();

    out.validity().setAllNull();

    if (pos.isIdentity()) {
        const size_t n = pos.size();
        for (size_t i = 0; i < n; ++i) {
            o[i] = cmp<T, Op>(a[i], b[i]);
        }
    } else {
        const auto idx = pos.indices();
        for (size_t k = 0; k < idx.size(); ++k) {
            const auto i = static_cast<size_t>(idx[k]);
            o[i] = cmp<T, Op>(a[i], b[i]);
        }
    }
}

template <CmpOp Op>
inline uint8_t cmpString(std::string_view a, std::string_view b) {
    if constexpr (Op == CmpOp::Eq) return a == b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Lt) return a <  b ? 1U : 0U;
    else /* Gt */                       return a >  b ? 1U : 0U;
}

template <CmpOp Op>
void runString(const ColumnVector& lhs, const ColumnVector& rhs,
               ColumnVector& out, const PositionList& pos) {
    uint8_t* o = out.typed<uint8_t>();
    out.validity().setAllNull();

    if (pos.isIdentity()) {
        for (size_t i = 0; i < pos.size(); ++i) {
            o[i] = cmpString<Op>(lhs.stringAt(i), rhs.stringAt(i));
        }
    } else {
        const auto idx = pos.indices();
        for (size_t k = 0; k < idx.size(); ++k) {
            const auto i = static_cast<size_t>(idx[k]);
            o[i] = cmpString<Op>(lhs.stringAt(i), rhs.stringAt(i));
        }
    }
}

template <CmpOp Op>
ColumnVector dispatch(DataType type,
                      const ColumnVector& lhs, const ColumnVector& rhs,
                      const PositionList& pos, MemoryManager* mm) {
    ColumnVector out = ColumnVector::make(Bool, lhs.numRows(), mm);
    switch (type.id) {
        case TypeId::INT32:   run<int32_t, Op>(lhs, rhs, out, pos); break;
        case TypeId::INT64:   run<int64_t, Op>(lhs, rhs, out, pos); break;
        case TypeId::FLOAT64: run<double,  Op>(lhs, rhs, out, pos); break;
        case TypeId::BOOL:    run<uint8_t, Op>(lhs, rhs, out, pos); break;
        case TypeId::STRING:  runString<Op>(lhs, rhs, out, pos);   break;
        default:
            raise(ErrorCode::TypeMismatch,
                  "ComparisonKernel: unsupported type {}", type.toString());
    }

    if (pos.isIdentity()) {
        std::memcpy(out.validity().data(), lhs.validity().data(), lhs.validity().sizeBytes());
        out.validity().andInPlace(rhs.validity());
    } else {
        for (size_t k = 0; k < pos.size(); ++k) {
            const auto i = static_cast<size_t>(pos.at(k));
            out.validity().setValid(i, lhs.validity().isValid(i) && rhs.validity().isValid(i));
        }
    }

    return out;
}

} // namespace

ColumnVector applyComparison(CmpOp op, DataType operandType,
                             const ColumnVector& lhs, const ColumnVector& rhs,
                             const PositionList& pos, MemoryManager* mm) {
    switch (op) {
        case CmpOp::Eq: return dispatch<CmpOp::Eq>(operandType, lhs, rhs, pos, mm);
        case CmpOp::Lt: return dispatch<CmpOp::Lt>(operandType, lhs, rhs, pos, mm);
        case CmpOp::Gt: return dispatch<CmpOp::Gt>(operandType, lhs, rhs, pos, mm);
    }
    raise(ErrorCode::Internal, "ComparisonKernel: unknown op");
}

} // namespace quantum::kernels
