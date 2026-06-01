#include "expression/kernels/ComparisonKernels.hpp"

#include <cstring>
#include <string_view>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum::kernels {

namespace {

template <typename T, CmpOp Op>
inline uint8_t cmp(T a, T b) {
    if constexpr (Op == CmpOp::Eq)       return a == b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Lt)  return a <  b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Gt)  return a >  b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Lte) return a <= b ? 1U : 0U;
    else /* Gte */                       return a >= b ? 1U : 0U;
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
    if constexpr (Op == CmpOp::Eq)       return a == b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Lt)  return a <  b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Gt)  return a >  b ? 1U : 0U;
    else if constexpr (Op == CmpOp::Lte) return a <= b ? 1U : 0U;
    else /* Gte */                       return a >= b ? 1U : 0U;
}

// Fast path for STRING equality when both sides are dict-encoded. Precompute
// a translation table mapping every rhs dict code to the equivalent lhs code
// (or -1 if rhs's string is absent from lhs's dict), then compare codes for
// each row. Big win for `col = literal` since the literal-broadcast is
// dict-encoded with dictSize == 1.
void runStringDictEq(const ColumnVector& lhs, const ColumnVector& rhs,
                     ColumnVector& out, const PositionList& pos) {
    uint8_t* o = out.typed<uint8_t>();
    out.validity().setAllNull();

    const size_t   lhsDictN = lhs.dictSize();
    const size_t   rhsDictN = rhs.dictSize();
    const int32_t* lhsCodes = lhs.values().as<int32_t>();
    const int32_t* rhsCodes = rhs.values().as<int32_t>();

    // Translate each rhs code to its equivalent lhs code (or -1 on miss).
    std::vector<int32_t> rhsToLhs(rhsDictN, -1);
    for (size_t r = 0; r < rhsDictN; ++r) {
        const std::string_view rs(rhs.dictValues().as<char>() + rhs.dictOffsets().as<int32_t>()[r],
                                  static_cast<size_t>(rhs.dictOffsets().as<int32_t>()[r + 1] -
                                                      rhs.dictOffsets().as<int32_t>()[r]));
        for (size_t l = 0; l < lhsDictN; ++l) {
            const std::string_view ls(
                lhs.dictValues().as<char>() + lhs.dictOffsets().as<int32_t>()[l],
                static_cast<size_t>(lhs.dictOffsets().as<int32_t>()[l + 1] -
                                    lhs.dictOffsets().as<int32_t>()[l]));
            if (ls == rs) { rhsToLhs[r] = static_cast<int32_t>(l); break; }
        }
    }

    auto emitOne = [&](size_t i) {
        const int32_t lc = lhsCodes[i];
        const int32_t rc = rhsCodes[i];
        o[i] = (rhsToLhs[static_cast<size_t>(rc)] == lc) ? 1U : 0U;
    };

    if (pos.isIdentity()) {
        for (size_t i = 0; i < pos.size(); ++i) emitOne(i);
    } else {
        const auto idx = pos.indices();
        for (size_t k = 0; k < idx.size(); ++k) emitOne(static_cast<size_t>(idx[k]));
    }
}

template <CmpOp Op>
void runString(const ColumnVector& lhs, const ColumnVector& rhs,
               ColumnVector& out, const PositionList& pos) {
    if constexpr (Op == CmpOp::Eq) {
        if (lhs.isDictEncoded() && rhs.isDictEncoded()) {
            runStringDictEq(lhs, rhs, out, pos);
            return;
        }
    }

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
        case CmpOp::Eq:  return dispatch<CmpOp::Eq> (operandType, lhs, rhs, pos, mm);
        case CmpOp::Lt:  return dispatch<CmpOp::Lt> (operandType, lhs, rhs, pos, mm);
        case CmpOp::Gt:  return dispatch<CmpOp::Gt> (operandType, lhs, rhs, pos, mm);
        case CmpOp::Lte: return dispatch<CmpOp::Lte>(operandType, lhs, rhs, pos, mm);
        case CmpOp::Gte: return dispatch<CmpOp::Gte>(operandType, lhs, rhs, pos, mm);
    }
    raise(ErrorCode::Internal, "ComparisonKernel: unknown op");
}

} // namespace quantum::kernels
