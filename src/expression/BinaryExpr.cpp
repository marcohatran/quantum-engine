#include "expression/BinaryExpr.hpp"

#include <fmt/format.h>

#include "common/Status.hpp"
#include "expression/kernels/ArithmeticKernels.hpp"
#include "expression/kernels/ComparisonKernels.hpp"

namespace quantum {

namespace {

const char* opName(BinaryOp op) noexcept {
    switch (op) {
        case BinaryOp::Add: return "+";
        case BinaryOp::Sub: return "-";
        case BinaryOp::Mul: return "*";
        case BinaryOp::Eq:  return "=";
        case BinaryOp::Lt:  return "<";
        case BinaryOp::Gt:  return ">";
    }
    return "?";
}

kernels::ArithOp toArith(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return kernels::ArithOp::Add;
        case BinaryOp::Sub: return kernels::ArithOp::Sub;
        case BinaryOp::Mul: return kernels::ArithOp::Mul;
        default: raise(ErrorCode::Internal, "BinaryExpr: not an arithmetic op");
    }
}

kernels::CmpOp toCmp(BinaryOp op) {
    switch (op) {
        case BinaryOp::Eq: return kernels::CmpOp::Eq;
        case BinaryOp::Lt: return kernels::CmpOp::Lt;
        case BinaryOp::Gt: return kernels::CmpOp::Gt;
        default: raise(ErrorCode::Internal, "BinaryExpr: not a comparison op");
    }
}

} // namespace

ColumnVector BinaryExpr::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    ColumnVector lhsV = lhs_->eval(batch, mm);
    ColumnVector rhsV = rhs_->eval(batch, mm);

    if (lhsV.type() != rhsV.type()) {
        raise(ErrorCode::TypeMismatch,
              "BinaryExpr: lhs type {} != rhs type {}",
              lhsV.type().toString(), rhsV.type().toString());
    }

    if (isArithmeticOp(op_)) {
        return kernels::applyArithmetic(toArith(op_), lhsV.type(),
                                        lhsV, rhsV, batch.positions(), mm);
    }
    return kernels::applyComparison(toCmp(op_), lhsV.type(),
                                    lhsV, rhsV, batch.positions(), mm);
}

DataType BinaryExpr::resultType(const Schema& s) const {
    DataType lt = lhs_->resultType(s);
    DataType rt = rhs_->resultType(s);
    if (lt != rt) {
        raise(ErrorCode::TypeMismatch,
              "BinaryExpr: lhs type {} != rhs type {}", lt.toString(), rt.toString());
    }
    return isComparisonOp(op_) ? Bool : lt;
}

std::string BinaryExpr::toString() const {
    return fmt::format("({} {} {})", lhs_->toString(), opName(op_), rhs_->toString());
}

} // namespace quantum
