#pragma once

#include "expression/Expression.hpp"

namespace quantum {

enum class BinaryOp { Add, Sub, Mul, Eq, Lt, Gt, Lte, Gte };

inline bool isArithmeticOp(BinaryOp op) noexcept {
    return op == BinaryOp::Add || op == BinaryOp::Sub || op == BinaryOp::Mul;
}

inline bool isComparisonOp(BinaryOp op) noexcept {
    return op == BinaryOp::Eq  || op == BinaryOp::Lt  || op == BinaryOp::Gt ||
           op == BinaryOp::Lte || op == BinaryOp::Gte;
}

class BinaryExpr final : public Expression {
public:
    BinaryExpr(BinaryOp op, ExpressionPtr lhs, ExpressionPtr rhs)
        : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    ColumnVector eval(const ColumnBatch& batch, MemoryManager* mm) const override;
    DataType     resultType(const Schema& inputSchema) const override;
    std::string  toString() const override;

    BinaryOp           op()  const noexcept { return op_; }
    const Expression&  lhs() const noexcept { return *lhs_; }
    const Expression&  rhs() const noexcept { return *rhs_; }

private:
    BinaryOp       op_;
    ExpressionPtr  lhs_;
    ExpressionPtr  rhs_;
};

} // namespace quantum
