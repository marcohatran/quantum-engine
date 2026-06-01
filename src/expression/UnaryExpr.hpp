#pragma once

#include <string>

#include "expression/Expression.hpp"

namespace quantum {

enum class UnaryOp { Not };

// UnaryOp::Not: boolean negation. Input must be BOOL. Validity is preserved
// row-by-row (NULL -> NULL); valid rows have their value bit flipped.
class UnaryExpr final : public Expression {
public:
    UnaryExpr(UnaryOp op, ExpressionPtr input)
        : op_(op), input_(std::move(input)) {}

    ColumnVector eval(const ColumnBatch& batch, MemoryManager* mm) const override;
    DataType     resultType(const Schema& inputSchema) const override;
    std::string  toString() const override;

    UnaryOp           op()    const noexcept { return op_; }
    const Expression& input() const noexcept { return *input_; }

private:
    UnaryOp        op_;
    ExpressionPtr  input_;
};

} // namespace quantum
