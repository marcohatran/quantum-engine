#pragma once

#include <string>

#include "expression/Expression.hpp"

namespace quantum {

enum class UnaryOp {
    Not,    // boolean negation; input must be BOOL; preserves validity
    IsNull, // returns BOOL — true iff input row is NULL. Output is always
            // valid (SQL `IS NULL` never returns NULL). Input may be any type.
};

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
