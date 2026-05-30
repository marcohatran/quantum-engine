#pragma once

#include <string>

#include "expression/Expression.hpp"

namespace quantum {

// SQL LIKE: `input LIKE 'pattern'` → BOOL column.
//
// Pattern semantics:
//   %  matches zero or more characters
//   _  matches exactly one character
// Match is anchored (entire input string must match the pattern).
//
// Null inputs propagate as null (validity bit cleared, value byte = 0).
// Phase 3 scope: pattern is a constant std::string; escape sequences not
// supported. `input` must evaluate to STRING.
class LikeExpr final : public Expression {
public:
    LikeExpr(ExpressionPtr input, std::string pattern);

    ColumnVector eval(const ColumnBatch& batch, MemoryManager* mm) const override;
    DataType     resultType(const Schema&) const override { return Bool; }
    std::string  toString() const override;

    const std::string& pattern() const noexcept { return pattern_; }

private:
    ExpressionPtr input_;
    std::string   pattern_;
};

} // namespace quantum
