#pragma once

#include <cstdint>
#include <variant>

#include "expression/Expression.hpp"

namespace quantum {

// Broadcast-typed scalar. eval() returns a ColumnVector of length
// batch.numRows() with every slot set to the literal value.
class Literal final : public Expression {
public:
    using Value = std::variant<bool, int32_t, int64_t, double>;

    explicit Literal(int32_t v) : value_(v), type_(Int32)   {}
    explicit Literal(int64_t v) : value_(v), type_(Int64)   {}
    explicit Literal(double  v) : value_(v), type_(Float64) {}
    explicit Literal(bool    v) : value_(v), type_(Bool)    {}

    ColumnVector eval(const ColumnBatch& batch, MemoryManager* mm) const override;
    DataType     resultType(const Schema&) const override { return type_; }
    std::string  toString() const override;

    DataType type() const noexcept { return type_; }
    const Value& value() const noexcept { return value_; }

private:
    Value    value_;
    DataType type_;
};

} // namespace quantum
