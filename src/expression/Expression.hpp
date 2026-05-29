#pragma once

#include <memory>
#include <string>

#include "common/DataType.hpp"
#include "vector/ColumnBatch.hpp"
#include "vector/ColumnVector.hpp"

namespace quantum {

class MemoryManager;

class Expression;
using ExpressionPtr = std::unique_ptr<Expression>;

// Pull-based expression interface. eval() returns a ColumnVector of length
// batch.numRows() (NOT activeRows()). Slots outside the position list are
// undefined but their validity bit is cleared.
class Expression {
public:
    virtual ~Expression() = default;

    virtual ColumnVector eval(const ColumnBatch& batch, MemoryManager* mm) const = 0;
    virtual DataType     resultType(const Schema& inputSchema) const = 0;
    virtual std::string  toString() const = 0;
};

} // namespace quantum
