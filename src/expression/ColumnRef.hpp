#pragma once

#include <cstddef>

#include "expression/Expression.hpp"

namespace quantum {

// References an input column by index. Phase 1 returns a deep copy of the
// referenced column; the zero-copy ColumnVectorView abstraction is deferred.
class ColumnRef final : public Expression {
public:
    explicit ColumnRef(size_t index) : index_(index) {}

    ColumnVector eval(const ColumnBatch& batch, MemoryManager* mm) const override;
    DataType     resultType(const Schema& inputSchema) const override {
        return inputSchema.field(index_).type;
    }
    std::string  toString() const override;

    size_t index() const noexcept { return index_; }

private:
    size_t index_;
};

} // namespace quantum
