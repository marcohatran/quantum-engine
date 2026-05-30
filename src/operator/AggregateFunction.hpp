#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "common/DataType.hpp"
#include "expression/Expression.hpp"
#include "vector/ColumnVector.hpp"

namespace quantum {

class MemoryManager;

// One aggregate's contribution to a per-group state row. The
// HashAggregateOperator owns a slab of `stateBytes` per group and tells each
// AggregateFunction where its slice lives via `stateOffset`.
class AggregateFunction {
public:
    virtual ~AggregateFunction() = default;

    // Bytes this aggregate needs inside the per-group state row.
    virtual size_t   stateSize()  const = 0;

    // Initialize an aggregate state at `state` to its identity element
    // (e.g. SUM -> 0, COUNT -> 0, MIN -> +INF). Called on first sight of a group.
    virtual void     init(uint8_t* state) const = 0;

    // Fold `input` (the AggregateFunction's pre-evaluated input column, or
    // nullptr for COUNT(*)) into `state` for active rows under `pos`.
    // `groupSlots[i]` is the per-row pointer into the per-group state slab
    // for input row `i`. `pos` is the input batch's position list.
    virtual void     update(uint8_t* const*       groupSlots,
                            const ColumnVector*   input,
                            const PositionList&   pos) const = 0;

    // Produce the final output column from the collected `states` (one
    // pointer per output row).
    virtual ColumnVector finalize(const std::vector<const uint8_t*>& states,
                                  MemoryManager* mm) const = 0;

    virtual DataType outputType() const = 0;

    // Optional input expression — nullopt for COUNT(*).
    virtual const Expression* inputExpr() const = 0;
};

using AggregateFunctionPtr = std::unique_ptr<AggregateFunction>;

// SUM(int64) -> int64. Skips null inputs.
class SumI64 final : public AggregateFunction {
public:
    explicit SumI64(ExpressionPtr input);

    size_t   stateSize() const override { return sizeof(int64_t); }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    DataType outputType() const override { return Int64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
};

// COUNT(*) -> int64. Counts every active row including nulls.
class CountStar final : public AggregateFunction {
public:
    size_t   stateSize() const override { return sizeof(int64_t); }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    DataType outputType() const override { return Int64; }
    const Expression* inputExpr() const override { return nullptr; }
};

} // namespace quantum
