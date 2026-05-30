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

// MIN(int64) -> int64. Returns NULL for a group with no non-null inputs.
// State layout: [int64_t value][uint8_t hasValue][7 bytes padding].
class MinI64 final : public AggregateFunction {
public:
    explicit MinI64(ExpressionPtr input);

    size_t   stateSize() const override { return 16; }
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

// MAX(int64) -> int64. Returns NULL for a group with no non-null inputs.
class MaxI64 final : public AggregateFunction {
public:
    explicit MaxI64(ExpressionPtr input);

    size_t   stateSize() const override { return 16; }
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

// MIN/MAX(STRING) -> STRING. Returns NULL for a group with no non-null inputs.
//
// State layout: int32 index into a mutable per-aggregate vector of current
// best-seen strings. Index -1 means "no value yet". The aggregate is owned by
// a single operator and called single-threaded, so the mutable bookkeeping is
// safe without locking.
class MinString final : public AggregateFunction {
public:
    explicit MinString(ExpressionPtr input);

    size_t   stateSize() const override { return sizeof(int32_t); }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    DataType outputType() const override { return DataType{TypeId::STRING}; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr                    input_;
    mutable std::vector<std::string> values_;  // index lookup via state
};

class MaxString final : public AggregateFunction {
public:
    explicit MaxString(ExpressionPtr input);

    size_t   stateSize() const override { return sizeof(int32_t); }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    DataType outputType() const override { return DataType{TypeId::STRING}; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr                    input_;
    mutable std::vector<std::string> values_;
};

// AVG(int64) -> double. State: [double sum][int64 count]. Skips nulls.
// Returns NULL for a group with no non-null inputs.
class AvgI64 final : public AggregateFunction {
public:
    explicit AvgI64(ExpressionPtr input);

    size_t   stateSize() const override { return 16; }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    DataType outputType() const override { return Float64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
};

} // namespace quantum
