#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <unordered_set>

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

    // Combine two per-group states into one. `dst` accumulates `src`.
    // Used by spill-and-merge: a spilled partial result is merged back into
    // the current in-memory state. Aggregates whose state isn't fully self-
    // contained (e.g. MinString stores its values out-of-line) raise
    // NotImplemented for v1.
    virtual void merge(uint8_t* dst, const uint8_t* src) const = 0;
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
    void     merge(uint8_t* dst, const uint8_t* src) const override;
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
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Int64; }
    const Expression* inputExpr() const override { return nullptr; }
};

// COUNT(col) -> int64. Counts active rows whose input value is non-null.
// Type-agnostic for the input (only the validity bitmap is read).
class CountColumn final : public AggregateFunction {
public:
    explicit CountColumn(ExpressionPtr input);

    size_t   stateSize() const override { return sizeof(int64_t); }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Int64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
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
    void     merge(uint8_t* dst, const uint8_t* src) const override;
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
    void     merge(uint8_t* dst, const uint8_t* src) const override;
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
    void     merge(uint8_t* dst, const uint8_t* src) const override;
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
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return DataType{TypeId::STRING}; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr                    input_;
    mutable std::vector<std::string> values_;
};

// COUNT(DISTINCT col) where col is INT64. Per group, tracks the set of
// distinct non-null int64 values seen and emits the set's cardinality.
// State layout: int32 index into a mutable vector of per-group sets owned
// by this aggregate function. Init grows the vector by one slot and
// returns its position. Operator is single-threaded, so the mutable
// bookkeeping is safe without locking.
class CountDistinctI64 final : public AggregateFunction {
public:
    explicit CountDistinctI64(ExpressionPtr input);

    size_t   stateSize() const override { return sizeof(int32_t); }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Int64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr                                  input_;
    mutable std::vector<std::unordered_set<int64_t>> sets_;
};

// SUM(double) -> double. Skips null inputs.
class SumF64 final : public AggregateFunction {
public:
    explicit SumF64(ExpressionPtr input);

    size_t   stateSize() const override { return sizeof(double); }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Float64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
};

// MIN(double) -> double. State: [double value][uint8_t hasValue].
class MinF64 final : public AggregateFunction {
public:
    explicit MinF64(ExpressionPtr input);

    size_t   stateSize() const override { return 16; }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Float64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
};

// MAX(double) -> double.
class MaxF64 final : public AggregateFunction {
public:
    explicit MaxF64(ExpressionPtr input);

    size_t   stateSize() const override { return 16; }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Float64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
};

// AVG(double) -> double. State: [double sum][int64 count].
class AvgF64 final : public AggregateFunction {
public:
    explicit AvgF64(ExpressionPtr input);

    size_t   stateSize() const override { return 16; }
    void     init(uint8_t* state) const override;
    void     update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const override;
    ColumnVector finalize(const std::vector<const uint8_t*>& states,
                          MemoryManager* mm) const override;
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Float64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
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
    void     merge(uint8_t* dst, const uint8_t* src) const override;
    DataType outputType() const override { return Float64; }
    const Expression* inputExpr() const override { return input_.get(); }

private:
    ExpressionPtr input_;
};

} // namespace quantum
