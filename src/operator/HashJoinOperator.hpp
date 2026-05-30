#pragma once

#include <memory>
#include <string>
#include <vector>

#include "expression/Expression.hpp"
#include "hash/HashTable.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Inner equi-join on a single int64 key. Two-phase execution:
//   1. Build (first getNext call): drain the build side, materialize a dense
//      row store, and populate a HashTable mapping each non-null build key to
//      the head of a chain of build-row indices (chain links in next_[]).
//   2. Probe (subsequent calls): for each probe batch, walk the chain for
//      each active probe row's key and emit one output batch whose columns
//      are the build-side columns followed by the probe-side columns.
//
// Phase 2 scope: inner join only, single int64 key on each side, all input
// columns passed through. Null keys never match (SQL semantics). Outer joins,
// composite keys, string keys, and spill come in later slices.
class HashJoinOperator final : public Operator {
public:
    HashJoinOperator(MemoryManager*          mm,
                     OperatorPtr             buildSide,
                     OperatorPtr             probeSide,
                     ExpressionPtr           buildKey,
                     ExpressionPtr           probeKey,
                     std::vector<std::string> outputNames);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    void                build();
    std::optional<ColumnBatch> probeOne();

    OperatorPtr   build_;
    OperatorPtr   probe_;
    ExpressionPtr buildKey_;
    ExpressionPtr probeKey_;
    Schema        outSchema_;

    // Build-side row store: a single dense batch holding every build row in
    // contiguous columnar buffers. Indexed by buildRow during probe.
    std::optional<ColumnBatch> rowStore_;

    // Hash index over rowStore_. Slot stores the head of a chain.
    std::unique_ptr<HashTable> index_;
    std::vector<int32_t>       next_;     // length == rowStore_->numRows()

    bool built_ = false;
};

} // namespace quantum
