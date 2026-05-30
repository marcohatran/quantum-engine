#pragma once

#include <memory>
#include <string>
#include <vector>

#include "expression/Expression.hpp"
#include "hash/BytesHashTable.hpp"
#include "hash/KeyCodec.hpp"
#include "operator/Operator.hpp"

namespace quantum {

enum class JoinType { Inner, LeftOuter };

// Equi-join on any number of fixed-width key columns. Two-phase execution:
//   1. Build (first getNext call): drain the build side into a dense row store
//      and populate a BytesHashTable whose key is the encoded composite of
//      all build-key columns. Duplicate keys form a chain via next_[].
//   2. Probe (subsequent calls): for each probe batch, encode each row's key
//      and walk the chain. NULL in any probe key column means no match
//      (LEFT OUTER still emits a row with NULL build columns).
//
// JoinType controls miss behaviour:
//   - Inner:     skip the row.
//   - LeftOuter: emit one output row with probe values and NULL build cols.
//
// Phase 2 scope: Inner + LeftOuter, fixed-width keys only, all input columns
// passed through. RIGHT/FULL OUTER, string keys, and spill come later.
class HashJoinOperator final : public Operator {
public:
    HashJoinOperator(MemoryManager*               mm,
                     OperatorPtr                  buildSide,
                     OperatorPtr                  probeSide,
                     std::vector<ExpressionPtr>   buildKeys,
                     std::vector<ExpressionPtr>   probeKeys,
                     std::vector<std::string>     outputNames,
                     JoinType                     joinType = JoinType::Inner);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    void                       build();
    std::optional<ColumnBatch> probeOne();

    OperatorPtr                build_;
    OperatorPtr                probe_;
    std::vector<ExpressionPtr> buildKeys_;
    std::vector<ExpressionPtr> probeKeys_;
    JoinType                   joinType_;
    Schema                     outSchema_;

    std::unique_ptr<KeyCodec>       codec_;
    std::optional<ColumnBatch>      rowStore_;
    std::unique_ptr<BytesHashTable> index_;
    std::vector<int32_t>            next_;

    bool built_ = false;
};

} // namespace quantum
