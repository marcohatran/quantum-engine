#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "expression/Expression.hpp"
#include "hash/BytesHashTable.hpp"
#include "hash/KeyCodec.hpp"
#include "hash/VarBytesHashTable.hpp"
#include "operator/Operator.hpp"

namespace quantum {

enum class JoinType { Inner, LeftOuter, RightOuter, FullOuter };

// Equi-join on any number of fixed-width or variable-width key columns.
// Phases:
//   1. Build (first getNext call): drain the build side into a dense row store
//      and populate a hash table whose key is the encoded composite of all
//      build-key columns. Duplicate keys chain via next_[].
//   2. Probe (subsequent calls until probe exhausted): for each probe batch,
//      encode each row's key, walk the chain, and record matches. NULL in any
//      probe key column means no match.
//   3. Right tail (RIGHT/FULL OUTER only, one batch after probe exhaustion):
//      emit one row per build row that was never matched during probe, with
//      NULL probe columns.
//
// Internally uses BytesHashTable when every key column is fixed-width and
// VarBytesHashTable when any key column is variable-width (e.g. STRING).
//
// JoinType controls miss behaviour:
//   - Inner:      skip both probe-misses and build-misses.
//   - LeftOuter:  emit probe-miss rows with NULL build cols.
//   - RightOuter: emit build-miss rows (right tail) with NULL probe cols.
//   - FullOuter:  both.
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

    std::unique_ptr<KeyCodec>          codec_;
    std::optional<ColumnBatch>         rowStore_;
    std::unique_ptr<BytesHashTable>    indexFixed_;
    std::unique_ptr<VarBytesHashTable> indexVar_;
    std::vector<int32_t>               next_;

    // Right-tail state (used by RightOuter / FullOuter).
    std::vector<bool> buildMatched_;
    bool              built_              = false;
    bool              probeExhausted_     = false;
    bool              rightTailEmitted_   = false;
};

} // namespace quantum
