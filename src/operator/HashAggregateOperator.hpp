#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "expression/Expression.hpp"
#include "hash/BytesHashTable.hpp"
#include "hash/KeyCodec.hpp"
#include "hash/VarBytesHashTable.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/Operator.hpp"
#include "spill/SpillFile.hpp"

namespace quantum {

// GROUP BY <key columns> with one or more aggregate functions. Two-phase
// execution: drain the child during the first getNext() to populate the hash
// table, then emit one batch holding (key cols..., agg cols...) for every
// distinct group. Subsequent getNext() calls return nullopt.
//
// Three internal paths chosen at construction:
//   - Int64FastPath: single INT64 key without spill triggers. Open
//     addressing straight on the key (no KeyCodec, no byte memcmp); group
//     state lives in a slot-indexed slab the bucket rehash never moves.
//   - StringFastPath: single STRING key. Maintains a global dictionary
//     across batches; for dict-encoded inputs the per-row hash lookup is
//     replaced with a batchCode -> globalCode table lookup (huge win for
//     low-cardinality columns). For non-dict inputs we still intern strings
//     into the same global dictionary, so output ordering is consistent.
//   - Fixed (BytesHashTable): every key column is fixed-width.
//   - Var   (VarBytesHashTable): at least one key column is variable-width
//     (string) and the StringFastPath isn't applicable (multi-key composite).
class HashAggregateOperator final : public Operator {
public:
    HashAggregateOperator(MemoryManager*                    mm,
                          OperatorPtr                       child,
                          std::vector<ExpressionPtr>        groupKeys,
                          std::vector<AggregateFunctionPtr> aggs,
                          std::vector<std::string>          outputNames,
                          // Spill v1: in StringFastPath, dump partial state
                          // to disk and reset once codeStates_ holds this many
                          // groups. 0 disables spill (default).
                          size_t                            spillTriggerGroups = 0,
                          // Optional fractional budget trigger: when the
                          // MemoryManager's usedFraction() rises above this
                          // (0.0..1.0), force a spill. 0 disables.
                          double                            spillFraction = 0.0);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    enum class Path { StringFastPath, Int64FastPath, Fixed, Var };

    void        build();
    ColumnBatch emit();
    ColumnBatch emitFixed();
    ColumnBatch emitVar();
    ColumnBatch emitStringFastPath();
    ColumnBatch emitInt64FastPath();

    // Int64FastPath key -> slot lookup (open addressing; the bucket table
    // rehashes but slot indices — and therefore state-slab positions —
    // never move).
    int32_t i64FindOrInsert(int64_t key);
    void    i64Grow();

    // Returns a pointer to the per-group state slab for the global code, growing
    // the state vector as needed.
    uint8_t* fastPathStateForCode(int32_t globalCode);
    int32_t  internGlobalString(std::string_view s);

    // Serialize the current StringFastPath state to a fresh SpillFile and
    // reset the in-memory dictionaries / states. Idempotent on an empty state.
    void     spillStringFastPath();
    // Same for Fixed / Var paths: walk the table cursor, dump each (key, state)
    // record, then replace the table with a fresh one.
    void     spillFixed();
    void     spillVar();
    // After all input has been consumed, drain every spill file back into the
    // current in-memory state via the appropriate find-or-insert + agg->merge.
    void     mergeSpillFiles();
    void     mergeSpillFilesFixed();
    void     mergeSpillFilesVar();
    // Spill record (de)serialization: each aggregate's serializeState /
    // deserializeState handles its own slice, so out-of-line state
    // (MIN/MAX(string), COUNT(DISTINCT) sets) round-trips through disk.
    void     serializeSlab(const uint8_t* slab, std::vector<uint8_t>& rec) const;
    void     mergeSerializedInto(uint8_t* dst, const uint8_t* data);

    OperatorPtr                       child_;
    std::vector<ExpressionPtr>        groupKeys_;
    std::vector<AggregateFunctionPtr> aggs_;
    Schema                            outSchema_;
    std::vector<DataType>             keyTypes_;
    std::unique_ptr<KeyCodec>         codec_;        // null when StringFastPath
    Path                              path_;

    std::vector<size_t> aggOffsets_;
    size_t              stateBytes_ = 0;
    std::vector<uint8_t> mergeScratch_;  // reused slab for mergeSerializedInto

    // Path::Fixed
    std::unique_ptr<BytesHashTable>    tableFixed_;
    // Path::Var
    std::unique_ptr<VarBytesHashTable> tableVar_;

    // Path::Int64FastPath state. Slot s owns codeStates_[s * stateBytes_]
    // (shared slab with StringFastPath; the two paths are exclusive).
    std::vector<int64_t> i64Keys_;        // slot -> key, insertion order
    std::vector<int64_t> i64BucketKeys_;  // open-addressing buckets
    std::vector<int32_t> i64BucketSlot_;  // -1 = empty
    size_t               i64Mask_ = 0;
    std::vector<int32_t> rowSlots_;       // per-batch scratch

    // Path::StringFastPath state
    std::unordered_map<std::string, int32_t> globalDict_;
    std::vector<std::string>                 globalDictReverse_;
    std::vector<uint8_t>                     codeStates_;     // dictN * stateBytes_
    std::vector<uint8_t>                     nullGroupState_; // stateBytes_
    bool                                     nullGroupInit_ = false;

    // Spill v1 state (StringFastPath only).
    size_t                  spillTriggerGroups_ = 0;
    double                  spillFraction_      = 0.0;
    std::vector<SpillFile>  spillFiles_;

    bool built_   = false;
    bool emitted_ = false;
};

} // namespace quantum
