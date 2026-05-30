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
                          size_t                            spillTriggerGroups = 0);

    void                       open() override;
    std::optional<ColumnBatch> getNext() override;
    void                       close() override;
    const Schema&              outputSchema() const override { return outSchema_; }

private:
    enum class Path { StringFastPath, Fixed, Var };

    void        build();
    ColumnBatch emit();
    ColumnBatch emitFixed();
    ColumnBatch emitVar();
    ColumnBatch emitStringFastPath();

    // Returns a pointer to the per-group state slab for the global code, growing
    // the state vector as needed.
    uint8_t* fastPathStateForCode(int32_t globalCode);
    int32_t  internGlobalString(std::string_view s);

    // Serialize the current StringFastPath state to a fresh SpillFile and
    // reset the in-memory dictionaries / states. Idempotent on an empty state.
    void     spillStringFastPath();
    // After all input has been consumed, drain every spill file back into the
    // current in-memory state via internGlobalString + agg->merge.
    void     mergeSpillFiles();

    OperatorPtr                       child_;
    std::vector<ExpressionPtr>        groupKeys_;
    std::vector<AggregateFunctionPtr> aggs_;
    Schema                            outSchema_;
    std::vector<DataType>             keyTypes_;
    std::unique_ptr<KeyCodec>         codec_;        // null when StringFastPath
    Path                              path_;

    std::vector<size_t> aggOffsets_;
    size_t              stateBytes_ = 0;

    // Path::Fixed
    std::unique_ptr<BytesHashTable>    tableFixed_;
    // Path::Var
    std::unique_ptr<VarBytesHashTable> tableVar_;

    // Path::StringFastPath state
    std::unordered_map<std::string, int32_t> globalDict_;
    std::vector<std::string>                 globalDictReverse_;
    std::vector<uint8_t>                     codeStates_;     // dictN * stateBytes_
    std::vector<uint8_t>                     nullGroupState_; // stateBytes_
    bool                                     nullGroupInit_ = false;

    // Spill v1 state (StringFastPath only).
    size_t                  spillTriggerGroups_ = 0;
    std::vector<SpillFile>  spillFiles_;

    bool built_   = false;
    bool emitted_ = false;
};

} // namespace quantum
