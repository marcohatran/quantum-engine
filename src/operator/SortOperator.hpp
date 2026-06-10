#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "expression/Expression.hpp"
#include "operator/Operator.hpp"
#include "spill/SpillFile.hpp"

namespace quantum {

// Sort by N key expressions. In-memory mode (default): materializes all
// input rows, sorts a row-index permutation by the key columns in declared
// order, then emits one densely packed batch in sorted order.
//
// External mode: when a spill trigger is configured (row-count and/or
// memory-fraction, mirroring HashAggregateOperator), buffered input is
// sorted and written to disk as a sorted run whenever the trigger fires.
// Runs are SpillFile records of [serialized key values][serialized row];
// after input is drained the runs are k-way merged by their head keys and
// the result streams out in fixed-size batches.
//
// `ascending` and `nullsFirst` are parallel arrays to `keys`. Each key
// expression is evaluated once per run and the resulting column is used by
// the comparator. Supported key types: INT32 / INT64 / FLOAT64 / BOOL /
// STRING. Mixed-type keys are supported (each key has its own type).
class SortOperator final : public Operator {
public:
    SortOperator(MemoryManager*               mm,
                 OperatorPtr                  child,
                 std::vector<ExpressionPtr>   keys,
                 std::vector<bool>            ascending,
                 std::vector<bool>            nullsFirst,
                 size_t                       spillTriggerRows = 0,
                 double                       spillFraction    = 0.0);

    void                       open()  override;
    void                       close() override;
    const Schema&              outputSchema() const override { return child_->outputSchema(); }
    std::optional<ColumnBatch> getNext() override;

private:
    // One deserialized sort-key value at a merge head. Fixed-width values
    // widen into i64/f64; the type comes from keyTypes_.
    struct KeyValue {
        bool        valid = false;
        int64_t     i64   = 0;
        double      f64   = 0.0;
        std::string str;
    };

    void build();      // drain child; spill sorted runs as triggers fire
    void spillRun();   // sort buffer_ and write one run to disk

    ColumnBatch          densifyBuffer();
    std::vector<int32_t> sortedPermutation(const ColumnBatch&         dense,
                                           std::vector<ColumnVector>& keyColsOut);
    ColumnBatch          permute(const ColumnBatch&          dense,
                                 const std::vector<int32_t>& perm);

    void advanceHead(size_t runIdx);
    int  compareHeads(const std::vector<KeyValue>& a,
                      const std::vector<KeyValue>& b) const;
    std::optional<ColumnBatch> nextMergedBatch();

    OperatorPtr                child_;
    std::vector<ExpressionPtr> keys_;
    std::vector<bool>          ascending_;
    std::vector<bool>          nullsFirst_;
    size_t                     spillTriggerRows_;
    double                     spillFraction_;

    std::vector<ColumnBatch> buffer_;
    size_t                   bufferedRows_ = 0;
    std::vector<SpillFile>   runs_;
    std::vector<DataType>    keyTypes_;  // captured on the first spilled run

    // Merge state — parallel arrays per run; an empty head = exhausted run.
    std::vector<SpillFile::Reader>     readers_;
    std::vector<std::vector<KeyValue>> heads_;

    bool        built_   = false;
    bool        merging_ = false;
    bool        emitted_ = false;
    ColumnBatch sorted_;
};

} // namespace quantum
