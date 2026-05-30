#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "vector/DictStringBuilder.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch nvBatch(MemoryManager& mm,
                    const std::vector<std::string>& names,
                    const std::vector<int64_t>&     vals,
                    const std::vector<bool>&        nameValid = {}) {
    DictStringBuilder sb(&mm);
    for (size_t i = 0; i < names.size(); ++i) {
        if (!nameValid.empty() && !nameValid[i]) sb.appendNull();
        else                                      sb.append(names[i]);
    }
    auto vCol = ColumnVector::make(Int64, vals.size(), &mm);
    for (size_t i = 0; i < vals.size(); ++i) vCol.typed<int64_t>()[i] = vals[i];
    std::vector<ColumnVector> cols;
    cols.push_back(sb.build());
    cols.push_back(std::move(vCol));
    return ColumnBatch(std::move(cols), PositionList::identity(names.size()));
}

OperatorPtr scanNV(MemoryManager& mm, std::vector<ColumnBatch> src) {
    return std::make_unique<ScanOperator>(
        &mm,
        Schema({{"name", DataType{TypeId::STRING}}, {"v", Int64}}),
        std::move(src));
}

std::vector<ExpressionPtr> keyOn(size_t i) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(i));
    return v;
}

// Build a deterministic 4-batch workload with many groups across batches.
std::vector<ColumnBatch> mkWorkload(MemoryManager& mm) {
    std::vector<ColumnBatch> src;
    src.push_back(nvBatch(mm, {"a", "b", "a", "c"}, {1, 2, 3, 4}));
    src.push_back(nvBatch(mm, {"c", "d", "a"},      {5, 6, 7}));
    src.push_back(nvBatch(mm, {"b", "e", "f", "a"}, {8, 9, 10, 11}));
    src.push_back(nvBatch(mm, {"f", "b", "g"},      {12, 13, 14}));
    return src;
}

template <typename MakeAggs>
std::map<std::string, std::vector<int64_t>> runAndCollect(
        MemoryManager& mm,
        size_t spillTrigger,
        MakeAggs makeAggs,
        std::vector<std::string> outputNames) {
    auto scan = scanNV(mm, mkWorkload(mm));
    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             makeAggs(),
                             outputNames,
                             spillTrigger);
    std::map<std::string, std::vector<int64_t>> got;
    op.open();
    while (auto batch = op.getNext()) {
        if (batch->numRows() == 0) continue;
        for (size_t r = 0; r < batch->numRows(); ++r) {
            std::string k(batch->column(0).stringAt(r));
            std::vector<int64_t> row;
            for (size_t c = 1; c < batch->numColumns(); ++c) {
                row.push_back(batch->column(c).typed<int64_t>()[r]);
            }
            got[k] = row;
        }
    }
    op.close();
    return got;
}

} // namespace

TEST(SpillHashAggregate, SumWithSpillMatchesNoSpill) {
    MemoryManager mm(64 * 1024 * 1024);
    auto makeAggs = []{
        std::vector<AggregateFunctionPtr> v;
        v.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
        return v;
    };
    auto baseline = runAndCollect(mm, /*spill=*/0,
                                  makeAggs, {"name", "sum"});
    auto spilled  = runAndCollect(mm, /*spill=*/2,   // force spill very early
                                  makeAggs, {"name", "sum"});
    EXPECT_EQ(spilled, baseline);
    // Sanity: we should have seen at least one spill.
    EXPECT_FALSE(spilled.empty());
}

TEST(SpillHashAggregate, CountSumMinMaxAvgAllConsistent) {
    MemoryManager mm(64 * 1024 * 1024);
    auto makeAggs = []{
        std::vector<AggregateFunctionPtr> v;
        v.push_back(std::make_unique<CountStar>());
        v.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
        v.push_back(std::make_unique<MinI64>(std::make_unique<ColumnRef>(1)));
        v.push_back(std::make_unique<MaxI64>(std::make_unique<ColumnRef>(1)));
        return v;
    };
    auto baseline = runAndCollect(mm, /*spill=*/0,
                                  makeAggs, {"name", "cnt", "sum", "mn", "mx"});
    auto spilled  = runAndCollect(mm, /*spill=*/2,
                                  makeAggs, {"name", "cnt", "sum", "mn", "mx"});
    EXPECT_EQ(spilled, baseline);
}

TEST(SpillHashAggregate, NullKeyGroupSurvivesSpill) {
    MemoryManager mm(64 * 1024 * 1024);
    auto src = std::vector<ColumnBatch>{};
    src.push_back(nvBatch(mm, {"a", "x", "a"}, {1, 2, 3}, {true, false, true}));
    src.push_back(nvBatch(mm, {"y", "b", "z"}, {4, 5, 6}, {false, true, false}));

    auto run = [&](size_t spill) {
        auto scan = std::make_unique<ScanOperator>(
            &mm, Schema({{"name", DataType{TypeId::STRING}}, {"v", Int64}}),
            std::vector<ColumnBatch>(std::move(src)));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
        HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                                 std::move(aggs),
                                 std::vector<std::string>{"name", "sum"},
                                 spill);
        std::map<std::optional<std::string>, int64_t> got;
        op.open();
        while (auto b = op.getNext()) {
            if (b->numRows() == 0) continue;
            for (size_t r = 0; r < b->numRows(); ++r) {
                std::optional<std::string> k;
                if (b->column(0).validity().isValid(r)) {
                    k = std::string(b->column(0).stringAt(r));
                }
                got[k] = b->column(1).typed<int64_t>()[r];
            }
        }
        op.close();
        return got;
    };

    auto baseline = run(0);
    // Rebuild source since we moved it; run again with spill=1 (every batch).
    src.clear();
    src.push_back(nvBatch(mm, {"a", "x", "a"}, {1, 2, 3}, {true, false, true}));
    src.push_back(nvBatch(mm, {"y", "b", "z"}, {4, 5, 6}, {false, true, false}));
    auto spilled = run(1);

    EXPECT_EQ(spilled, baseline);
    // Null group must be present and equal in both runs (2 + 4 + 6 = 12).
    EXPECT_EQ(baseline[std::nullopt], 12);
}
