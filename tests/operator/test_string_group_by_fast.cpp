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

// These tests exercise HashAggregateOperator's StringFastPath: a single
// STRING group key that routes through the global-dict bookkeeping path
// instead of the byte-keyed hash table.

namespace {

ColumnBatch buildBatch(MemoryManager& mm, bool dict,
                       const std::vector<std::string>& names,
                       const std::vector<int64_t>&     values,
                       const std::vector<bool>&        validity = {}) {
    const size_t n = names.size();
    ColumnVector nameCol;
    if (dict) {
        DictStringBuilder b(&mm);
        for (size_t i = 0; i < n; ++i) {
            if (!validity.empty() && !validity[i]) b.appendNull();
            else                                    b.append(names[i]);
        }
        nameCol = b.build();
    } else {
        StringBuilder b(&mm);
        for (size_t i = 0; i < n; ++i) {
            if (!validity.empty() && !validity[i]) b.appendNull();
            else                                    b.append(names[i]);
        }
        nameCol = b.build();
    }
    auto valCol = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) valCol.typed<int64_t>()[i] = values[i];
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(nameCol));
    cols.push_back(std::move(valCol));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

OperatorPtr scanNameVal(MemoryManager& mm, std::vector<ColumnBatch> src) {
    return std::make_unique<ScanOperator>(
        &mm,
        Schema({{"name", DataType{TypeId::STRING}}, {"v", Int64}}),
        std::move(src));
}

std::vector<ExpressionPtr> keyOn(size_t idx) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(idx));
    return v;
}

std::map<std::optional<std::string>, int64_t> drainGroupBySum(HashAggregateOperator& op) {
    std::map<std::optional<std::string>, int64_t> got;
    op.open();
    while (auto batch = op.getNext()) {
        if (batch->numRows() == 0) continue;
        for (size_t i = 0; i < batch->numRows(); ++i) {
            std::optional<std::string> key;
            if (batch->column(0).validity().isValid(i)) {
                key = std::string(batch->column(0).stringAt(i));
            }
            got[key] = batch->column(1).typed<int64_t>()[i];
        }
    }
    op.close();
    return got;
}

} // namespace

TEST(StringGroupByFast, DictInputAcrossMultipleBatches) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(buildBatch(mm, /*dict=*/true,
        {"alice", "bob",   "alice"},   {10, 20, 30}));
    src.push_back(buildBatch(mm, /*dict=*/true,
        {"carol", "bob",   "alice"},   {40, 50, 60}));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs), {"name", "sum"});
    auto got = drainGroupBySum(op);

    EXPECT_EQ(got.size(), 3u);
    EXPECT_EQ(got[std::optional<std::string>{"alice"}], 10 + 30 + 60);
    EXPECT_EQ(got[std::optional<std::string>{"bob"}],   20 + 50);
    EXPECT_EQ(got[std::optional<std::string>{"carol"}], 40);
}

TEST(StringGroupByFast, NonDictInputStillCorrect) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(buildBatch(mm, /*dict=*/false,
        {"alice", "bob", "alice", "alice", "bob"}, {1, 2, 3, 4, 5}));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs), {"name", "sum"});
    auto got = drainGroupBySum(op);

    EXPECT_EQ(got.size(), 2u);
    EXPECT_EQ(got[std::optional<std::string>{"alice"}], 1 + 3 + 4);
    EXPECT_EQ(got[std::optional<std::string>{"bob"}],   2 + 5);
}

TEST(StringGroupByFast, MixedDictAndNonDictBatches) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(buildBatch(mm, /*dict=*/true,
        {"alice", "bob"}, {1, 2}));
    src.push_back(buildBatch(mm, /*dict=*/false,
        {"alice", "carol", "bob"}, {3, 4, 5}));
    src.push_back(buildBatch(mm, /*dict=*/true,
        {"carol", "alice"}, {6, 7}));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs), {"name", "sum"});
    auto got = drainGroupBySum(op);

    EXPECT_EQ(got.size(), 3u);
    EXPECT_EQ(got[std::optional<std::string>{"alice"}], 1 + 3 + 7);
    EXPECT_EQ(got[std::optional<std::string>{"bob"}],   2 + 5);
    EXPECT_EQ(got[std::optional<std::string>{"carol"}], 4 + 6);
}

TEST(StringGroupByFast, NullKeyIsItsOwnGroup) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(buildBatch(mm, /*dict=*/true,
        {"alice", "x", "alice", "y", "bob"},
        {10,      11,  12,      13,  14},
        {true,    false, true,  false, true}));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs), {"name", "sum"});
    auto got = drainGroupBySum(op);

    EXPECT_EQ(got.size(), 3u);
    EXPECT_EQ(got[std::optional<std::string>{"alice"}], 10 + 12);
    EXPECT_EQ(got[std::optional<std::string>{"bob"}],   14);
    EXPECT_EQ(got[std::nullopt],                        11 + 13);
}

TEST(StringGroupByFast, ManyGroupsTriggersResize) {
    MemoryManager mm(64 * 1024 * 1024);
    // Force codeStates_ to grow several times by producing many distinct keys.
    std::vector<std::string> names;
    std::vector<int64_t>     vals;
    for (int i = 0; i < 200; ++i) {
        names.push_back("k_" + std::to_string(i));
        vals.push_back(i);
    }
    std::vector<ColumnBatch> src;
    src.push_back(buildBatch(mm, /*dict=*/true, names, vals));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs), {"name", "sum"});
    auto got = drainGroupBySum(op);

    EXPECT_EQ(got.size(), 200u);
    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(got[std::optional<std::string>{"k_" + std::to_string(i)}], i);
    }
}
