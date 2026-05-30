#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/HashJoinOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch nameValueBatch(MemoryManager& mm,
                           const std::vector<std::string>& names,
                           const std::vector<int64_t>&     values,
                           const std::vector<bool>&        nameValidity = {}) {
    const size_t n = names.size();
    StringBuilder sb(&mm);
    for (size_t i = 0; i < n; ++i) {
        if (!nameValidity.empty() && !nameValidity[i]) sb.appendNull();
        else                                            sb.append(names[i]);
    }
    auto nameCol = sb.build();

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

} // namespace

TEST(HashAggregate, GroupByString) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(nameValueBatch(mm,
        {"alice", "bob", "alice", "carol", "bob"},
        {10,      20,    30,      40,      50}));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<CountStar>());

    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs),
                             {"name", "sum_v", "cnt"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);

    std::map<std::string, std::pair<int64_t, int64_t>> got;
    for (size_t i = 0; i < batch->numRows(); ++i) {
        got[std::string(batch->column(0).stringAt(i))] = {
            batch->column(1).typed<int64_t>()[i],
            batch->column(2).typed<int64_t>()[i],
        };
    }
    EXPECT_EQ(got["alice"], (std::pair<int64_t,int64_t>{40, 2}));
    EXPECT_EQ(got["bob"],   (std::pair<int64_t,int64_t>{70, 2}));
    EXPECT_EQ(got["carol"], (std::pair<int64_t,int64_t>{40, 1}));
}

TEST(HashAggregate, GroupByStringWithNullKeys) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(nameValueBatch(mm,
        {"alice", "x", "alice", "x"},
        {10,      11,  12,      13},
        {true,    false, true,  false}));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));

    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs),
                             {"name", "sum_v"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 2u);  // {alice}, {NULL}

    std::map<std::optional<std::string>, int64_t> got;
    for (size_t i = 0; i < batch->numRows(); ++i) {
        std::optional<std::string> key;
        if (batch->column(0).validity().isValid(i)) {
            key = std::string(batch->column(0).stringAt(i));
        }
        got[key] = batch->column(1).typed<int64_t>()[i];
    }
    EXPECT_EQ(got[std::optional<std::string>{"alice"}], 22);
    EXPECT_EQ(got[std::nullopt],                       24);
}

TEST(HashAggregate, GroupByCompositeStringPlusInt) {
    MemoryManager mm(64 * 1024 * 1024);

    auto build = [&](const std::vector<std::string>& names,
                     const std::vector<int64_t>&     buckets,
                     const std::vector<int64_t>&     vals) {
        const size_t n = names.size();
        StringBuilder sb(&mm);
        for (const auto& s : names) sb.append(s);
        auto nameCol = sb.build();
        auto bucketCol = ColumnVector::make(Int64, n, &mm);
        auto valCol    = ColumnVector::make(Int64, n, &mm);
        for (size_t i = 0; i < n; ++i) {
            bucketCol.typed<int64_t>()[i] = buckets[i];
            valCol.typed<int64_t>()[i]    = vals[i];
        }
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(nameCol));
        cols.push_back(std::move(bucketCol));
        cols.push_back(std::move(valCol));
        return ColumnBatch(std::move(cols), PositionList::identity(n));
    };

    std::vector<ColumnBatch> src;
    src.push_back(build(
        {"a", "a", "b", "a", "b"},
        {1,   2,   1,   1,   1},
        {10,  20,  30,  40,  50}));
    auto scan = std::make_unique<ScanOperator>(
        &mm,
        Schema({{"name", DataType{TypeId::STRING}}, {"bucket", Int64}, {"v", Int64}}),
        std::move(src));

    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    keys.push_back(std::make_unique<ColumnRef>(1));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(2)));

    HashAggregateOperator op(&mm, std::move(scan), std::move(keys),
                             std::move(aggs),
                             {"name", "bucket", "sum_v"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);  // (a,1), (a,2), (b,1)

    std::map<std::pair<std::string,int64_t>, int64_t> got;
    for (size_t i = 0; i < batch->numRows(); ++i) {
        got[{std::string(batch->column(0).stringAt(i)),
             batch->column(1).typed<int64_t>()[i]}] = batch->column(2).typed<int64_t>()[i];
    }
    EXPECT_EQ((got[{"a", 1}]), 10 + 40);
    EXPECT_EQ((got[{"a", 2}]), 20);
    EXPECT_EQ((got[{"b", 1}]), 30 + 50);
}

TEST(HashJoin, InnerJoinOnStringKey) {
    MemoryManager mm(64 * 1024 * 1024);

    std::vector<ColumnBatch> build;
    build.push_back(nameValueBatch(mm, {"alice", "bob", "carol"}, {100, 200, 300}));

    std::vector<ColumnBatch> probe;
    probe.push_back(nameValueBatch(mm, {"bob", "dan", "alice"}, {2, 4, 1}));

    auto bScan = scanNameVal(mm, std::move(build));
    auto pScan = scanNameVal(mm, std::move(probe));

    HashJoinOperator op(&mm, std::move(bScan), std::move(pScan),
                        keyOn(0), keyOn(0),
                        {"bname", "bv", "pname", "pv"});

    std::map<std::string, std::pair<int64_t, int64_t>> got;
    op.open();
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            got[std::string(b->column(0).stringAt(i))] = {
                b->column(1).typed<int64_t>()[i],
                b->column(3).typed<int64_t>()[i],
            };
        }
    }
    op.close();

    EXPECT_EQ(got.size(), 2u);
    EXPECT_EQ(got["alice"], (std::pair<int64_t,int64_t>{100, 1}));
    EXPECT_EQ(got["bob"],   (std::pair<int64_t,int64_t>{200, 2}));
}

TEST(HashJoin, LeftOuterOnStringKeyEmitsNullBuildForMisses) {
    MemoryManager mm(64 * 1024 * 1024);

    std::vector<ColumnBatch> build;
    build.push_back(nameValueBatch(mm, {"alice"}, {100}));

    std::vector<ColumnBatch> probe;
    probe.push_back(nameValueBatch(mm, {"alice", "bob"}, {1, 2}));

    auto bScan = scanNameVal(mm, std::move(build));
    auto pScan = scanNameVal(mm, std::move(probe));

    HashJoinOperator op(&mm, std::move(bScan), std::move(pScan),
                        keyOn(0), keyOn(0),
                        {"bname", "bv", "pname", "pv"},
                        JoinType::LeftOuter);

    std::map<std::string, std::pair<std::optional<std::string>, int64_t>> got;
    op.open();
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            std::optional<std::string> bname;
            if (b->column(0).validity().isValid(i)) {
                bname = std::string(b->column(0).stringAt(i));
            }
            got[std::string(b->column(2).stringAt(i))] = {
                bname, b->column(3).typed<int64_t>()[i],
            };
        }
    }
    op.close();

    EXPECT_EQ(got.size(), 2u);
    EXPECT_EQ(got["alice"], (std::pair<std::optional<std::string>, int64_t>{
        std::optional<std::string>{"alice"}, 1}));
    EXPECT_EQ(got["bob"], (std::pair<std::optional<std::string>, int64_t>{
        std::nullopt, 2}));
}

TEST(HashJoin, NullStringKeysNeverMatch) {
    MemoryManager mm(64 * 1024 * 1024);

    std::vector<ColumnBatch> build;
    build.push_back(nameValueBatch(mm,
        {"alice", "x", "bob"}, {100, 999, 200}, {true, false, true}));

    std::vector<ColumnBatch> probe;
    probe.push_back(nameValueBatch(mm,
        {"alice", "x", "bob"}, {1, 88, 2}, {true, false, true}));

    auto bScan = scanNameVal(mm, std::move(build));
    auto pScan = scanNameVal(mm, std::move(probe));

    HashJoinOperator op(&mm, std::move(bScan), std::move(pScan),
                        keyOn(0), keyOn(0),
                        {"bname", "bv", "pname", "pv"});

    std::map<std::string, std::pair<int64_t,int64_t>> got;
    op.open();
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            got[std::string(b->column(0).stringAt(i))] = {
                b->column(1).typed<int64_t>()[i],
                b->column(3).typed<int64_t>()[i],
            };
        }
    }
    op.close();

    EXPECT_EQ(got.size(), 2u);
    EXPECT_EQ(got["alice"], (std::pair<int64_t,int64_t>{100, 1}));
    EXPECT_EQ(got["bob"],   (std::pair<int64_t,int64_t>{200, 2}));
}
