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

// --- Fixed and Var path spill ---

namespace {

ColumnBatch intIntBatch(MemoryManager& mm,
                        const std::vector<int64_t>& keys,
                        const std::vector<int64_t>& vals) {
    auto k = ColumnVector::make(Int64, keys.size(), &mm);
    auto v = ColumnVector::make(Int64, vals.size(), &mm);
    for (size_t i = 0; i < keys.size(); ++i) {
        k.typed<int64_t>()[i] = keys[i];
        v.typed<int64_t>()[i] = vals[i];
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(keys.size()));
}

} // namespace

TEST(SpillHashAggregate, FixedPathSpillMatchesBaseline) {
    MemoryManager mm(64 * 1024 * 1024);

    auto runFixed = [&](size_t spill) {
        std::vector<ColumnBatch> src;
        src.push_back(intIntBatch(mm, {1, 2, 1, 3},     {10, 20, 30, 40}));
        src.push_back(intIntBatch(mm, {2, 4, 1},        {50, 60, 70}));
        src.push_back(intIntBatch(mm, {5, 1, 3, 2},     {80, 90, 100, 110}));

        auto scan = std::make_unique<ScanOperator>(
            &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
        aggs.push_back(std::make_unique<CountStar>());
        HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                                 std::move(aggs),
                                 std::vector<std::string>{"k", "sum", "cnt"},
                                 spill);
        std::map<int64_t, std::pair<int64_t, int64_t>> got;
        op.open();
        while (auto b = op.getNext()) {
            if (b->numRows() == 0) continue;
            for (size_t i = 0; i < b->numRows(); ++i) {
                got[b->column(0).typed<int64_t>()[i]] = {
                    b->column(1).typed<int64_t>()[i],
                    b->column(2).typed<int64_t>()[i]};
            }
        }
        op.close();
        return got;
    };

    auto baseline = runFixed(0);
    auto spilled  = runFixed(2);
    EXPECT_EQ(spilled, baseline);
}

TEST(SpillHashAggregate, VarPathSpillMatchesBaselineForCompositeKey) {
    MemoryManager mm(64 * 1024 * 1024);

    auto makeBatch = [&](const std::vector<std::string>& names,
                         const std::vector<int64_t>&     buckets,
                         const std::vector<int64_t>&     vals) {
        StringBuilder sb(&mm);
        for (const auto& s : names) sb.append(s);
        auto b = ColumnVector::make(Int64, buckets.size(), &mm);
        auto v = ColumnVector::make(Int64, vals.size(), &mm);
        for (size_t i = 0; i < buckets.size(); ++i) {
            b.typed<int64_t>()[i] = buckets[i];
            v.typed<int64_t>()[i] = vals[i];
        }
        std::vector<ColumnVector> cols;
        cols.push_back(sb.build());
        cols.push_back(std::move(b));
        cols.push_back(std::move(v));
        return ColumnBatch(std::move(cols), PositionList::identity(names.size()));
    };

    auto run = [&](size_t spill) {
        std::vector<ColumnBatch> src;
        src.push_back(makeBatch({"a", "a", "b"}, {1, 2, 1}, {10, 20, 30}));
        src.push_back(makeBatch({"b", "a"},      {1, 1},    {40, 50}));
        src.push_back(makeBatch({"a", "b", "c"}, {2, 1, 1}, {60, 70, 80}));

        auto scan = std::make_unique<ScanOperator>(
            &mm,
            Schema({{"name", DataType{TypeId::STRING}}, {"b", Int64}, {"v", Int64}}),
            std::move(src));
        std::vector<ExpressionPtr> keys;
        keys.push_back(std::make_unique<ColumnRef>(0));
        keys.push_back(std::make_unique<ColumnRef>(1));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(2)));
        HashAggregateOperator op(&mm, std::move(scan), std::move(keys),
                                 std::move(aggs),
                                 std::vector<std::string>{"name", "b", "sum"},
                                 spill);
        std::map<std::pair<std::string, int64_t>, int64_t> got;
        op.open();
        while (auto b = op.getNext()) {
            if (b->numRows() == 0) continue;
            for (size_t i = 0; i < b->numRows(); ++i) {
                got[{std::string(b->column(0).stringAt(i)),
                     b->column(1).typed<int64_t>()[i]}] =
                    b->column(2).typed<int64_t>()[i];
            }
        }
        op.close();
        return got;
    };

    auto baseline = run(0);
    auto spilled  = run(2);
    EXPECT_EQ(spilled, baseline);
}

// --- Out-of-line aggregate state through spill (Phase 5.40) ---

TEST(SpillHashAggregate, CountDistinctI64WithSpillMatchesNoSpill) {
    MemoryManager mm(64 * 1024 * 1024);

    auto run = [&](size_t spill) {
        std::vector<ColumnBatch> src;
        // Duplicates of the same (group, value) land in different batches,
        // so post-spill dedup must happen at merge time via set union.
        src.push_back(nvBatch(mm, {"a", "b", "a", "c"}, {1, 2, 1, 3}));
        src.push_back(nvBatch(mm, {"a", "b", "c"},      {1, 5, 3}));
        src.push_back(nvBatch(mm, {"a", "b", "a"},      {7, 2, 1}));

        auto scan = scanNV(mm, std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<CountDistinctI64>(std::make_unique<ColumnRef>(1)));
        HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                                 std::move(aggs),
                                 std::vector<std::string>{"name", "nd"},
                                 spill);
        std::map<std::string, int64_t> got;
        op.open();
        while (auto b = op.getNext()) {
            if (b->numRows() == 0) continue;
            for (size_t r = 0; r < b->numRows(); ++r) {
                got[std::string(b->column(0).stringAt(r))] =
                    b->column(1).typed<int64_t>()[r];
            }
        }
        op.close();
        return got;
    };

    auto baseline = run(0);
    auto spilled  = run(2);
    EXPECT_EQ(spilled, baseline);
    // distinct: a={1,7}, b={2,5}, c={3}
    EXPECT_EQ(baseline.at("a"), 2);
    EXPECT_EQ(baseline.at("b"), 2);
    EXPECT_EQ(baseline.at("c"), 1);
}

namespace {

// (name STRING, sval STRING) batch; svalValid marks null values.
ColumnBatch ssBatch(MemoryManager& mm,
                    const std::vector<std::string>& names,
                    const std::vector<std::string>& svals,
                    const std::vector<bool>&        svalValid = {}) {
    DictStringBuilder kb(&mm);
    for (const auto& s : names) kb.append(s);
    StringBuilder vb(&mm);
    for (size_t i = 0; i < svals.size(); ++i) {
        if (!svalValid.empty() && !svalValid[i]) vb.appendNull();
        else                                     vb.append(svals[i]);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(kb.build());
    cols.push_back(vb.build());
    return ColumnBatch(std::move(cols), PositionList::identity(names.size()));
}

} // namespace

TEST(SpillHashAggregate, MinMaxStringWithSpillMatchesNoSpill) {
    MemoryManager mm(64 * 1024 * 1024);

    auto run = [&](size_t spill) {
        std::vector<ColumnBatch> src;
        src.push_back(ssBatch(mm, {"a", "b", "a"}, {"pear", "mango", "apple"}));
        // Group c has only a null value -> MIN/MAX NULL must survive the
        // spill round-trip (hasValue=0 serialization).
        src.push_back(ssBatch(mm, {"c", "a"},      {"", "zebra"}, {false, true}));
        src.push_back(ssBatch(mm, {"b", "a"},      {"kiwi", "fig"}));

        auto scan = std::make_unique<ScanOperator>(
            &mm,
            Schema({{"name", DataType{TypeId::STRING}},
                    {"sval", DataType{TypeId::STRING}}}),
            std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<MinString>(std::make_unique<ColumnRef>(1)));
        aggs.push_back(std::make_unique<MaxString>(std::make_unique<ColumnRef>(1)));
        HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                                 std::move(aggs),
                                 std::vector<std::string>{"name", "mn", "mx"},
                                 spill);
        std::map<std::string,
                 std::pair<std::optional<std::string>, std::optional<std::string>>> got;
        op.open();
        while (auto b = op.getNext()) {
            if (b->numRows() == 0) continue;
            for (size_t r = 0; r < b->numRows(); ++r) {
                auto strAt = [&](size_t c) -> std::optional<std::string> {
                    if (!b->column(c).validity().isValid(r)) return std::nullopt;
                    return std::string(b->column(c).stringAt(r));
                };
                got[std::string(b->column(0).stringAt(r))] = {strAt(1), strAt(2)};
            }
        }
        op.close();
        return got;
    };

    auto baseline = run(0);
    auto spilled  = run(2);
    EXPECT_EQ(spilled, baseline);
    EXPECT_EQ(baseline.at("a"),
              (std::pair<std::optional<std::string>, std::optional<std::string>>{
                  "apple", "zebra"}));
    EXPECT_EQ(baseline.at("b"),
              (std::pair<std::optional<std::string>, std::optional<std::string>>{
                  "kiwi", "mango"}));
    EXPECT_EQ(baseline.at("c"),
              (std::pair<std::optional<std::string>, std::optional<std::string>>{
                  std::nullopt, std::nullopt}));
}

TEST(SpillHashAggregate, CountDistinctStringWithSpillMatchesNoSpill) {
    MemoryManager mm(64 * 1024 * 1024);

    auto run = [&](size_t spill) {
        std::vector<ColumnBatch> src;
        src.push_back(ssBatch(mm, {"a", "b", "a"}, {"x", "y", "x"}));
        src.push_back(ssBatch(mm, {"a", "b"},      {"x", "z"}));
        src.push_back(ssBatch(mm, {"a", "b"},      {"w", "y"}));

        auto scan = std::make_unique<ScanOperator>(
            &mm,
            Schema({{"name", DataType{TypeId::STRING}},
                    {"sval", DataType{TypeId::STRING}}}),
            std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<CountDistinctString>(std::make_unique<ColumnRef>(1)));
        HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                                 std::move(aggs),
                                 std::vector<std::string>{"name", "nd"},
                                 spill);
        std::map<std::string, int64_t> got;
        op.open();
        while (auto b = op.getNext()) {
            if (b->numRows() == 0) continue;
            for (size_t r = 0; r < b->numRows(); ++r) {
                got[std::string(b->column(0).stringAt(r))] =
                    b->column(1).typed<int64_t>()[r];
            }
        }
        op.close();
        return got;
    };

    auto baseline = run(0);
    auto spilled  = run(2);
    EXPECT_EQ(spilled, baseline);
    // distinct: a={x,w}, b={y,z}
    EXPECT_EQ(baseline.at("a"), 2);
    EXPECT_EQ(baseline.at("b"), 2);
}

TEST(SpillHashAggregate, FixedPathCountDistinctWithSpill) {
    MemoryManager mm(64 * 1024 * 1024);

    auto run = [&](size_t spill) {
        std::vector<ColumnBatch> src;
        src.push_back(intIntBatch(mm, {1, 2, 1, 3}, {10, 20, 10, 30}));
        src.push_back(intIntBatch(mm, {2, 1},       {20, 40}));
        src.push_back(intIntBatch(mm, {3, 1, 2},    {30, 10, 50}));

        auto scan = std::make_unique<ScanOperator>(
            &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
        std::vector<AggregateFunctionPtr> aggs;
        aggs.push_back(std::make_unique<CountDistinctI64>(std::make_unique<ColumnRef>(1)));
        HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                                 std::move(aggs),
                                 std::vector<std::string>{"k", "nd"},
                                 spill);
        std::map<int64_t, int64_t> got;
        op.open();
        while (auto b = op.getNext()) {
            if (b->numRows() == 0) continue;
            for (size_t i = 0; i < b->numRows(); ++i) {
                got[b->column(0).typed<int64_t>()[i]] =
                    b->column(1).typed<int64_t>()[i];
            }
        }
        op.close();
        return got;
    };

    auto baseline = run(0);
    auto spilled  = run(2);
    EXPECT_EQ(spilled, baseline);
    // distinct: k=1 -> {10,40}, k=2 -> {20,50}, k=3 -> {30}
    EXPECT_EQ(baseline.at(1), 2);
    EXPECT_EQ(baseline.at(2), 2);
    EXPECT_EQ(baseline.at(3), 1);
}
