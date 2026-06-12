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
#include "operator/ScanOperator.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

std::vector<ColumnBatch> mkBatches(MemoryManager& mm,
                                   const std::vector<std::vector<int64_t>>& keys,
                                   const std::vector<std::vector<int64_t>>& vals,
                                   const std::vector<std::vector<bool>>&    keyValidity = {}) {
    std::vector<ColumnBatch> out;
    for (size_t b = 0; b < keys.size(); ++b) {
        const size_t n = keys[b].size();
        auto k = ColumnVector::make(Int64, n, &mm);
        auto v = ColumnVector::make(Int64, n, &mm);
        for (size_t i = 0; i < n; ++i) {
            k.typed<int64_t>()[i] = keys[b][i];
            v.typed<int64_t>()[i] = vals[b][i];
        }
        if (!keyValidity.empty()) {
            for (size_t i = 0; i < n; ++i) k.validity().setValid(i, keyValidity[b][i]);
        }
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(k));
        cols.push_back(std::move(v));
        out.push_back(ColumnBatch(std::move(cols), PositionList::identity(n)));
    }
    return out;
}

OperatorPtr scan(MemoryManager& mm, std::vector<ColumnBatch> src) {
    return std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
}

std::vector<ExpressionPtr> singleKey(size_t idx) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(idx));
    return v;
}

} // namespace

TEST(HashAggregate, EmptyInputProducesEmptyOutput) {
    MemoryManager mm(64 * 1024 * 1024);
    auto child = scan(mm, {});
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(child), singleKey(0),
                             std::move(aggs), {"k", "sum"});
    op.open();
    auto batch = op.getNext();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->numRows(), 0u);
    EXPECT_FALSE(op.getNext().has_value());
    op.close();
}

TEST(HashAggregate, SingleGroupSum) {
    MemoryManager mm(64 * 1024 * 1024);
    auto child = scan(mm, mkBatches(mm, {{7, 7, 7, 7}}, {{1, 2, 3, 4}}));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(child), singleKey(0),
                             std::move(aggs), {"k", "sum"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 1u);
    EXPECT_EQ(batch->column(0).typed<int64_t>()[0], 7);
    EXPECT_EQ(batch->column(1).typed<int64_t>()[0], 10);
}

TEST(HashAggregate, MultipleGroupsAcrossBatches) {
    MemoryManager mm(64 * 1024 * 1024);
    auto child = scan(mm, mkBatches(mm,
                                    {{1, 2, 1, 3}, {2, 3, 1}},
                                    {{10, 20, 30, 40}, {50, 60, 70}}));

    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<CountStar>());

    HashAggregateOperator op(&mm, std::move(child), singleKey(0),
                             std::move(aggs), {"k", "sum_v", "cnt"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);

    std::map<int64_t, std::pair<int64_t, int64_t>> got;
    const auto* k   = batch->column(0).typed<int64_t>();
    const auto* sum = batch->column(1).typed<int64_t>();
    const auto* cnt = batch->column(2).typed<int64_t>();
    for (size_t i = 0; i < 3; ++i) got[k[i]] = {sum[i], cnt[i]};

    EXPECT_EQ(got[1], (std::pair<int64_t,int64_t>{10 + 30 + 70, 3}));
    EXPECT_EQ(got[2], (std::pair<int64_t,int64_t>{20 + 50,      2}));
    EXPECT_EQ(got[3], (std::pair<int64_t,int64_t>{40 + 60,      2}));
}

TEST(HashAggregate, NullKeyIsItsOwnGroup) {
    MemoryManager mm(64 * 1024 * 1024);
    auto child = scan(mm, mkBatches(mm,
                                    {{1, 0, 1, 0, 2}},
                                    {{10, 11, 12, 13, 14}},
                                    {{true, false, true, false, true}}));

    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<CountStar>());

    HashAggregateOperator op(&mm, std::move(child), singleKey(0),
                             std::move(aggs), {"k", "sum_v", "cnt"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);  // {NULL}, 1, 2

    // Collect: for each output row, key (or nullopt) → (sum, cnt).
    std::map<std::optional<int64_t>, std::pair<int64_t, int64_t>> got;
    for (size_t i = 0; i < batch->numRows(); ++i) {
        std::optional<int64_t> key;
        if (batch->column(0).validity().isValid(i)) key = batch->column(0).typed<int64_t>()[i];
        got[key] = {batch->column(1).typed<int64_t>()[i],
                    batch->column(2).typed<int64_t>()[i]};
    }
    EXPECT_EQ(got[std::nullopt], (std::pair<int64_t,int64_t>{11 + 13, 2}));
    EXPECT_EQ(got[int64_t{1}],   (std::pair<int64_t,int64_t>{10 + 12, 2}));
    EXPECT_EQ(got[int64_t{2}],   (std::pair<int64_t,int64_t>{14,      1}));
}

TEST(HashAggregate, MinMaxAvgEndToEnd) {
    MemoryManager mm(64 * 1024 * 1024);
    auto child = scan(mm, mkBatches(mm,
                                    {{1, 1, 1, 2, 2}},
                                    {{10, 30, 20, 5, 15}}));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<MinI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<MaxI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<AvgI64>(std::make_unique<ColumnRef>(1)));

    HashAggregateOperator op(&mm, std::move(child), singleKey(0),
                             std::move(aggs), {"k", "mn", "mx", "av"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 2u);

    std::map<int64_t, std::tuple<int64_t, int64_t, double>> got;
    for (size_t i = 0; i < batch->numRows(); ++i) {
        got[batch->column(0).typed<int64_t>()[i]] = {
            batch->column(1).typed<int64_t>()[i],
            batch->column(2).typed<int64_t>()[i],
            batch->column(3).typed<double>()[i],
        };
    }
    EXPECT_EQ(std::get<0>(got[1]), 10);
    EXPECT_EQ(std::get<1>(got[1]), 30);
    EXPECT_DOUBLE_EQ(std::get<2>(got[1]), 20.0);

    EXPECT_EQ(std::get<0>(got[2]), 5);
    EXPECT_EQ(std::get<1>(got[2]), 15);
    EXPECT_DOUBLE_EQ(std::get<2>(got[2]), 10.0);
}

TEST(HashAggregate, TwoColumnGroupKey) {
    MemoryManager mm(64 * 1024 * 1024);

    // 3-column batches: (k1, k2, v). Build by hand because mkBatches is 2-col.
    auto makeBatch = [&](const std::vector<int64_t>& k1,
                         const std::vector<int64_t>& k2,
                         const std::vector<int64_t>& v) {
        const size_t n = k1.size();
        auto c1 = ColumnVector::make(Int64, n, &mm);
        auto c2 = ColumnVector::make(Int64, n, &mm);
        auto cv = ColumnVector::make(Int64, n, &mm);
        for (size_t i = 0; i < n; ++i) {
            c1.typed<int64_t>()[i] = k1[i];
            c2.typed<int64_t>()[i] = k2[i];
            cv.typed<int64_t>()[i] = v[i];
        }
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(c1));
        cols.push_back(std::move(c2));
        cols.push_back(std::move(cv));
        return ColumnBatch(std::move(cols), PositionList::identity(n));
    };

    std::vector<ColumnBatch> src;
    src.push_back(makeBatch({1, 1, 2, 1, 2},
                            {10, 20, 10, 10, 10},
                            {100, 200, 300, 400, 500}));
    auto scanOp = std::make_unique<ScanOperator>(
        &mm, Schema({{"k1", Int64}, {"k2", Int64}, {"v", Int64}}), std::move(src));

    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    keys.push_back(std::make_unique<ColumnRef>(1));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(2)));

    HashAggregateOperator op(&mm, std::move(scanOp), std::move(keys),
                             std::move(aggs), {"k1", "k2", "sum"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);  // (1,10), (1,20), (2,10)

    std::map<std::pair<int64_t,int64_t>, int64_t> got;
    for (size_t i = 0; i < batch->numRows(); ++i) {
        got[{batch->column(0).typed<int64_t>()[i],
             batch->column(1).typed<int64_t>()[i]}] = batch->column(2).typed<int64_t>()[i];
    }
    EXPECT_EQ((got[{1, 10}]), 100 + 400);
    EXPECT_EQ((got[{1, 20}]), 200);
    EXPECT_EQ((got[{2, 10}]), 300 + 500);
}

// Regression (perf-compare bug): inserting more new groups in ONE batch
// than the hash table's grow threshold (1024 * 0.7 = 716) used to rehash
// mid-batch and dangle every already-collected state pointer — silently
// dropping updates. The operator now reserves for the batch up front.
TEST(HashAggregate, ManyNewGroupsInOneBatchSurviveRehash) {
    MemoryManager mm(64 * 1024 * 1024);

    // Fixed path: single INT64 key, 2000 distinct keys × 2 rows, one batch.
    constexpr int64_t kGroups = 2000;
    const size_t n = static_cast<size_t>(kGroups) * 2;
    auto k = ColumnVector::make(Int64, n, &mm);
    auto v = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        k.typed<int64_t>()[i] = static_cast<int64_t>(i) % kGroups;
        v.typed<int64_t>()[i] = 1;
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    std::vector<ColumnBatch> src;
    src.emplace_back(std::move(cols), PositionList::identity(n));

    auto scan = std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<CountStar>());
    HashAggregateOperator op(&mm, std::move(scan), std::move(keys),
                             std::move(aggs), {"k", "s", "c"});

    op.open();
    size_t groups = 0;
    int64_t totalCount = 0;
    while (auto b = op.getNext()) {
        for (size_t i = 0; i < b->numRows(); ++i) {
            EXPECT_EQ(b->column(1).typed<int64_t>()[i], 2)
                << "group " << b->column(0).typed<int64_t>()[i];
            totalCount += b->column(2).typed<int64_t>()[i];
            ++groups;
        }
    }
    op.close();
    EXPECT_EQ(groups, static_cast<size_t>(kGroups));
    EXPECT_EQ(totalCount, static_cast<int64_t>(n));
}

TEST(HashAggregate, ManyNewGroupsVarPathSurviveRehash) {
    MemoryManager mm(64 * 1024 * 1024);

    // Var path: (STRING, INT64) composite key, 1500 distinct keys × 2 rows.
    constexpr int64_t kGroups = 1500;
    const size_t n = static_cast<size_t>(kGroups) * 2;
    StringBuilder sb(&mm);
    auto b2 = ColumnVector::make(Int64, n, &mm);
    auto v  = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        const auto g = static_cast<int64_t>(i) % kGroups;
        sb.append("key_" + std::to_string(g % 50));
        b2.typed<int64_t>()[i] = g;
        v.typed<int64_t>()[i]  = 1;
    }
    std::vector<ColumnVector> cols;
    cols.push_back(sb.build());
    cols.push_back(std::move(b2));
    cols.push_back(std::move(v));
    std::vector<ColumnBatch> src;
    src.emplace_back(std::move(cols), PositionList::identity(n));

    auto scan = std::make_unique<ScanOperator>(
        &mm,
        Schema({{"s", DataType{TypeId::STRING}}, {"b", Int64}, {"v", Int64}}),
        std::move(src));
    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    keys.push_back(std::make_unique<ColumnRef>(1));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<CountStar>());
    HashAggregateOperator op(&mm, std::move(scan), std::move(keys),
                             std::move(aggs), {"s", "b", "c"});

    op.open();
    size_t groups = 0;
    while (auto b = op.getNext()) {
        for (size_t i = 0; i < b->numRows(); ++i) {
            EXPECT_EQ(b->column(2).typed<int64_t>()[i], 2);
            ++groups;
        }
    }
    op.close();
    EXPECT_EQ(groups, static_cast<size_t>(kGroups));
}

// --- Int64FastPath (P1) ---

namespace {

std::map<std::optional<int64_t>, std::pair<int64_t, int64_t>> runInt64Agg(
        MemoryManager& mm, std::vector<ColumnBatch> src, size_t spillTrigger) {
    auto scan = std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<CountStar>());
    // spillTrigger == 0 selects Int64FastPath; > 0 forces the Fixed path.
    HashAggregateOperator op(&mm, std::move(scan), std::move(keys),
                             std::move(aggs), {"k", "s", "c"}, spillTrigger);
    std::map<std::optional<int64_t>, std::pair<int64_t, int64_t>> got;
    op.open();
    while (auto b = op.getNext()) {
        for (size_t i = 0; i < b->numRows(); ++i) {
            std::optional<int64_t> k;
            if (b->column(0).validity().isValid(i)) {
                k = b->column(0).typed<int64_t>()[i];
            }
            got[k] = {b->column(1).typed<int64_t>()[i],
                      b->column(2).typed<int64_t>()[i]};
        }
    }
    op.close();
    return got;
}

ColumnBatch kvNullableBatch(MemoryManager& mm,
                            const std::vector<int64_t>& ks,
                            const std::vector<bool>&    kValid,
                            const std::vector<int64_t>& vs) {
    auto k = ColumnVector::make(Int64, ks.size(), &mm);
    auto v = ColumnVector::make(Int64, vs.size(), &mm);
    k.validity().setAllValid();
    for (size_t i = 0; i < ks.size(); ++i) {
        k.typed<int64_t>()[i] = ks[i];
        v.typed<int64_t>()[i] = vs[i];
        if (!kValid.empty()) k.validity().setValid(i, kValid[i]);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(ks.size()));
}

} // namespace

TEST(HashAggregate, Int64FastPathMatchesFixedPath) {
    MemoryManager mm(64 * 1024 * 1024);
    auto mkSrc = [&] {
        std::vector<ColumnBatch> src;
        // 5000 groups across batches — forces several i64Grow rehashes —
        // with negative keys, dupes across batches, and null keys.
        for (int b = 0; b < 3; ++b) {
            std::vector<int64_t> ks, vs;
            std::vector<bool>    valid;
            for (int i = 0; i < 4000; ++i) {
                const int64_t g = (b * 4000 + i) % 5000;
                ks.push_back(g - 2500);  // negatives included
                vs.push_back(i);
                valid.push_back(i % 97 != 0);  // sprinkle null keys
            }
            src.push_back(kvNullableBatch(mm, ks, valid, vs));
        }
        return src;
    };

    auto fast    = runInt64Agg(mm, mkSrc(), /*spillTrigger=*/0);
    // A low group trigger forces several spill+merge round-trips through
    // the SAME fast path (the path no longer falls back when spill is
    // configured).
    auto spilled = runInt64Agg(mm, mkSrc(), /*spillTrigger=*/700);
    EXPECT_EQ(fast.size(), spilled.size());
    EXPECT_EQ(fast, spilled);
    EXPECT_TRUE(fast.count(std::nullopt) == 1);  // null group present
}
