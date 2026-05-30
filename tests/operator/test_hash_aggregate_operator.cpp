#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {

// Build batches with two int64 columns: (group_key, value).
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

} // namespace

TEST(HashAggregate, EmptyInputProducesEmptyOutput) {
    MemoryManager mm(64 * 1024 * 1024);
    auto child = scan(mm, {});
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(child),
                             std::make_unique<ColumnRef>(0),
                             std::move(aggs),
                             {"k", "sum"});
    op.open();
    auto batch = op.getNext();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->numRows(), 0u);
    EXPECT_FALSE(op.getNext().has_value());
    op.close();
}

TEST(HashAggregate, SingleGroupSum) {
    MemoryManager mm(64 * 1024 * 1024);
    auto child = scan(mm, mkBatches(mm,
                                    {{7, 7, 7, 7}},
                                    {{1, 2, 3, 4}}));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(child),
                             std::make_unique<ColumnRef>(0),
                             std::move(aggs),
                             {"k", "sum"});
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

    HashAggregateOperator op(&mm, std::move(child),
                             std::make_unique<ColumnRef>(0),
                             std::move(aggs),
                             {"k", "sum_v", "cnt"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);

    // Collect (k, sum, cnt) into a map for stable comparison.
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> got;
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
    auto child = scan(mm,
                      mkBatches(mm,
                                {{1, 0, 1, 0, 2}},
                                {{10, 11, 12, 13, 14}},
                                {{true, false, true, false, true}}));

    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    aggs.push_back(std::make_unique<CountStar>());

    HashAggregateOperator op(&mm, std::move(child),
                             std::make_unique<ColumnRef>(0),
                             std::move(aggs),
                             {"k", "sum_v", "cnt"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);  // null, 1, 2

    // Row 0 is the NULL group (operator emits it first).
    EXPECT_FALSE(batch->column(0).validity().isValid(0));
    EXPECT_EQ(batch->column(1).typed<int64_t>()[0], 11 + 13);
    EXPECT_EQ(batch->column(2).typed<int64_t>()[0], 2);

    // Remaining two are keyed.
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> got;
    for (size_t i = 1; i < 3; ++i) {
        got[batch->column(0).typed<int64_t>()[i]] = {
            batch->column(1).typed<int64_t>()[i],
            batch->column(2).typed<int64_t>()[i]
        };
    }
    EXPECT_EQ(got[1], (std::pair<int64_t,int64_t>{10 + 12, 2}));
    EXPECT_EQ(got[2], (std::pair<int64_t,int64_t>{14,      1}));
}
