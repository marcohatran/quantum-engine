#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/ScanOperator.hpp"
#include "operator/SortOperator.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch kvBatch(MemoryManager& mm,
                    const std::vector<int64_t>& k,
                    const std::vector<int64_t>& v) {
    auto kc = ColumnVector::make(Int64, k.size(), &mm);
    auto vc = ColumnVector::make(Int64, v.size(), &mm);
    for (size_t i = 0; i < k.size(); ++i) {
        kc.typed<int64_t>()[i] = k[i];
        vc.typed<int64_t>()[i] = v[i];
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(kc));
    cols.push_back(std::move(vc));
    return ColumnBatch(std::move(cols), PositionList::identity(k.size()));
}

OperatorPtr mkScan(MemoryManager& mm, std::vector<ColumnBatch> batches) {
    return std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(batches));
}

std::vector<std::pair<int64_t,int64_t>> drainKV(Operator& op) {
    std::vector<std::pair<int64_t,int64_t>> out;
    op.open();
    while (auto b = op.getNext()) {
        const auto* k = b->column(0).typed<int64_t>();
        const auto* v = b->column(1).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) out.emplace_back(k[i], v[i]);
    }
    op.close();
    return out;
}

} // namespace

TEST(Sort, AscendingByFirstKey) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(kvBatch(mm, {3, 1, 5, 2}, {30, 10, 50, 20}));
    in.push_back(kvBatch(mm, {4},          {40}));

    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    SortOperator op(&mm, mkScan(mm, std::move(in)),
                    std::move(keys),
                    /*ascending=*/{true},
                    /*nullsFirst=*/{true});

    auto got = drainKV(op);
    using P = std::pair<int64_t,int64_t>;
    EXPECT_EQ(got, std::vector<P>({{1,10},{2,20},{3,30},{4,40},{5,50}}));
}

TEST(Sort, DescendingBySecondKey) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> in;
    in.push_back(kvBatch(mm, {1,1,1,2,2}, {30, 10, 20, 50, 40}));

    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    keys.push_back(std::make_unique<ColumnRef>(1));
    SortOperator op(&mm, mkScan(mm, std::move(in)),
                    std::move(keys),
                    /*ascending=*/{true, false},
                    /*nullsFirst=*/{true, false});

    auto got = drainKV(op);
    using P = std::pair<int64_t,int64_t>;
    // k=1 group sorted by v desc; k=2 group sorted by v desc; outer k asc.
    EXPECT_EQ(got, std::vector<P>({{1,30},{1,20},{1,10},{2,50},{2,40}}));
}

TEST(Sort, EmptyInput) {
    MemoryManager mm(64 * 1024);
    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    SortOperator op(&mm, mkScan(mm, {}),
                    std::move(keys), {true}, {true});
    auto got = drainKV(op);
    EXPECT_EQ(got.size(), 0u);
}

// --- External sort (spilled runs + k-way merge), Phase 5.42 ---

namespace {

// (k INT64 nullable, s STRING nullable) batch.
ColumnBatch ksBatch(MemoryManager& mm,
                    const std::vector<int64_t>&     k,
                    const std::vector<bool>&        kValid,
                    const std::vector<std::string>& s,
                    const std::vector<bool>&        sValid) {
    auto kc = ColumnVector::make(Int64, k.size(), &mm);
    kc.validity().setAllValid();
    for (size_t i = 0; i < k.size(); ++i) {
        kc.typed<int64_t>()[i] = k[i];
        kc.validity().setValid(i, kValid[i]);
    }
    StringBuilder sb(&mm);
    for (size_t i = 0; i < s.size(); ++i) {
        if (sValid[i]) sb.append(s[i]);
        else           sb.appendNull();
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(kc));
    cols.push_back(sb.build());
    return ColumnBatch(std::move(cols), PositionList::identity(k.size()));
}

} // namespace

TEST(Sort, SpilledSortMatchesInMemory) {
    MemoryManager mm(64 * 1024 * 1024);

    auto mkIn = [&] {
        std::vector<ColumnBatch> in;
        in.push_back(kvBatch(mm, {3, 1, 5, 2}, {30, 10, 50, 20}));
        in.push_back(kvBatch(mm, {4, 1},       {40, 11}));
        in.push_back(kvBatch(mm, {2, 6},       {21, 60}));
        return in;
    };
    auto run = [&](size_t trigger) {
        std::vector<ExpressionPtr> keys;
        keys.push_back(std::make_unique<ColumnRef>(0));
        keys.push_back(std::make_unique<ColumnRef>(1));
        SortOperator op(&mm, mkScan(mm, mkIn()),
                        std::move(keys),
                        /*ascending=*/{true, false},
                        /*nullsFirst=*/{true, true},
                        /*spillTriggerRows=*/trigger);
        return drainKV(op);
    };

    auto baseline = run(0);   // in-memory
    auto spilled  = run(2);   // a run per input batch
    EXPECT_EQ(spilled, baseline);
    // k asc, v desc within equal k: (1,11),(1,10),(2,21),(2,20),...
    EXPECT_EQ(baseline.front(), (std::pair<int64_t,int64_t>{1, 11}));
    EXPECT_EQ(baseline.back(),  (std::pair<int64_t,int64_t>{6, 60}));
}

TEST(Sort, SpilledSortStringsAndNulls) {
    MemoryManager mm(64 * 1024 * 1024);

    auto mkIn = [&] {
        std::vector<ColumnBatch> in;
        in.push_back(ksBatch(mm, {2, 0, 1}, {true, false, true},
                             {"pear", "fig", "apple"}, {true, true, true}));
        in.push_back(ksBatch(mm, {3, 4}, {true, true},
                             {"", "kiwi"}, {false, true}));
        return in;
    };
    auto run = [&](size_t trigger) {
        std::vector<ExpressionPtr> keys;
        keys.push_back(std::make_unique<ColumnRef>(1));  // sort by the string
        auto scan = std::make_unique<ScanOperator>(
            &mm,
            Schema({{"k", Int64}, {"s", DataType{TypeId::STRING}}}),
            mkIn());
        SortOperator op(&mm, std::move(scan), std::move(keys),
                        /*ascending=*/{true},
                        /*nullsFirst=*/{false},
                        /*spillTriggerRows=*/trigger);
        std::vector<std::pair<std::optional<int64_t>, std::optional<std::string>>> out;
        op.open();
        while (auto b = op.getNext()) {
            for (size_t i = 0; i < b->numRows(); ++i) {
                std::optional<int64_t> k;
                if (b->column(0).validity().isValid(i)) {
                    k = b->column(0).typed<int64_t>()[i];
                }
                std::optional<std::string> s;
                if (b->column(1).validity().isValid(i)) {
                    s = std::string(b->column(1).stringAt(i));
                }
                out.emplace_back(k, s);
            }
        }
        op.close();
        return out;
    };

    auto baseline = run(0);
    auto spilled  = run(1);   // every batch becomes a run
    EXPECT_EQ(spilled, baseline);
    // ascending by s, nulls last: apple, fig, kiwi, pear, NULL.
    ASSERT_EQ(baseline.size(), 5u);
    EXPECT_EQ(baseline[0].second, std::optional<std::string>("apple"));
    EXPECT_EQ(baseline[1].second, std::optional<std::string>("fig"));
    EXPECT_EQ(baseline[1].first,  std::nullopt);  // k null travels with row
    EXPECT_EQ(baseline[4].second, std::nullopt);
    EXPECT_EQ(baseline[4].first,  std::optional<int64_t>(3));
}

TEST(Sort, SpilledSortStreamsMultipleBatches) {
    MemoryManager mm(64 * 1024 * 1024);

    // 10k rows in descending order across 10 batches of 1000; merged output
    // must come back ascending and span multiple 4096-row output batches.
    std::vector<ColumnBatch> in;
    int64_t next = 10000;
    for (int b = 0; b < 10; ++b) {
        std::vector<int64_t> k, v;
        for (int i = 0; i < 1000; ++i) { k.push_back(next); v.push_back(next); --next; }
        in.push_back(kvBatch(mm, k, v));
    }

    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    SortOperator op(&mm, mkScan(mm, std::move(in)),
                    std::move(keys),
                    /*ascending=*/{true},
                    /*nullsFirst=*/{true},
                    /*spillTriggerRows=*/2500);

    size_t batches = 0;
    int64_t expect = 1;
    op.open();
    while (auto b = op.getNext()) {
        ++batches;
        const auto* k = b->column(0).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) {
            ASSERT_EQ(k[i], expect);
            ++expect;
        }
    }
    op.close();
    EXPECT_EQ(expect, 10001);   // all 10k rows, globally ordered
    EXPECT_GE(batches, 2u);     // streamed, not one giant batch
}

TEST(Sort, SpillTriggerWithEmptyInput) {
    MemoryManager mm(64 * 1024);
    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    SortOperator op(&mm, mkScan(mm, {}),
                    std::move(keys),
                    /*ascending=*/{true},
                    /*nullsFirst=*/{true},
                    /*spillTriggerRows=*/1);
    op.open();
    EXPECT_FALSE(op.getNext().has_value());
    op.close();
}
