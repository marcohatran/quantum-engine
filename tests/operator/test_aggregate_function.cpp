#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"

using namespace quantum;

namespace {
ColumnBatch oneIntCol(MemoryManager& mm, std::vector<int64_t> vals,
                      std::vector<bool> validity = {}) {
    auto col = ColumnVector::make(Int64, vals.size(), &mm);
    auto* p  = col.typed<int64_t>();
    for (size_t i = 0; i < vals.size(); ++i) p[i] = vals[i];
    if (!validity.empty()) {
        for (size_t i = 0; i < validity.size(); ++i) {
            col.validity().setValid(i, validity[i]);
        }
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    return ColumnBatch(std::move(cols), PositionList::identity(vals.size()));
}
} // namespace

TEST(SumI64, SumsAllValues) {
    MemoryManager mm(64 * 1024);
    SumI64 agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {1, 2, 3, 4, 5});

    uint8_t state[8];
    agg.init(state);
    auto input = agg.inputExpr()->eval(batch, &mm);
    std::vector<uint8_t*> slots(5, state);
    agg.update(slots.data(), &input, batch.positions());

    int64_t sum;
    std::memcpy(&sum, state, sizeof(sum));
    EXPECT_EQ(sum, 15);
}

TEST(SumI64, SkipsNulls) {
    MemoryManager mm(64 * 1024);
    SumI64 agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {1, 2, 100, 4, 5}, {true, true, false, true, true});

    uint8_t state[8];
    agg.init(state);
    auto input = agg.inputExpr()->eval(batch, &mm);
    std::vector<uint8_t*> slots(5, state);
    agg.update(slots.data(), &input, batch.positions());

    int64_t sum;
    std::memcpy(&sum, state, sizeof(sum));
    EXPECT_EQ(sum, 12);  // 1+2+4+5; 100 skipped
}

TEST(CountStar, CountsActiveRowsIncludingNulls) {
    MemoryManager mm(64 * 1024);
    CountStar agg;
    auto batch = oneIntCol(mm, {1, 2, 3, 4}, {true, false, true, false});

    uint8_t state[8];
    agg.init(state);
    std::vector<uint8_t*> slots(4, state);
    agg.update(slots.data(), nullptr, batch.positions());

    int64_t cnt;
    std::memcpy(&cnt, state, sizeof(cnt));
    EXPECT_EQ(cnt, 4);
}

TEST(CountColumn, CountsNonNullRowsOnly) {
    MemoryManager mm(64 * 1024);
    CountColumn agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {1, 2, 3, 4, 5}, {true, false, true, false, true});

    uint8_t state[8];
    agg.init(state);
    auto input = agg.inputExpr()->eval(batch, &mm);
    std::vector<uint8_t*> slots(5, state);
    agg.update(slots.data(), &input, batch.positions());

    int64_t cnt;
    std::memcpy(&cnt, state, sizeof(cnt));
    EXPECT_EQ(cnt, 3);  // rows 0, 2, 4 are valid; 1 and 3 are NULL.
}

TEST(CountColumn, AllNullsCountsZero) {
    MemoryManager mm(64 * 1024);
    CountColumn agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {1, 2, 3}, {false, false, false});

    uint8_t state[8];
    agg.init(state);
    auto input = agg.inputExpr()->eval(batch, &mm);
    std::vector<uint8_t*> slots(3, state);
    agg.update(slots.data(), &input, batch.positions());

    int64_t cnt;
    std::memcpy(&cnt, state, sizeof(cnt));
    EXPECT_EQ(cnt, 0);
}

namespace {
// Drive an aggregate against a single all-into-one-group batch and return
// the finalized output as a one-row ColumnVector. Convenience for tests.
ColumnVector runAggSingleGroup(MemoryManager& mm,
                               AggregateFunction& agg,
                               const ColumnBatch& batch) {
    std::vector<uint8_t> state(agg.stateSize());
    agg.init(state.data());
    std::optional<ColumnVector> input;
    if (const Expression* e = agg.inputExpr()) input = e->eval(batch, &mm);
    std::vector<uint8_t*> slots(batch.activeRows(), state.data());
    agg.update(slots.data(), input.has_value() ? &*input : nullptr, batch.positions());
    return agg.finalize({state.data()}, &mm);
}
} // namespace

TEST(MinI64, FindsMinAndSkipsNulls) {
    MemoryManager mm(64 * 1024);
    MinI64 agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {7, 100, 3, -2, 5}, {true, false, true, true, true});
    auto out = runAggSingleGroup(mm, agg, batch);
    EXPECT_TRUE(out.validity().isValid(0));
    EXPECT_EQ(out.typed<int64_t>()[0], -2);
}

TEST(MinI64, EmptyGroupReturnsNull) {
    MemoryManager mm(64 * 1024);
    MinI64 agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {1, 2, 3}, {false, false, false});
    auto out = runAggSingleGroup(mm, agg, batch);
    EXPECT_FALSE(out.validity().isValid(0));
}

TEST(MaxI64, FindsMax) {
    MemoryManager mm(64 * 1024);
    MaxI64 agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {7, -100, 3, 42, 5});
    auto out = runAggSingleGroup(mm, agg, batch);
    EXPECT_EQ(out.typed<int64_t>()[0], 42);
}

TEST(AvgI64, ComputesMeanOfActiveValues) {
    MemoryManager mm(64 * 1024);
    AvgI64 agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {1, 2, 3, 4, 100}, {true, true, true, true, false});
    auto out = runAggSingleGroup(mm, agg, batch);
    EXPECT_TRUE(out.validity().isValid(0));
    EXPECT_DOUBLE_EQ(out.typed<double>()[0], (1.0 + 2 + 3 + 4) / 4.0);
}

TEST(AvgI64, EmptyGroupReturnsNull) {
    MemoryManager mm(64 * 1024);
    AvgI64 agg(std::make_unique<ColumnRef>(0));
    auto batch = oneIntCol(mm, {1, 2}, {false, false});
    auto out = runAggSingleGroup(mm, agg, batch);
    EXPECT_FALSE(out.validity().isValid(0));
}

#include "vector/StringBuilder.hpp"

namespace {
ColumnBatch oneStringCol(MemoryManager& mm,
                         const std::vector<std::string>& vals,
                         const std::vector<bool>&        validity = {}) {
    StringBuilder sb(&mm);
    for (size_t i = 0; i < vals.size(); ++i) {
        if (!validity.empty() && !validity[i]) sb.appendNull();
        else                                   sb.append(vals[i]);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(sb.build());
    return ColumnBatch(std::move(cols), PositionList::identity(vals.size()));
}
} // namespace

TEST(MinString, FindsLexMin) {
    MemoryManager mm(64 * 1024);
    MinString agg(std::make_unique<ColumnRef>(0));
    auto batch = oneStringCol(mm, {"banana", "apple", "cherry"});
    auto out = runAggSingleGroup(mm, agg, batch);
    EXPECT_TRUE(out.validity().isValid(0));
    EXPECT_EQ(out.stringAt(0), "apple");
}

TEST(MaxString, FindsLexMax) {
    MemoryManager mm(64 * 1024);
    MaxString agg(std::make_unique<ColumnRef>(0));
    auto batch = oneStringCol(mm, {"banana", "apple", "cherry"});
    auto out = runAggSingleGroup(mm, agg, batch);
    EXPECT_TRUE(out.validity().isValid(0));
    EXPECT_EQ(out.stringAt(0), "cherry");
}

TEST(MinString, SkipsNullsAndAllNullReturnsNull) {
    MemoryManager mm(64 * 1024);
    {
        MinString agg(std::make_unique<ColumnRef>(0));
        auto batch = oneStringCol(mm, {"x", "apple", "y"}, {false, true, false});
        auto out = runAggSingleGroup(mm, agg, batch);
        EXPECT_TRUE(out.validity().isValid(0));
        EXPECT_EQ(out.stringAt(0), "apple");
    }
    {
        MinString agg(std::make_unique<ColumnRef>(0));
        auto batch = oneStringCol(mm, {"x", "y"}, {false, false});
        auto out = runAggSingleGroup(mm, agg, batch);
        EXPECT_FALSE(out.validity().isValid(0));
    }
}
