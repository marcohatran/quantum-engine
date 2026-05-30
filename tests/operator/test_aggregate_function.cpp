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
