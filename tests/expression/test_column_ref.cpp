#include <gtest/gtest.h>

#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

TEST(ColumnRef, ReturnsReferencedColumn) {
    MemoryManager mm(8192);

    auto a = ColumnVector::make(Int64, 8, &mm);
    auto b = ColumnVector::make(Int64, 8, &mm);
    for (size_t i = 0; i < 8; ++i) {
        a.typed<int64_t>()[i] = static_cast<int64_t>(i);
        b.typed<int64_t>()[i] = static_cast<int64_t>(i * 10);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(a));
    cols.push_back(std::move(b));
    ColumnBatch batch(std::move(cols), PositionList::identity(8));

    ColumnRef ref(1);
    auto result = ref.eval(batch, &mm);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(result.typed<int64_t>()[i], static_cast<int64_t>(i * 10));
    }
}
