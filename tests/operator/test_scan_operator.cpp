#include <gtest/gtest.h>

#include <vector>

#include "memory/MemoryManager.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {
ColumnBatch makeBatch(MemoryManager& mm, size_t n, int64_t base) {
    auto col = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) col.typed<int64_t>()[i] = base + static_cast<int64_t>(i);
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(col));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}
} // namespace

TEST(ScanOperator, DrainsAllBatches) {
    MemoryManager mm(64 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(makeBatch(mm, 8, 0));
    src.push_back(makeBatch(mm, 8, 100));

    ScanOperator scan(&mm, Schema({{"a", Int64}}), std::move(src));
    scan.open();
    size_t batches = 0;
    while (auto b = scan.getNext()) ++batches;
    scan.close();
    EXPECT_EQ(batches, 2u);
}
