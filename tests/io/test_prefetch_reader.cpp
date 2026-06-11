#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "common/Status.hpp"
#include "io/PrefetchReader.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

namespace {

// Produces `count` single-column batches with ascending values; can be
// told to throw partway through.
class StubReader final : public FileFormatReader {
public:
    StubReader(MemoryManager* mm, int count, int throwAt = -1)
        : mm_(mm), count_(count), throwAt_(throwAt),
          schema_({{"v", Int64}}) {}

    void open() override { next_ = 0; }
    void close() override {}
    const Schema& schema() const override { return schema_; }

    std::optional<ColumnBatch> nextBatch() override {
        if (next_ == throwAt_) throw std::runtime_error("stub failure");
        if (next_ >= count_) return std::nullopt;
        auto col = ColumnVector::make(Int64, 1, mm_);
        col.validity().setAllValid();
        col.typed<int64_t>()[0] = next_++;
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(col));
        return ColumnBatch(std::move(cols), PositionList::identity(1));
    }

private:
    MemoryManager* mm_;
    int            count_;
    int            throwAt_;
    int            next_ = 0;
    Schema         schema_;
};

} // namespace

TEST(PrefetchReader, PreservesBatchOrderAndCount) {
    MemoryManager mm(1 << 20);
    PrefetchReader reader(std::make_unique<StubReader>(&mm, 100), /*depth=*/3);
    reader.open();
    int expect = 0;
    while (auto b = reader.nextBatch()) {
        EXPECT_EQ(b->column(0).typed<int64_t>()[0], expect++);
    }
    reader.close();
    EXPECT_EQ(expect, 100);
}

TEST(PrefetchReader, ProducerExceptionRethrownOnConsumer) {
    MemoryManager mm(1 << 20);
    PrefetchReader reader(std::make_unique<StubReader>(&mm, 100, /*throwAt=*/5), 2);
    reader.open();
    int got = 0;
    try {
        while (reader.nextBatch()) ++got;
        FAIL() << "expected the stub failure to propagate";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "stub failure");
    }
    EXPECT_LE(got, 5);
    reader.close();
}

TEST(PrefetchReader, EarlyCloseDoesNotHang) {
    MemoryManager mm(1 << 20);
    PrefetchReader reader(std::make_unique<StubReader>(&mm, 100000), 2);
    reader.open();
    auto b = reader.nextBatch();
    ASSERT_TRUE(b.has_value());
    reader.close();  // producer throttled mid-stream — must unblock + join
}

TEST(PrefetchReader, ReopenAfterCloseWorks) {
    MemoryManager mm(1 << 20);
    PrefetchReader reader(std::make_unique<StubReader>(&mm, 3), 2);
    for (int round = 0; round < 2; ++round) {
        reader.open();
        int n = 0;
        while (reader.nextBatch()) ++n;
        EXPECT_EQ(n, 3);
        reader.close();
    }
}
