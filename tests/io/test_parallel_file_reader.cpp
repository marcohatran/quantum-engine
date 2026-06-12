#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/Status.hpp"
#include "io/ParallelFileReader.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

namespace {

// Emits `count` single-row batches carrying base, base+1, ...; can be told
// to throw partway through.
class StubReader final : public FileFormatReader {
public:
    StubReader(MemoryManager* mm, int base, int count, int throwAt = -1)
        : mm_(mm), base_(base), count_(count), throwAt_(throwAt),
          schema_({{"v", Int64}}) {}

    void open() override {
        next_ = 0;
        opened_ = true;
    }
    void close() override { opened_ = false; }
    const Schema& schema() const override { return schema_; }

    std::optional<ColumnBatch> nextBatch() override {
        if (!opened_) throw std::logic_error("nextBatch on closed reader");
        if (next_ == throwAt_) throw std::runtime_error("stub failure");
        if (next_ >= count_) return std::nullopt;
        auto col = ColumnVector::make(Int64, 1, mm_);
        col.validity().setAllValid();
        col.typed<int64_t>()[0] = base_ + next_++;
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(col));
        return ColumnBatch(std::move(cols), PositionList::identity(1));
    }

private:
    MemoryManager* mm_;
    int            base_;
    int            count_;
    int            throwAt_;
    int            next_   = 0;
    bool           opened_ = false;
    Schema         schema_;
};

// Batch order across files is nondeterministic — collect and sort.
std::vector<int64_t> drainSorted(FileFormatReader& reader) {
    std::vector<int64_t> got;
    reader.open();
    while (auto b = reader.nextBatch()) {
        got.push_back(b->column(0).typed<int64_t>()[0]);
    }
    reader.close();
    std::sort(got.begin(), got.end());
    return got;
}

std::vector<int64_t> iota(int n) {
    std::vector<int64_t> v(n);
    for (int i = 0; i < n; ++i) v[i] = i;
    return v;
}

std::vector<FileFormatReaderPtr> stubFiles(MemoryManager* mm, int files,
                                           int rowsPerFile) {
    std::vector<FileFormatReaderPtr> readers;
    for (int f = 0; f < files; ++f) {
        readers.push_back(std::make_unique<StubReader>(
            mm, f * rowsPerFile, rowsPerFile));
    }
    return readers;
}

} // namespace

TEST(ParallelFileReader, AllRowsFromAllFilesExactlyOnce) {
    MemoryManager mm(1 << 20);
    // Parallelism above, at, and below the file count, incl. more
    // producers than scheduler threads (extras exit as no-ops).
    for (size_t par : {1u, 2u, 5u, 32u}) {
        ParallelFileReader reader(stubFiles(&mm, 5, 40), par);
        EXPECT_EQ(drainSorted(reader), iota(200)) << "parallelism=" << par;
    }
}

TEST(ParallelFileReader, SingleFileWorks) {
    MemoryManager mm(1 << 20);
    ParallelFileReader reader(stubFiles(&mm, 1, 25), 4);
    EXPECT_EQ(drainSorted(reader), iota(25));
}

TEST(ParallelFileReader, EmptyFilesMixedIn) {
    MemoryManager mm(1 << 20);
    std::vector<FileFormatReaderPtr> readers;
    readers.push_back(std::make_unique<StubReader>(&mm, 0, 0));
    readers.push_back(std::make_unique<StubReader>(&mm, 0, 10));
    readers.push_back(std::make_unique<StubReader>(&mm, 0, 0));
    readers.push_back(std::make_unique<StubReader>(&mm, 10, 10));
    ParallelFileReader reader(std::move(readers), 3);
    EXPECT_EQ(drainSorted(reader), iota(20));
}

TEST(ParallelFileReader, ProducerExceptionRethrownOnConsumer) {
    MemoryManager mm(1 << 20);
    std::vector<FileFormatReaderPtr> readers;
    for (int f = 0; f < 4; ++f) {
        readers.push_back(std::make_unique<StubReader>(
            &mm, f * 100, 100, /*throwAt=*/f == 2 ? 7 : -1));
    }
    ParallelFileReader reader(std::move(readers), 4);
    reader.open();
    try {
        while (reader.nextBatch()) {}
        FAIL() << "expected the stub failure to propagate";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "stub failure");
    }
    reader.close();
}

TEST(ParallelFileReader, EarlyCloseUnblocksThrottledProducers) {
    MemoryManager mm(1 << 20);
    // Files much larger than the queue capacity: producers will be
    // throttled mid-file when close() arrives. Must unblock + join.
    ParallelFileReader reader(stubFiles(&mm, 4, 50000), 4, /*queueDepth=*/4);
    reader.open();
    ASSERT_TRUE(reader.nextBatch().has_value());
    reader.close();
}

TEST(ParallelFileReader, ReopenAfterCloseWorks) {
    MemoryManager mm(1 << 20);
    ParallelFileReader reader(stubFiles(&mm, 3, 10), 2);
    for (int round = 0; round < 3; ++round) {
        EXPECT_EQ(drainSorted(reader), iota(30)) << "round=" << round;
    }
}

TEST(ParallelFileReader, CloseImmediatelyAfterOpen) {
    MemoryManager mm(1 << 20);
    ParallelFileReader reader(stubFiles(&mm, 8, 10000), 8, /*queueDepth=*/2);
    reader.open();
    reader.close();  // producers may not have started yet — must not hang
    reader.close();  // idempotent
}
