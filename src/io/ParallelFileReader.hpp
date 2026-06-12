#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "common/Macros.hpp"
#include "io/FileFormat.hpp"

namespace quantum {

// Merges the batches of many FileFormatReaders, decoding up to
// `parallelism` files concurrently. open() submits producer tasks to the
// TaskScheduler; each claims files from a shared cursor, decodes them, and
// pushes batches into one bounded queue the consumer pops from. Whenever
// the queue is empty and unclaimed files remain, the consumer claims one
// and decodes it inline — it participates instead of idling, and the scan
// keeps making progress even if every scheduler thread is busy (a join
// with two parallel scans cannot deadlock on pool exhaustion).
//
// Batches arrive in nondeterministic file order — callers must not rely
// on file order (SQL guarantees none without ORDER BY; the ordered
// alternative is ConcatFileReader).
//
// Threading: each inner reader is touched by exactly one claimer at a
// time. ColumnBatches cross threads but their buffers live in the
// thread-safe MemoryManager. End-of-stream is tracked per *file*
// (filesFinished == files), never by producer-task completion, so a
// submitted task that the pool never ran cannot stall the consumer.
// close() joins only the producers that are actually running; tasks still
// waiting in the pool queue keep the shared state alive via shared_ptr
// and exit as no-ops when their epoch has passed.
class ParallelFileReader final : public FileFormatReader {
public:
    // queueDepth = max batches buffered ahead (default 4 * parallelism).
    ParallelFileReader(std::vector<FileFormatReaderPtr> files,
                       size_t parallelism, size_t queueDepth = 0);
    ~ParallelFileReader() override;

    QUANTUM_DISALLOW_COPY(ParallelFileReader);

    void                       open() override;
    std::optional<ColumnBatch> nextBatch() override;
    void                       close() override;

    const Schema& schema() const override;

private:
    // Shared with producer tasks; outlives this reader if the pool runs a
    // task after destruction (it exits as a no-op but must be able to
    // lock the mutex and check the epoch).
    struct State {
        std::mutex                       mu;
        std::condition_variable          cvNotEmpty;  // consumer waits
        std::condition_variable          cvNotFull;   // producers wait
        std::vector<FileFormatReaderPtr> files;
        std::deque<ColumnBatch>          queue;
        size_t                           queueCap;
        size_t                           nextFile        = 0;
        size_t                           filesFinished   = 0;
        size_t                           activeProducers = 0;
        uint64_t                         epoch           = 0;
        bool                             stop            = false;
        std::exception_ptr               error;
    };

    static void producerLoop(const std::shared_ptr<State>& st, uint64_t epoch);

    std::shared_ptr<State> st_;
    size_t                 parallelism_;
    FileFormatReader*      inline_  = nullptr;  // file the consumer is decoding
    bool                   started_ = false;
};

} // namespace quantum
