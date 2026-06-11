#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>

#include "common/Macros.hpp"
#include "io/FileFormat.hpp"

namespace quantum {

// Decode-ahead decorator: runs the wrapped reader's open() + nextBatch()
// loop on a TaskScheduler task, pushing batches into a small bounded
// queue the consumer pops from. Overlaps file decode with downstream
// compute (filter / aggregation) — the scan cost hides behind whatever
// the rest of the pipeline is doing.
//
// Threading: the inner reader is touched ONLY by the producer task;
// ColumnBatches cross threads but their buffers live in the thread-safe
// MemoryManager. Producer exceptions are captured and rethrown from
// nextBatch() on the consumer.
class PrefetchReader final : public FileFormatReader {
public:
    explicit PrefetchReader(FileFormatReaderPtr inner, size_t queueDepth = 4);
    ~PrefetchReader() override;

    QUANTUM_DISALLOW_COPY(PrefetchReader);

    void                       open() override;
    std::optional<ColumnBatch> nextBatch() override;
    void                       close() override;

    const Schema& schema() const override { return inner_->schema(); }

private:
    void joinProducer();

    FileFormatReaderPtr inner_;
    size_t              queueDepth_;

    std::mutex              mu_;
    std::condition_variable cv_;
    std::deque<ColumnBatch> queue_;
    bool                    producerDone_ = false;
    bool                    stop_         = false;
    bool                    started_      = false;
    std::exception_ptr      error_;
};

} // namespace quantum
