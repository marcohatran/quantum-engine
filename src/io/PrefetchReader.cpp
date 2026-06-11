#include "io/PrefetchReader.hpp"

#include <utility>

#include "common/Status.hpp"
#include "exec/TaskScheduler.hpp"

namespace quantum {

PrefetchReader::PrefetchReader(FileFormatReaderPtr inner, size_t queueDepth)
    : inner_(std::move(inner)), queueDepth_(queueDepth == 0 ? 1 : queueDepth) {
    if (!inner_) {
        raise(ErrorCode::InvalidArgument, "PrefetchReader: null inner reader");
    }
}

PrefetchReader::~PrefetchReader() { joinProducer(); }

void PrefetchReader::open() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.clear();
        producerDone_ = false;
        stop_         = false;
        error_        = nullptr;
        started_      = true;
    }
    TaskScheduler::global().submit([this] {
        try {
            inner_->open();
            while (true) {
                auto batch = inner_->nextBatch();
                std::unique_lock<std::mutex> lock(mu_);
                if (stop_) break;
                if (!batch.has_value()) break;
                cv_.wait(lock, [this] {
                    return stop_ || queue_.size() < queueDepth_;
                });
                if (stop_) break;
                queue_.push_back(std::move(*batch));
                cv_.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu_);
            error_ = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            producerDone_ = true;
        }
        cv_.notify_all();
    });
}

std::optional<ColumnBatch> PrefetchReader::nextBatch() {
    std::unique_lock<std::mutex> lock(mu_);
    if (!started_) {
        raise(ErrorCode::InvalidArgument, "PrefetchReader: nextBatch before open");
    }
    cv_.wait(lock, [this] { return !queue_.empty() || producerDone_; });
    if (!queue_.empty()) {
        ColumnBatch batch = std::move(queue_.front());
        queue_.pop_front();
        cv_.notify_all();  // wake the producer if it was throttled
        return batch;
    }
    if (error_) std::rethrow_exception(error_);
    return std::nullopt;
}

void PrefetchReader::close() {
    joinProducer();
    inner_->close();
}

void PrefetchReader::joinProducer() {
    std::unique_lock<std::mutex> lock(mu_);
    if (!started_) return;
    stop_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return producerDone_; });
    queue_.clear();
    started_ = false;
}

} // namespace quantum
