#include "io/ParallelFileReader.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "common/Status.hpp"
#include "exec/TaskScheduler.hpp"

namespace quantum {

ParallelFileReader::ParallelFileReader(std::vector<FileFormatReaderPtr> files,
                                       size_t parallelism, size_t queueDepth)
    : st_(std::make_shared<State>()),
      parallelism_(parallelism == 0 ? 1 : parallelism) {
    if (files.empty()) {
        raise(ErrorCode::InvalidArgument,
              "ParallelFileReader: requires at least one underlying reader");
    }
    st_->files    = std::move(files);
    st_->queueCap = queueDepth != 0 ? queueDepth : 4 * parallelism_;
}

ParallelFileReader::~ParallelFileReader() { close(); }

const Schema& ParallelFileReader::schema() const {
    return st_->files.front()->schema();
}

void ParallelFileReader::producerLoop(const std::shared_ptr<State>& st,
                                      uint64_t epoch) {
    std::unique_lock<std::mutex> lock(st->mu);
    if (st->epoch != epoch || st->stop) return;  // stale task — no-op
    ++st->activeProducers;
    while (!st->stop && st->nextFile < st->files.size()) {
        FileFormatReader* file = st->files[st->nextFile++].get();
        lock.unlock();
        try {
            file->open();
            while (true) {
                auto batch = file->nextBatch();
                if (!batch.has_value()) break;
                std::unique_lock<std::mutex> push(st->mu);
                st->cvNotFull.wait(push, [&] {
                    return st->stop || st->queue.size() < st->queueCap;
                });
                if (st->stop) break;
                st->queue.push_back(std::move(*batch));
                st->cvNotEmpty.notify_one();
            }
            file->close();
        } catch (...) {
            std::lock_guard<std::mutex> g(st->mu);
            st->error = std::current_exception();
            st->stop  = true;
            st->cvNotFull.notify_all();
        }
        lock.lock();
        ++st->filesFinished;
        st->cvNotEmpty.notify_one();  // consumer may be waiting for EOS
    }
    --st->activeProducers;
    st->cvNotEmpty.notify_all();
    st->cvNotFull.notify_all();  // close() waits on activeProducers == 0
}

void ParallelFileReader::open() {
    uint64_t epoch;
    {
        std::lock_guard<std::mutex> g(st_->mu);
        st_->queue.clear();
        st_->nextFile      = 0;
        st_->filesFinished = 0;
        st_->stop          = false;
        st_->error         = nullptr;
        epoch              = ++st_->epoch;
    }
    started_ = true;
    const size_t producers = std::min(parallelism_, st_->files.size());
    for (size_t p = 0; p < producers; ++p) {
        TaskScheduler::global().submit(
            [st = st_, epoch] { producerLoop(st, epoch); });
    }
}

std::optional<ColumnBatch> ParallelFileReader::nextBatch() {
    if (!started_) {
        raise(ErrorCode::InvalidArgument,
              "ParallelFileReader: nextBatch before open");
    }
    while (true) {
        if (inline_ != nullptr) {
            std::optional<ColumnBatch> batch;
            try {
                batch = inline_->nextBatch();
            } catch (...) {
                inline_ = nullptr;
                std::lock_guard<std::mutex> g(st_->mu);
                st_->stop = true;
                st_->cvNotFull.notify_all();
                throw;
            }
            if (batch.has_value()) return batch;
            inline_->close();
            inline_ = nullptr;
            std::lock_guard<std::mutex> g(st_->mu);
            ++st_->filesFinished;
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(st_->mu);
            while (true) {
                if (!st_->queue.empty()) {
                    ColumnBatch batch = std::move(st_->queue.front());
                    st_->queue.pop_front();
                    st_->cvNotFull.notify_one();  // free slot for a producer
                    return batch;
                }
                if (st_->error) std::rethrow_exception(st_->error);
                if (st_->filesFinished == st_->files.size()) return std::nullopt;
                // Starvation escape hatch, not a fast path: claim a file
                // ourselves only when no producer is running — i.e. the
                // pool never started our tasks (every thread is blocked
                // on some other scan's queue). Eagerly claiming while
                // producers run would serialize a whole file's decode
                // into this thread's compute loop.
                if (st_->activeProducers == 0 &&
                    st_->nextFile < st_->files.size()) {
                    inline_ = st_->files[st_->nextFile++].get();
                    break;  // open + decode it ourselves, outside the lock
                }
                // Bounded wait: producers may be about to start (or all
                // busy mid-file) — re-check for starvation on timeout.
                st_->cvNotEmpty.wait_for(lock, std::chrono::milliseconds(5));
            }
        }
        try {
            inline_->open();
        } catch (...) {
            inline_ = nullptr;
            std::lock_guard<std::mutex> g(st_->mu);
            st_->stop = true;
            st_->cvNotFull.notify_all();
            throw;
        }
    }
}

void ParallelFileReader::close() {
    if (!started_) return;
    {
        std::unique_lock<std::mutex> lock(st_->mu);
        st_->stop = true;
        st_->cvNotFull.notify_all();
        // Join the producers that are actually running; tasks the pool
        // never started exit as no-ops once the epoch moves on.
        st_->cvNotEmpty.wait(lock, [&] { return st_->activeProducers == 0; });
        st_->queue.clear();
        ++st_->epoch;
    }
    if (inline_ != nullptr) {
        inline_->close();
        inline_ = nullptr;
    }
    started_ = false;
}

} // namespace quantum
