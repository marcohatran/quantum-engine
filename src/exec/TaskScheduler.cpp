#include "exec/TaskScheduler.hpp"

#include <atomic>
#include <cstdlib>
#include <exception>
#include <utility>

namespace quantum {

TaskScheduler::TaskScheduler(size_t threads)
    : threads_(threads == 0 ? 1 : threads) {
    workers_.reserve(threads_);
    for (size_t i = 0; i < threads_; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

TaskScheduler::~TaskScheduler() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void TaskScheduler::submit(std::function<void()> fn) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.push_back(std::move(fn));
    }
    cv_.notify_one();
}

void TaskScheduler::workerLoop() {
    while (true) {
        std::function<void()> fn;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) return;
            fn = std::move(queue_.front());
            queue_.pop_front();
        }
        fn();
    }
}

void TaskScheduler::parallelFor(size_t n, const std::function<void(size_t)>& fn) {
    if (n == 0) return;
    if (n == 1) { fn(0); return; }

    struct State {
        std::atomic<size_t>     next{0};
        std::atomic<size_t>     done{0};
        std::mutex              mu;
        std::condition_variable cv;
        std::exception_ptr      error;
    };
    auto state = std::make_shared<State>();

    auto drain = [state, n, &fn] {
        while (true) {
            const size_t i = state->next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(state->mu);
                if (!state->error) state->error = std::current_exception();
            }
            if (state->done.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
                std::lock_guard<std::mutex> lock(state->mu);
                state->cv.notify_all();
            }
        }
    };

    // Helpers from the pool plus the calling thread. `fn` outlives the
    // helpers because we block below until every index completed.
    const size_t helpers = std::min(threads_, n - 1);
    for (size_t i = 0; i < helpers; ++i) submit(drain);
    drain();

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->done.load() == n; });
    if (state->error) std::rethrow_exception(state->error);
}

TaskScheduler& TaskScheduler::global() {
    static TaskScheduler scheduler([] {
        if (const char* env = std::getenv("QUANTUM_SCHEDULER_THREADS")) {
            const long v = std::strtol(env, nullptr, 10);
            if (v > 0) return static_cast<size_t>(v);
        }
        const unsigned hw = std::thread::hardware_concurrency();
        return static_cast<size_t>(hw > 1 ? hw / 2 : 1);
    }());
    return scheduler;
}

} // namespace quantum
