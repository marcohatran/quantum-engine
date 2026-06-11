#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "common/Macros.hpp"

namespace quantum {

// Fixed-size worker pool for intra-operator parallelism (Phase 6). The
// engine's operators stay single-threaded; the scheduler runs coarse,
// independent units — per-file Parquet decode today, exchange partitions
// later. Work submitted here must only touch thread-safe components
// (MemoryManager is; everything else needs external coordination).
class TaskScheduler {
public:
    explicit TaskScheduler(size_t threads);
    ~TaskScheduler();

    QUANTUM_DISALLOW_COPY(TaskScheduler);

    size_t threads() const noexcept { return threads_; }

    // Fire-and-forget; the task must not throw (wrap with parallelFor for
    // exception propagation).
    void submit(std::function<void()> fn);

    // Runs fn(0..n-1) across the pool, the calling thread included, and
    // returns when every index has finished. The first exception thrown
    // by any fn is rethrown on the caller. Not reentrant — do not call
    // parallelFor from inside a parallelFor task.
    void parallelFor(size_t n, const std::function<void(size_t)>& fn);

    // Process-global pool. Sized by QUANTUM_SCHEDULER_THREADS, else half
    // the hardware threads (min 1). Spark supplies task-level parallelism
    // around the engine, so the default deliberately undersubscribes.
    static TaskScheduler& global();

private:
    void workerLoop();

    size_t                            threads_;
    std::vector<std::thread>          workers_;
    std::deque<std::function<void()>> queue_;
    std::mutex                        mu_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
};

} // namespace quantum
