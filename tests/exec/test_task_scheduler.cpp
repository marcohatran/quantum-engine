#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "exec/TaskScheduler.hpp"

using namespace quantum;

TEST(TaskScheduler, ParallelForCoversEveryIndexExactlyOnce) {
    TaskScheduler sched(4);
    constexpr size_t kN = 1000;
    std::vector<std::atomic<int>> hits(kN);
    sched.parallelFor(kN, [&](size_t i) {
        hits[i].fetch_add(1, std::memory_order_relaxed);
    });
    for (size_t i = 0; i < kN; ++i) {
        EXPECT_EQ(hits[i].load(), 1) << "index " << i;
    }
}

TEST(TaskScheduler, ParallelForActuallyUsesWorkers) {
    TaskScheduler sched(4);
    std::atomic<int> concurrent{0};
    std::atomic<int> peak{0};
    sched.parallelFor(64, [&](size_t) {
        const int now = concurrent.fetch_add(1) + 1;
        int p = peak.load();
        while (now > p && !peak.compare_exchange_weak(p, now)) {}
        // Hold briefly so overlap is observable.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        concurrent.fetch_sub(1);
    });
    EXPECT_GE(peak.load(), 2) << "expected overlapping execution";
}

TEST(TaskScheduler, ParallelForPropagatesFirstException) {
    TaskScheduler sched(2);
    EXPECT_THROW(
        sched.parallelFor(16, [&](size_t i) {
            if (i == 7) throw std::runtime_error("boom");
        }),
        std::runtime_error);
}

TEST(TaskScheduler, EmptyAndSingleAreInline) {
    TaskScheduler sched(2);
    sched.parallelFor(0, [&](size_t) { FAIL() << "must not run"; });
    std::atomic<int> count{0};
    sched.parallelFor(1, [&](size_t i) {
        EXPECT_EQ(i, 0u);
        count.fetch_add(1);
    });
    EXPECT_EQ(count.load(), 1);
}

TEST(TaskScheduler, SubmitRunsDetachedWork) {
    TaskScheduler sched(2);
    std::atomic<int> ran{0};
    std::mutex mu;
    std::condition_variable cv;
    for (int i = 0; i < 8; ++i) {
        sched.submit([&] {
            if (ran.fetch_add(1) + 1 == 8) {
                std::lock_guard<std::mutex> lock(mu);
                cv.notify_all();
            }
        });
    }
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return ran.load() == 8; });
    EXPECT_EQ(ran.load(), 8);
}
