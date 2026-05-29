#include <gtest/gtest.h>

#include <future>
#include <vector>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "memory/Buffer.hpp"

using namespace quantum;

TEST(MemoryManager, AllocateAndReleaseTracksUsage) {
    MemoryManager mm(1024);
    {
        Buffer b = mm.allocate(256);
        EXPECT_EQ(mm.used(), 256u);
        EXPECT_EQ(b.size(),  256u);
    }
    EXPECT_EQ(mm.used(), 0u);
}

TEST(MemoryManager, OOMThrows) {
    MemoryManager mm(512);
    EXPECT_THROW(mm.allocate(1024), QuantumOOM);
    EXPECT_EQ(mm.used(), 0u);
}

TEST(MemoryManager, ReserveHonorsBudget) {
    MemoryManager mm(512);
    EXPECT_TRUE (mm.reserve(256));
    EXPECT_TRUE (mm.reserve(256));
    EXPECT_FALSE(mm.reserve(1));
}

TEST(MemoryManager, ThreadSafeReserve) {
    MemoryManager mm(1024 * 8);
    std::vector<std::future<size_t>> tasks;
    for (int i = 0; i < 8; ++i) {
        tasks.push_back(std::async(std::launch::async, [&] {
            size_t hits = 0;
            for (int j = 0; j < 1000; ++j) {
                if (mm.reserve(8)) ++hits;
            }
            return hits;
        }));
    }
    size_t total = 0;
    for (auto& t : tasks) total += t.get();
    EXPECT_EQ(total * 8, mm.used());
    EXPECT_LE(mm.used(), 1024u * 8u);
}
