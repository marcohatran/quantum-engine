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

// --- Buffer recycling (R1) ---

TEST(MemoryManager, FreedBuffersAreRecycledBySize) {
    MemoryManager mm(1 << 20);
    const uint8_t* first = nullptr;
    {
        Buffer b = mm.allocate(4096);
        first = b.data();
        ASSERT_NE(first, nullptr);
    }
    EXPECT_EQ(mm.used(), 0u);

    // Same size comes back from the free list — identical pointer.
    Buffer reused = mm.allocate(4096);
    EXPECT_EQ(reused.data(), first);
    EXPECT_EQ(mm.used(), 4096u);

    // A different size must not alias the cached 4096 block.
    Buffer other = mm.allocate(8192);
    EXPECT_NE(other.data(), first);
}

TEST(MemoryManager, RecyclingBoundsArenaGrowthUnderChurn) {
    // Steady-state churn must NOT exhaust the budget-sized arena: 10k
    // alloc/free rounds of 64KB through a 1MB budget only works if blocks
    // recycle (bump allocation alone would need ~640MB).
    MemoryManager mm(1 << 20);
    const uint8_t* steady = nullptr;
    for (int i = 0; i < 10000; ++i) {
        Buffer b = mm.allocate(64 * 1024);
        if (i == 1) steady = b.data();
        if (i > 1) EXPECT_EQ(b.data(), steady) << "round " << i;
    }
    EXPECT_EQ(mm.used(), 0u);
}

TEST(MemoryManager, BudgetAccountingUnchangedByRecycling) {
    MemoryManager mm(8192);
    Buffer a = mm.allocate(4096);
    EXPECT_EQ(mm.used(), 4096u);
    {
        Buffer b = mm.allocate(4096);
        EXPECT_EQ(mm.used(), 8192u);
        EXPECT_THROW(mm.allocate(64), QuantumError);  // budget full
    }
    EXPECT_EQ(mm.used(), 4096u);   // b released
    Buffer c = mm.allocate(4096);  // recycled block re-counts
    EXPECT_EQ(mm.used(), 8192u);
}

TEST(MemoryManager, TrimReleasesArenaWhenIdle) {
    MemoryManager mm(16 << 20);
    {
        auto a = mm.allocate(1 << 20);
        auto b = mm.allocate(2 << 20);
        EXPECT_FALSE(mm.trim());  // live buffers — must refuse
        EXPECT_GT(mm.arenaReservedBytes(), 0u);
    }
    // Buffers destroyed: bytes sit in the free lists, arena still mapped.
    EXPECT_EQ(mm.used(), 0u);
    EXPECT_GT(mm.arenaReservedBytes(), 0u);

    EXPECT_TRUE(mm.trim());
    EXPECT_EQ(mm.arenaReservedBytes(), 0u);

    // Engine remains usable after a trim, and recycling still works.
    auto c = mm.allocate(1 << 20);
    EXPECT_GT(mm.arenaReservedBytes(), 0u);
    EXPECT_EQ(mm.used(), 1u << 20);
}
