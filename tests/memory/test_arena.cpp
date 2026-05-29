#include <gtest/gtest.h>

#include <cstdint>

#include "memory/Arena.hpp"

using namespace quantum;

TEST(Arena, BumpWithinChunk) {
    Arena a(1024);
    auto* p = a.allocate(128);
    auto* q = a.allocate(128);
    ASSERT_NE(p, nullptr);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
    EXPECT_EQ(q, p + 128);
}

TEST(Arena, ChunkOverflowGrows) {
    Arena a(256);
    auto* p = a.allocate(200);
    auto* q = a.allocate(200);  // Forces a new chunk
    ASSERT_NE(p, nullptr);
    ASSERT_NE(q, nullptr);
    EXPECT_GE(a.bytesReserved(), 512u);
}

TEST(Arena, ResetReleasesEverything) {
    Arena a(256);
    a.allocate(200);
    a.allocate(200);
    a.reset();
    EXPECT_EQ(a.bytesReserved(),  0u);
    EXPECT_EQ(a.bytesAllocated(), 0u);
}

TEST(Arena, AlignmentInvariant) {
    Arena a(4096);
    for (size_t s : {1u, 7u, 17u, 33u, 65u, 1000u}) {
        auto* p = a.allocate(s);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u) << "s=" << s;
    }
}
