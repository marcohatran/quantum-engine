#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

#include "memory/Buffer.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

TEST(Buffer, AlignedAndOwning) {
    MemoryManager mm(1024);
    Buffer b = mm.allocate(128);
    ASSERT_NE(b.data(), nullptr);
    EXPECT_EQ(b.size(), 128u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b.data()) % 64, 0u);
}

TEST(Buffer, MoveTransfersOwnership) {
    MemoryManager mm(1024);
    Buffer a = mm.allocate(128);
    EXPECT_EQ(mm.used(), 128u);
    Buffer b = std::move(a);
    EXPECT_EQ(mm.used(), 128u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(b.size(), 128u);
}

TEST(Buffer, ZeroSize) {
    MemoryManager mm(1024);
    Buffer b = mm.allocate(0);
    EXPECT_EQ(b.size(), 0u);
    EXPECT_EQ(mm.used(), 0u);
}
