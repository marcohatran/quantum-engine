#include <gtest/gtest.h>

#include <cstring>
#include <set>

#include "hash/HashTable.hpp"
#include "memory/MemoryManager.hpp"

using namespace quantum;

namespace {
void initInt64Zero(uint8_t* p) {
    int64_t z = 0;
    std::memcpy(p, &z, sizeof(z));
}
} // namespace

TEST(HashTable, InsertAndCount) {
    MemoryManager mm(1U << 20);
    HashTable t(sizeof(int64_t), &mm);
    for (int64_t k = 0; k < 100; ++k) {
        auto* s = t.findOrInsert(k, initInt64Zero);
        int64_t v;
        std::memcpy(&v, s, sizeof(v));
        ++v;
        std::memcpy(s, &v, sizeof(v));
    }
    EXPECT_EQ(t.numGroups(), 100u);
}

TEST(HashTable, RepeatedKeysReturnSameSlot) {
    MemoryManager mm(1U << 20);
    HashTable t(sizeof(int64_t), &mm);
    auto* a = t.findOrInsert(42, initInt64Zero);
    auto* b = t.findOrInsert(42, initInt64Zero);
    EXPECT_EQ(a, b);
    EXPECT_EQ(t.numGroups(), 1u);
}

TEST(HashTable, NullGroupSlotIsDistinct) {
    MemoryManager mm(1U << 20);
    HashTable t(sizeof(int64_t), &mm);
    auto* keyed = t.findOrInsert(0, initInt64Zero);
    auto* nul   = t.findOrInsertNull(initInt64Zero);
    EXPECT_NE(keyed, nul);
    EXPECT_TRUE(t.hasNullGroup());
    EXPECT_EQ(t.numGroups(), 1u);  // null group does not count
}

TEST(HashTable, GrowsAndPreservesContents) {
    MemoryManager mm(8U << 20);
    HashTable t(sizeof(int64_t), &mm, /*initialCapacity=*/16);

    for (int64_t k = 0; k < 5000; ++k) {
        auto* s = t.findOrInsert(k, initInt64Zero);
        int64_t v = k * 2;
        std::memcpy(s, &v, sizeof(v));
    }
    ASSERT_EQ(t.numGroups(), 5000u);

    std::set<int64_t> seen;
    for (auto cur = t.cursor(); cur.valid(); cur.next()) {
        int64_t v;
        std::memcpy(&v, cur.state(), sizeof(v));
        EXPECT_EQ(v, cur.key() * 2);
        seen.insert(cur.key());
    }
    EXPECT_EQ(seen.size(), 5000u);
    EXPECT_EQ(*seen.begin(),  0);
    EXPECT_EQ(*seen.rbegin(), 4999);
}
