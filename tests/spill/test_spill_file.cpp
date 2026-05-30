#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <string_view>

#include "spill/SpillFile.hpp"

using namespace quantum;

TEST(SpillFile, RoundtripKeyedRecords) {
    constexpr size_t stateBytes = 16;
    SpillFile f(stateBytes, "test_keyed");

    uint8_t stateA[stateBytes];
    uint8_t stateB[stateBytes];
    std::memset(stateA, 0xAA, stateBytes);
    std::memset(stateB, 0x55, stateBytes);

    const std::string keyA = "alice";
    const std::string keyB = "bob";
    f.write(reinterpret_cast<const uint8_t*>(keyA.data()),
            static_cast<int32_t>(keyA.size()), stateA);
    f.write(reinterpret_cast<const uint8_t*>(keyB.data()),
            static_cast<int32_t>(keyB.size()), stateB);

    auto r = f.reader();
    ASSERT_TRUE(r.next());
    EXPECT_FALSE(r.isNullGroup());
    EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(r.key()), r.keyLen()),
              std::string_view{"alice"});
    EXPECT_EQ(std::memcmp(r.state(), stateA, stateBytes), 0);

    ASSERT_TRUE(r.next());
    EXPECT_FALSE(r.isNullGroup());
    EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(r.key()), r.keyLen()),
              std::string_view{"bob"});
    EXPECT_EQ(std::memcmp(r.state(), stateB, stateBytes), 0);

    EXPECT_FALSE(r.next());
}

TEST(SpillFile, NullGroupSentinel) {
    constexpr size_t stateBytes = 8;
    SpillFile f(stateBytes, "test_null");

    uint8_t state[stateBytes] = {1, 2, 3, 4, 5, 6, 7, 8};
    f.writeNullGroup(state);

    auto r = f.reader();
    ASSERT_TRUE(r.next());
    EXPECT_TRUE(r.isNullGroup());
    EXPECT_EQ(std::memcmp(r.state(), state, stateBytes), 0);
    EXPECT_FALSE(r.next());
}
