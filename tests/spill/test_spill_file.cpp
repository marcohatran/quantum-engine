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

TEST(SpillFile, VariableLengthStateRecords) {
    SpillFile f(/*stateBytes=*/4, "test_varlen");

    const std::string keyA = "k1";
    const std::vector<uint8_t> stateA = {1, 2, 3};                  // shorter
    const std::vector<uint8_t> stateB = {9, 8, 7, 6, 5, 4, 3, 2};   // longer

    f.write(reinterpret_cast<const uint8_t*>(keyA.data()),
            static_cast<int32_t>(keyA.size()), stateA.data(), stateA.size());
    f.writeNullGroup(stateB.data(), stateB.size());
    f.finishWriting();

    auto r = f.reader();
    ASSERT_TRUE(r.next());
    EXPECT_EQ(r.stateLen(), stateA.size());
    EXPECT_EQ(std::memcmp(r.state(), stateA.data(), stateA.size()), 0);
    ASSERT_TRUE(r.next());
    EXPECT_TRUE(r.isNullGroup());
    EXPECT_EQ(r.stateLen(), stateB.size());
    EXPECT_EQ(std::memcmp(r.state(), stateB.data(), stateB.size()), 0);
    EXPECT_FALSE(r.next());
}
