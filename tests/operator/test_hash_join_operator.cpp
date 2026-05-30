#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/HashJoinOperator.hpp"
#include "operator/ScanOperator.hpp"

using namespace quantum;

namespace {

ColumnBatch kv(MemoryManager& mm,
               const std::vector<int64_t>& keys,
               const std::vector<int64_t>& vals,
               const std::vector<bool>& keyValidity = {}) {
    const size_t n = keys.size();
    auto k = ColumnVector::make(Int64, n, &mm);
    auto v = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        k.typed<int64_t>()[i] = keys[i];
        v.typed<int64_t>()[i] = vals[i];
    }
    if (!keyValidity.empty()) {
        for (size_t i = 0; i < n; ++i) k.validity().setValid(i, keyValidity[i]);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

OperatorPtr scan(MemoryManager& mm,
                 const Schema& schema,
                 std::vector<ColumnBatch> src) {
    return std::make_unique<ScanOperator>(&mm, schema, std::move(src));
}

// Drain a join into a set of (build_key, build_val, probe_key, probe_val)
// for easy comparison ignoring emission order.
using Row = std::tuple<int64_t, int64_t, int64_t, int64_t>;
std::multiset<Row> drainJoin(HashJoinOperator& op) {
    std::multiset<Row> out;
    op.open();
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        const auto* bk = b->column(0).typed<int64_t>();
        const auto* bv = b->column(1).typed<int64_t>();
        const auto* pk = b->column(2).typed<int64_t>();
        const auto* pv = b->column(3).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) {
            out.emplace(bk[i], bv[i], pk[i], pv[i]);
        }
    }
    op.close();
    return out;
}

OperatorPtr mkScan(MemoryManager& mm, std::vector<ColumnBatch> batches) {
    return scan(mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(batches));
}

} // namespace

TEST(HashJoin, EmptyBuildEmptyProbe) {
    MemoryManager mm(64 * 1024 * 1024);
    HashJoinOperator op(&mm, mkScan(mm, {}), mkScan(mm, {}),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});
    EXPECT_TRUE(drainJoin(op).empty());
}

TEST(HashJoin, EmptyBuildYieldsNoMatches) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 2, 3}, {10, 20, 30}));
    HashJoinOperator op(&mm, mkScan(mm, {}), mkScan(mm, std::move(probe)),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});
    EXPECT_TRUE(drainJoin(op).empty());
}

TEST(HashJoin, EmptyProbeYieldsNoMatches) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2, 3}, {10, 20, 30}));
    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, {}),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});
    EXPECT_TRUE(drainJoin(op).empty());
}

TEST(HashJoin, SimpleOneToOne) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2, 3}, {100, 200, 300}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {2, 3, 4}, {20, 30, 40}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});

    auto got = drainJoin(op);
    std::multiset<Row> want = {{2, 200, 2, 20}, {3, 300, 3, 30}};
    EXPECT_EQ(got, want);
}

TEST(HashJoin, MultiMatchOnSameKey) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 1, 2}, {101, 102, 200}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 2, 1}, {11, 22, 33}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});

    auto got = drainJoin(op);
    std::multiset<Row> want = {
        {1, 101, 1, 11}, {1, 102, 1, 11},
        {2, 200, 2, 22},
        {1, 101, 1, 33}, {1, 102, 1, 33},
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, NullKeysDoNotMatch) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 0, 2}, {100, 999, 200}, {true, false, true}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 0, 0, 2}, {10, 88, 99, 20}, {true, false, true, true}));
    //                                          ^ probe key NULL — never matches
    //                                              ^ probe key 0 not in build (build's 0 was NULL)

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});

    auto got = drainJoin(op);
    std::multiset<Row> want = {{1, 100, 1, 10}, {2, 200, 2, 20}};
    EXPECT_EQ(got, want);
}

TEST(HashJoin, NoMatchesAtAll) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2, 3}, {100, 200, 300}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {10, 20, 30}, {1, 2, 3}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});
    EXPECT_TRUE(drainJoin(op).empty());
}

TEST(HashJoin, BuildSpansMultipleBatches) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2},    {100, 200}));
    build.push_back(kv(mm, {3, 4, 5}, {300, 400, 500}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {5, 1, 4}, {55, 11, 44}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        std::make_unique<ColumnRef>(0),
                        std::make_unique<ColumnRef>(0),
                        {"bk", "bv", "pk", "pv"});

    auto got = drainJoin(op);
    std::multiset<Row> want = {
        {5, 500, 5, 55},
        {1, 100, 1, 11},
        {4, 400, 4, 44},
    };
    EXPECT_EQ(got, want);
}
