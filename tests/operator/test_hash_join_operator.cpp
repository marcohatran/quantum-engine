#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <tuple>
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

OperatorPtr mkScan(MemoryManager& mm, std::vector<ColumnBatch> batches) {
    return std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(batches));
}

std::vector<ExpressionPtr> keyOn(size_t idx) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(idx));
    return v;
}

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

struct OuterRow {
    bool    matched;
    int64_t bk, bv, pk, pv;
    bool operator<(const OuterRow& o) const {
        return std::tie(matched, bk, bv, pk, pv) < std::tie(o.matched, o.bk, o.bv, o.pk, o.pv);
    }
    bool operator==(const OuterRow& o) const {
        return std::tie(matched, bk, bv, pk, pv) == std::tie(o.matched, o.bk, o.bv, o.pk, o.pv);
    }
};

std::multiset<OuterRow> drainOuter(HashJoinOperator& op) {
    std::multiset<OuterRow> out;
    op.open();
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        const auto* bk = b->column(0).typed<int64_t>();
        const auto* bv = b->column(1).typed<int64_t>();
        const auto* pk = b->column(2).typed<int64_t>();
        const auto* pv = b->column(3).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) {
            const bool matched = b->column(0).validity().isValid(i);
            out.insert({matched,
                        matched ? bk[i] : 0,
                        matched ? bv[i] : 0,
                        pk[i],
                        pv[i]});
        }
    }
    op.close();
    return out;
}

} // namespace

TEST(HashJoin, EmptyBuildEmptyProbe) {
    MemoryManager mm(64 * 1024 * 1024);
    HashJoinOperator op(&mm, mkScan(mm, {}), mkScan(mm, {}),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});
    EXPECT_TRUE(drainJoin(op).empty());
}

TEST(HashJoin, EmptyBuildYieldsNoMatches) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 2, 3}, {10, 20, 30}));
    HashJoinOperator op(&mm, mkScan(mm, {}), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});
    EXPECT_TRUE(drainJoin(op).empty());
}

TEST(HashJoin, EmptyProbeYieldsNoMatches) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2, 3}, {10, 20, 30}));
    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, {}),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});
    EXPECT_TRUE(drainJoin(op).empty());
}

TEST(HashJoin, SimpleOneToOne) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2, 3}, {100, 200, 300}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {2, 3, 4}, {20, 30, 40}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});

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
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});

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

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});

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
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});
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
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"});

    auto got = drainJoin(op);
    std::multiset<Row> want = {
        {5, 500, 5, 55},
        {1, 100, 1, 11},
        {4, 400, 4, 44},
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, LeftOuterEmitsUnmatchedProbeRows) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2}, {100, 200}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 3, 2, 4}, {11, 33, 22, 44}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::LeftOuter);

    auto got = drainOuter(op);
    std::multiset<OuterRow> want = {
        {true,  1, 100, 1, 11},
        {false, 0, 0,   3, 33},
        {true,  2, 200, 2, 22},
        {false, 0, 0,   4, 44},
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, LeftOuterWithEmptyBuild) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 2}, {10, 20}));

    HashJoinOperator op(&mm, mkScan(mm, {}), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::LeftOuter);

    auto got = drainOuter(op);
    std::multiset<OuterRow> want = {
        {false, 0, 0, 1, 10},
        {false, 0, 0, 2, 20},
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, LeftOuterWithNullProbeKey) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1}, {100}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 99}, {11, 22}, {true, false}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::LeftOuter);

    auto got = drainOuter(op);
    std::multiset<OuterRow> want = {
        {true,  1, 100, 1,  11},
        {false, 0, 0,   99, 22},
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, TwoColumnKey) {
    MemoryManager mm(64 * 1024 * 1024);

    auto threeColBatch = [&](const std::vector<int64_t>& k1,
                             const std::vector<int64_t>& k2,
                             const std::vector<int64_t>& v) {
        const size_t n = k1.size();
        auto c1 = ColumnVector::make(Int64, n, &mm);
        auto c2 = ColumnVector::make(Int64, n, &mm);
        auto cv = ColumnVector::make(Int64, n, &mm);
        for (size_t i = 0; i < n; ++i) {
            c1.typed<int64_t>()[i] = k1[i];
            c2.typed<int64_t>()[i] = k2[i];
            cv.typed<int64_t>()[i] = v[i];
        }
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(c1));
        cols.push_back(std::move(c2));
        cols.push_back(std::move(cv));
        return ColumnBatch(std::move(cols), PositionList::identity(n));
    };

    std::vector<ColumnBatch> build;
    build.push_back(threeColBatch({1, 1, 2}, {10, 20, 10}, {100, 200, 300}));
    std::vector<ColumnBatch> probe;
    probe.push_back(threeColBatch({2, 1, 1, 3}, {10, 20, 10, 10}, {3000, 2000, 1000, 4000}));

    auto bScan = std::make_unique<ScanOperator>(
        &mm, Schema({{"bk1", Int64}, {"bk2", Int64}, {"bv", Int64}}), std::move(build));
    auto pScan = std::make_unique<ScanOperator>(
        &mm, Schema({{"pk1", Int64}, {"pk2", Int64}, {"pv", Int64}}), std::move(probe));

    std::vector<ExpressionPtr> bk;
    bk.push_back(std::make_unique<ColumnRef>(0));
    bk.push_back(std::make_unique<ColumnRef>(1));
    std::vector<ExpressionPtr> pk;
    pk.push_back(std::make_unique<ColumnRef>(0));
    pk.push_back(std::make_unique<ColumnRef>(1));

    HashJoinOperator op(&mm, std::move(bScan), std::move(pScan),
                        std::move(bk), std::move(pk),
                        {"bk1", "bk2", "bv", "pk1", "pk2", "pv"});

    std::multiset<std::tuple<int64_t,int64_t,int64_t,int64_t,int64_t,int64_t>> got;
    op.open();
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            got.emplace(
                b->column(0).typed<int64_t>()[i],
                b->column(1).typed<int64_t>()[i],
                b->column(2).typed<int64_t>()[i],
                b->column(3).typed<int64_t>()[i],
                b->column(4).typed<int64_t>()[i],
                b->column(5).typed<int64_t>()[i]);
        }
    }
    op.close();

    decltype(got) want = {
        {2, 10, 300, 2, 10, 3000},
        {1, 20, 200, 1, 20, 2000},
        {1, 10, 100, 1, 10, 1000},
    };
    EXPECT_EQ(got, want);
}
