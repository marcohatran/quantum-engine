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
               const std::vector<bool>&    keyValidity = {}) {
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

// (matched-build, build_k, build_v, matched-probe, probe_k, probe_v)
struct OuterRow {
    bool    bMatched, pMatched;
    int64_t bk, bv, pk, pv;
    bool operator<(const OuterRow& o) const {
        return std::tie(bMatched, pMatched, bk, bv, pk, pv) <
               std::tie(o.bMatched, o.pMatched, o.bk, o.bv, o.pk, o.pv);
    }
    bool operator==(const OuterRow& o) const {
        return std::tie(bMatched, pMatched, bk, bv, pk, pv) ==
               std::tie(o.bMatched, o.pMatched, o.bk, o.bv, o.pk, o.pv);
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
            const bool bM = b->column(0).validity().isValid(i);
            const bool pM = b->column(2).validity().isValid(i);
            out.insert({bM, pM,
                        bM ? bk[i] : 0, bM ? bv[i] : 0,
                        pM ? pk[i] : 0, pM ? pv[i] : 0});
        }
    }
    op.close();
    return out;
}

} // namespace

TEST(HashJoin, RightOuterEmitsUnmatchedBuildRows) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2, 3}, {100, 200, 300}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {2, 4}, {22, 44}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::RightOuter);

    auto got = drainOuter(op);
    std::multiset<OuterRow> want = {
        {true,  true,  2, 200, 2, 22},
        {true,  false, 1, 100, 0, 0},
        {true,  false, 3, 300, 0, 0},
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, FullOuterEmitsBothMissTypes) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 2}, {100, 200}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {2, 3}, {22, 33}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::FullOuter);

    auto got = drainOuter(op);
    std::multiset<OuterRow> want = {
        {true,  true,  2, 200, 2, 22},
        {true,  false, 1, 100, 0, 0},      // build-only
        {false, true,  0, 0,   3, 33},     // probe-only
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, RightOuterWithEmptyProbeEmitsAllBuildRows) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {7, 9}, {77, 99}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, {}),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::RightOuter);

    auto got = drainOuter(op);
    std::multiset<OuterRow> want = {
        {true, false, 7, 77, 0, 0},
        {true, false, 9, 99, 0, 0},
    };
    EXPECT_EQ(got, want);
}

TEST(HashJoin, RightOuterWithEmptyBuildEmitsNothing) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1}, {11}));

    HashJoinOperator op(&mm, mkScan(mm, {}), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::RightOuter);

    auto got = drainOuter(op);
    EXPECT_TRUE(got.empty());
}

TEST(HashJoin, FullOuterWithBothEmptyEmitsNothing) {
    MemoryManager mm(64 * 1024 * 1024);
    HashJoinOperator op(&mm, mkScan(mm, {}), mkScan(mm, {}),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::FullOuter);
    auto got = drainOuter(op);
    EXPECT_TRUE(got.empty());
}

TEST(HashJoin, RightOuterIncludesNullBuildKeysInTail) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    // The "0" build row has a NULL key. It can never match a probe, so RIGHT
    // OUTER must still emit it in the tail with NULL probe cols.
    build.push_back(kv(mm, {1, 0, 2}, {100, 999, 200}, {true, false, true}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {2}, {22}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::RightOuter);

    op.open();
    bool sawNullBuildRow = false;
    int  totalRows       = 0;
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            ++totalRows;
            const bool bkValid = b->column(0).validity().isValid(i);
            const bool pkValid = b->column(2).validity().isValid(i);
            // The NULL-keyed build row appears in the tail: bk column validity
            // reflects the original (false), and probe cols are NULL.
            if (!bkValid) {
                EXPECT_FALSE(pkValid);
                sawNullBuildRow = true;
            }
        }
    }
    op.close();
    EXPECT_TRUE(sawNullBuildRow);
    EXPECT_EQ(totalRows, 3);   // (k=1)=>NULL probe, (k=NULL)=>NULL probe, (k=2,p=2)=match
}

TEST(HashJoin, RightOuterDuplicateBuildKey) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> build;
    build.push_back(kv(mm, {1, 1, 2}, {101, 102, 200}));
    std::vector<ColumnBatch> probe;
    probe.push_back(kv(mm, {1, 9}, {11, 99}));

    HashJoinOperator op(&mm, mkScan(mm, std::move(build)), mkScan(mm, std::move(probe)),
                        keyOn(0), keyOn(0), {"bk", "bv", "pk", "pv"},
                        JoinType::RightOuter);

    auto got = drainOuter(op);
    // Probe key 1 matches both build rows (101 and 102). Build key 2 has no
    // match in probe and should appear in the tail. Probe key 9 has no match
    // in build and is dropped (RIGHT OUTER, not FULL OUTER).
    std::multiset<OuterRow> want = {
        {true,  true,  1, 101, 1, 11},
        {true,  true,  1, 102, 1, 11},
        {true,  false, 2, 200, 0, 0},
    };
    EXPECT_EQ(got, want);
}
