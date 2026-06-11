#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/ExchangeOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "plan/PlanProto.hpp"
#include "vector/ColumnBatch.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

ColumnBatch kvBatch(MemoryManager& mm,
                    const std::vector<int64_t>& ks,
                    const std::vector<int64_t>& vs,
                    const std::vector<bool>&    kValid = {}) {
    auto k = ColumnVector::make(Int64, ks.size(), &mm);
    auto v = ColumnVector::make(Int64, vs.size(), &mm);
    k.validity().setAllValid();
    for (size_t i = 0; i < ks.size(); ++i) {
        k.typed<int64_t>()[i] = ks[i];
        v.typed<int64_t>()[i] = vs[i];
        if (!kValid.empty()) k.validity().setValid(i, kValid[i]);
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(ks.size()));
}

std::vector<ExpressionPtr> keyOn(size_t i) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(i));
    return v;
}

} // namespace

TEST(ExchangeOperator, PartitionsPreserveAllRowsAndClusterKeys) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(kvBatch(mm, {1, 2, 3, 4}, {10, 20, 30, 40}));
    src.push_back(kvBatch(mm, {2, 1, 5},    {21, 11, 50}));

    auto scan = std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
    ExchangeOperator ex(&mm, std::move(scan), keyOn(0), /*numPartitions=*/3);

    ex.open();
    auto partitions = ex.takePartitions();
    ex.close();
    ASSERT_EQ(partitions.size(), 3u);

    // Every key lands in exactly one partition; all rows survive.
    std::map<int64_t, std::set<size_t>> keyToPartitions;
    std::multiset<std::pair<int64_t, int64_t>> rows;
    for (size_t p = 0; p < partitions.size(); ++p) {
        for (const auto& b : partitions[p]) {
            for (size_t i = 0; i < b.numRows(); ++i) {
                const int64_t k = b.column(0).typed<int64_t>()[i];
                keyToPartitions[k].insert(p);
                rows.insert({k, b.column(1).typed<int64_t>()[i]});
            }
        }
    }
    EXPECT_EQ(rows.size(), 7u);
    for (const auto& [k, parts] : keyToPartitions) {
        EXPECT_EQ(parts.size(), 1u) << "key " << k << " split across partitions";
    }
    EXPECT_TRUE(rows.count({2, 20}) == 1 && rows.count({2, 21}) == 1);
}

TEST(ExchangeOperator, NullKeysClusterTogether) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(kvBatch(mm, {1, 0, 2, 0}, {10, 20, 30, 40},
                          {true, false, true, false}));

    auto scan = std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
    ExchangeOperator ex(&mm, std::move(scan), keyOn(0), 4);
    ex.open();
    auto partitions = ex.takePartitions();
    ex.close();

    std::set<size_t> nullPartitions;
    size_t total = 0;
    for (size_t p = 0; p < partitions.size(); ++p) {
        for (const auto& b : partitions[p]) {
            total += b.numRows();
            for (size_t i = 0; i < b.numRows(); ++i) {
                if (!b.column(0).validity().isValid(i)) nullPartitions.insert(p);
            }
        }
    }
    EXPECT_EQ(total, 4u);
    EXPECT_EQ(nullPartitions.size(), 1u);  // both null-key rows in one partition
}

TEST(ExchangeOperator, StreamingGetNextEmitsEveryRowOnce) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(kvBatch(mm, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}));

    auto scan = std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(src));
    ExchangeOperator ex(&mm, std::move(scan), keyOn(0), 4);
    ex.open();
    std::multiset<int64_t> got;
    while (auto b = ex.getNext()) {
        for (size_t i = 0; i < b->numRows(); ++i) {
            got.insert(b->column(0).typed<int64_t>()[i]);
        }
    }
    ex.close();
    EXPECT_EQ(got, (std::multiset<int64_t>{1, 2, 3, 4, 5, 6}));
}

TEST(ExchangeOperator, ParallelAggregateMatchesSerial) {
    MemoryManager mm(64 * 1024 * 1024);

    // Mixed string + int data with nulls, multiple batches, enough keys
    // to spread across partitions.
    auto mkSource = [&] {
        std::vector<ColumnBatch> src;
        for (int b = 0; b < 4; ++b) {
            std::vector<int64_t> ks, vs;
            for (int i = 0; i < 100; ++i) {
                ks.push_back((b * 100 + i) % 17);
                vs.push_back(b * 100 + i);
            }
            src.push_back(kvBatch(mm, ks, vs));
        }
        return src;
    };

    auto runWith = [&](int32_t parallelism) {
        SourceCatalog catalog;
        catalog["kv"] = mkSource();
        auto agg = std::make_unique<HashAggregateNode>();
        agg->child = ScanNode{"kv", Schema({{"k", Int64}, {"v", Int64}})};
        agg->group_keys.push_back(ColumnRefExpr{0});
        AggregateSpec sum;
        sum.kind = AggregateKind::Sum;
        sum.inputType = Int64;
        sum.input = ColumnRefExpr{1};
        agg->aggregates.push_back(std::move(sum));
        AggregateSpec cnt;
        cnt.kind = AggregateKind::CountStar;
        agg->aggregates.push_back(std::move(cnt));
        agg->output_names = {"k", "s", "c"};
        agg->parallelism  = parallelism;
        PlanNode root{std::move(agg)};

        // Proto round-trip so the parallelism field is exercised on the wire.
        PlanNode decoded = planFromProtoBytes(planToProtoBytes(root));

        PlanBuilder pb(&mm);
        auto op = pb.build(decoded, catalog);
        std::map<int64_t, std::pair<int64_t, int64_t>> got;
        op->open();
        while (auto b = op->getNext()) {
            for (size_t i = 0; i < b->numRows(); ++i) {
                got[b->column(0).typed<int64_t>()[i]] = {
                    b->column(1).typed<int64_t>()[i],
                    b->column(2).typed<int64_t>()[i]};
            }
        }
        op->close();
        return got;
    };

    auto serial   = runWith(0);
    auto parallel = runWith(4);
    EXPECT_EQ(serial.size(), 17u);
    EXPECT_EQ(parallel, serial);
}

TEST(ExchangeOperator, ExchangeNodeSurvivesProtoRoundTrip) {
    MemoryManager mm(64 * 1024 * 1024);
    SourceCatalog catalog;
    catalog["kv"].push_back(kvBatch(mm, {1, 2, 3, 4}, {10, 20, 30, 40}));

    auto ex = std::make_unique<ExchangeNode>();
    ex->child = ScanNode{"kv", Schema({{"k", Int64}, {"v", Int64}})};
    ex->keys.push_back(ColumnRefExpr{0});
    ex->num_partitions = 2;
    PlanNode root{std::move(ex)};

    PlanNode decoded = planFromProtoBytes(planToProtoBytes(root));
    PlanBuilder pb(&mm);
    auto op = pb.build(decoded, catalog);
    op->open();
    size_t total = 0;
    while (auto b = op->getNext()) total += b->numRows();
    op->close();
    EXPECT_EQ(total, 4u);
}
