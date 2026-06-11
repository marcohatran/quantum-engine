#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/Metrics.hpp"
#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "vector/ColumnBatch.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

ColumnBatch intIntBatch(MemoryManager& mm,
                        const std::vector<int64_t>& ks,
                        const std::vector<int64_t>& vs) {
    auto k = ColumnVector::make(Int64, ks.size(), &mm);
    auto v = ColumnVector::make(Int64, vs.size(), &mm);
    for (size_t i = 0; i < ks.size(); ++i) {
        k.typed<int64_t>()[i] = ks[i];
        v.typed<int64_t>()[i] = vs[i];
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    cols.push_back(std::move(v));
    return ColumnBatch(std::move(cols), PositionList::identity(ks.size()));
}

ScalarExpr col(int32_t i) { return ColumnRefExpr{i}; }

} // namespace

TEST(Metrics, PlanBuilderWrapsOperatorsAndCounts) {
    MemoryManager mm(64 * 1024 * 1024);
    SourceCatalog catalog;
    catalog["kv"].push_back(intIntBatch(mm, {1, 2, 1, 3}, {10, 20, 30, 40}));

    // SELECT k, SUM(v) FROM kv GROUP BY k — Scan under HashAggregate.
    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = ScanNode{"kv", Schema({{"k", Int64}, {"v", Int64}})};
    agg->group_keys.push_back(col(0));
    AggregateSpec sum;
    sum.kind = AggregateKind::Sum;
    sum.inputType = Int64;
    sum.input = col(1);
    agg->aggregates.push_back(std::move(sum));
    agg->output_names = {"k", "sum_v"};
    PlanNode root{std::move(agg)};

    QueryMetrics qm;
    PlanBuilder pb(&mm, &qm);
    auto op = pb.build(root, catalog);

    op->open();
    size_t rowsOut = 0;
    while (auto b = op->getNext()) rowsOut += b->activeRows();
    op->close();

    // Build order is bottom-up: Scan first, HashAggregate (root) last.
    ASSERT_EQ(qm.operators().size(), 2u);
    EXPECT_EQ(qm.operators()[0]->name, "ScanOperator");
    EXPECT_EQ(qm.operators()[1]->name, "HashAggregateOperator");
    EXPECT_EQ(qm.root(), qm.operators()[1].get());

    EXPECT_EQ(qm.operators()[0]->rows, 4u);      // scan emits all input rows
    EXPECT_EQ(qm.root()->rows, rowsOut);         // 3 groups
    EXPECT_EQ(qm.root()->rows, 3u);
    EXPECT_GE(qm.operators()[0]->batches, 1u);

    // Trace JSON parses and carries the operator names.
    auto j = nlohmann::json::parse(qm.toJson());
    ASSERT_TRUE(j.contains("operators"));
    ASSERT_EQ(j["operators"].size(), 2u);
    EXPECT_EQ(j["operators"][1]["operator"], "HashAggregateOperator");
    EXPECT_EQ(j["operators"][1]["rows"], 3);
}

TEST(Metrics, NoSinkMeansNoWrapping) {
    MemoryManager mm(64 * 1024 * 1024);
    SourceCatalog catalog;
    catalog["kv"].push_back(intIntBatch(mm, {1}, {10}));

    PlanNode root{ScanNode{"kv", Schema({{"k", Int64}, {"v", Int64}})}};
    PlanBuilder pb(&mm);  // no metrics sink
    auto op = pb.build(root, catalog);
    op->open();
    auto b = op->getNext();
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(b->numRows(), 1u);
    op->close();
}

TEST(Metrics, PrometheusTextExposesCumulativeCounters) {
    metrics::resetForTesting();
    metrics::recordQuery(/*rowsOut=*/42, /*execNanos=*/1000);
    metrics::recordSpill();
    metrics::recordSpill();

    const std::string text = metrics::prometheusText();
    EXPECT_NE(text.find("# TYPE quantum_queries_total counter"), std::string::npos);
    EXPECT_NE(text.find("quantum_queries_total 1"),      std::string::npos);
    EXPECT_NE(text.find("quantum_rows_emitted_total 42"), std::string::npos);
    EXPECT_NE(text.find("quantum_spills_total 2"),        std::string::npos);
    EXPECT_NE(text.find("quantum_exec_nanos_total 1000"), std::string::npos);
    metrics::resetForTesting();
}

TEST(Metrics, SpillSitesFeedTheCounter) {
    metrics::resetForTesting();

    MemoryManager mm(64 * 1024 * 1024);
    SourceCatalog catalog;
    catalog["kv"].push_back(intIntBatch(mm, {1, 2}, {10, 20}));
    catalog["kv"].push_back(intIntBatch(mm, {3, 4}, {30, 40}));

    // HashAggregate built directly with an aggressive spill trigger.
    auto scanBatches = std::move(catalog["kv"]);
    auto scan = std::make_unique<ScanOperator>(
        &mm, Schema({{"k", Int64}, {"v", Int64}}), std::move(scanBatches));
    std::vector<ExpressionPtr> keys;
    keys.push_back(std::make_unique<ColumnRef>(0));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));
    HashAggregateOperator op(&mm, std::move(scan), std::move(keys),
                             std::move(aggs), {"k", "s"},
                             /*spillTriggerGroups=*/1);
    op.open();
    while (op.getNext()) {}
    op.close();

    const std::string text = metrics::prometheusText();
    EXPECT_EQ(text.find("quantum_spills_total 0\n"), std::string::npos)
        << "expected at least one spill recorded:\n" << text;
    metrics::resetForTesting();
}
