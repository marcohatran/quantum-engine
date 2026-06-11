#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "quantum/quantum_c.h"

// Build the plan IR + proto bytes via the in-process C++ helpers; this is
// what the JVM bridge will do in a different process — there's no semantic
// difference, and exercising the C entry points proves the ABI is correct.
#include "plan/Plan.hpp"
#include "plan/PlanProto.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

ScalarExpr col(int32_t i) { return ColumnRefExpr{i}; }
ScalarExpr lit(int64_t v) { return LiteralExpr{v}; }
ScalarExpr bin(BinaryOpType op, ScalarExpr l, ScalarExpr r) {
    auto e = std::make_unique<BinaryOpExpr>();
    e->op  = op;
    e->lhs = std::move(l);
    e->rhs = std::move(r);
    return ScalarExpr{std::move(e)};
}

} // namespace

TEST(QuantumC, EndToEndIntAggregation) {
    QEngine* eng = q_create_engine(64 * 1024 * 1024);
    ASSERT_NE(eng, nullptr);

    // Build a single source "kv" with (k INT64, v INT64).
    const std::vector<int64_t> ks = {1, 2, 1, 3, 2};
    const std::vector<int64_t> vs = {10, 20, 30, 40, 50};
    QColumn* kCol = q_column_int64(eng, ks.data(), nullptr, ks.size());
    QColumn* vCol = q_column_int64(eng, vs.data(), nullptr, vs.size());
    ASSERT_NE(kCol, nullptr);
    ASSERT_NE(vCol, nullptr);

    QColumn* cols[2] = {kCol, vCol};
    QBatch* batch = q_batch_new(eng, cols, 2);
    ASSERT_NE(batch, nullptr);

    QBatch* batches[1] = {batch};
    q_register_source(eng, "kv", batches, 1);

    // SELECT k, SUM(v), COUNT(*) FROM kv GROUP BY k
    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = ScanNode{"kv", Schema({{"k", Int64}, {"v", Int64}})};
    agg->group_keys.push_back(col(0));
    AggregateSpec sum; sum.kind = AggregateKind::Sum; sum.inputType = Int64; sum.input = col(1);
    agg->aggregates.push_back(std::move(sum));
    AggregateSpec cnt; cnt.kind = AggregateKind::CountStar;
    agg->aggregates.push_back(std::move(cnt));
    agg->output_names = {"k", "sum_v", "cnt"};
    PlanNode root{std::move(agg)};
    auto bytes = planToProtoBytes(root);

    QResult* res = q_execute(eng,
                             reinterpret_cast<const uint8_t*>(bytes.data()),
                             bytes.size());
    ASSERT_NE(res, nullptr) << q_last_error();

    std::map<int64_t, std::pair<int64_t, int64_t>> got;
    while (const QBatch* b = q_result_next(res)) {
        const size_t n = q_batch_num_rows(b);
        ASSERT_EQ(q_batch_num_cols(b), 3u);
        ASSERT_EQ(q_batch_col_type(b, 0), Q_TYPE_INT64);
        ASSERT_EQ(q_batch_col_type(b, 1), Q_TYPE_INT64);
        ASSERT_EQ(q_batch_col_type(b, 2), Q_TYPE_INT64);
        const int64_t* kData   = q_batch_col_int64(b, 0);
        const int64_t* sumData = q_batch_col_int64(b, 1);
        const int64_t* cntData = q_batch_col_int64(b, 2);
        ASSERT_NE(kData,   nullptr);
        ASSERT_NE(sumData, nullptr);
        ASSERT_NE(cntData, nullptr);
        for (size_t i = 0; i < n; ++i) {
            got[kData[i]] = {sumData[i], cntData[i]};
        }
    }

    EXPECT_EQ(got[1], (std::pair<int64_t,int64_t>{40, 2}));
    EXPECT_EQ(got[2], (std::pair<int64_t,int64_t>{70, 2}));
    EXPECT_EQ(got[3], (std::pair<int64_t,int64_t>{40, 1}));

    q_destroy_result(res);
    q_destroy_engine(eng);
}

TEST(QuantumC, GroupByStringKey) {
    QEngine* eng = q_create_engine(64 * 1024 * 1024);

    const std::vector<std::string> names = {"alice", "bob", "alice", "carol", "bob"};
    const std::vector<int64_t>     vals  = {10, 20, 30, 40, 50};

    std::vector<const char*> name_ptrs;
    std::vector<int32_t>     name_lens;
    for (const auto& s : names) {
        name_ptrs.push_back(s.data());
        name_lens.push_back(static_cast<int32_t>(s.size()));
    }

    QColumn* nameCol = q_column_string(eng, name_ptrs.data(), name_lens.data(),
                                       nullptr, names.size());
    QColumn* valCol  = q_column_int64(eng, vals.data(), nullptr, vals.size());
    ASSERT_NE(nameCol, nullptr);
    ASSERT_NE(valCol,  nullptr);

    QColumn* cols[2] = {nameCol, valCol};
    QBatch* batch = q_batch_new(eng, cols, 2);
    QBatch* batches[1] = {batch};
    q_register_source(eng, "nv", batches, 1);

    // SELECT name, SUM(v) FROM nv GROUP BY name
    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = ScanNode{"nv", Schema({{"name", DataType{TypeId::STRING}}, {"v", Int64}})};
    agg->group_keys.push_back(col(0));
    AggregateSpec sum; sum.kind = AggregateKind::Sum; sum.inputType = Int64; sum.input = col(1);
    agg->aggregates.push_back(std::move(sum));
    agg->output_names = {"name", "sum_v"};
    PlanNode root{std::move(agg)};
    auto bytes = planToProtoBytes(root);

    QResult* res = q_execute(eng,
                             reinterpret_cast<const uint8_t*>(bytes.data()),
                             bytes.size());
    ASSERT_NE(res, nullptr) << q_last_error();

    std::map<std::string, int64_t> got;
    while (const QBatch* b = q_result_next(res)) {
        ASSERT_EQ(q_batch_col_type(b, 0), Q_TYPE_STRING);
        const size_t n = q_batch_num_rows(b);
        for (size_t i = 0; i < n; ++i) {
            const char* nameBytes = nullptr;
            const int64_t len = q_batch_col_string_at(b, 0, i, &nameBytes);
            ASSERT_GE(len, 0);
            got[std::string(nameBytes, static_cast<size_t>(len))] =
                q_batch_col_int64(b, 1)[i];
        }
    }

    EXPECT_EQ(got["alice"], 40);
    EXPECT_EQ(got["bob"],   70);
    EXPECT_EQ(got["carol"], 40);

    q_destroy_result(res);
    q_destroy_engine(eng);
}

TEST(QuantumC, BadPlanReturnsNullWithErrorMessage) {
    QEngine* eng = q_create_engine(1024);
    ASSERT_NE(eng, nullptr);

    const uint8_t garbage[] = {0xFF, 0xFF, 0xFF, 0xFF};
    QResult* res = q_execute(eng, garbage, sizeof(garbage));
    EXPECT_EQ(res, nullptr);

    const char* err = q_last_error();
    ASSERT_NE(err, nullptr);
    EXPECT_NE(std::string(err).find("PlanProto"), std::string::npos);

    q_destroy_engine(eng);
}

TEST(QuantumC, TraceJsonAndPrometheusCounters) {
    QEngine* eng = q_create_engine(64 * 1024 * 1024);
    ASSERT_NE(eng, nullptr);

    const std::vector<int64_t> ks = {1, 2, 3};
    QColumn* kCol = q_column_int64(eng, ks.data(), nullptr, ks.size());
    QColumn* cols1[1] = {kCol};
    QBatch* batch = q_batch_new(eng, cols1, 1);
    QBatch* batches[1] = {batch};
    q_register_source(eng, "src", batches, 1);

    PlanNode root{ScanNode{"src", Schema({{"k", Int64}})}};
    auto bytes = planToProtoBytes(root);
    QResult* res = q_execute(eng,
                             reinterpret_cast<const uint8_t*>(bytes.data()),
                             bytes.size());
    ASSERT_NE(res, nullptr) << q_last_error();
    while (q_result_next(res) != nullptr) {}

    const char* trace = q_result_trace_json(res);
    ASSERT_NE(trace, nullptr) << q_last_error();
    const std::string t(trace);
    EXPECT_NE(t.find("\"operators\""),   std::string::npos) << t;
    EXPECT_NE(t.find("ScanOperator"),    std::string::npos) << t;
    EXPECT_NE(t.find("\"rows\":3"),      std::string::npos) << t;

    q_destroy_result(res);  // folds the query into the global counters
    const char* prom = q_metrics_prometheus();
    ASSERT_NE(prom, nullptr);
    const std::string p(prom);
    EXPECT_NE(p.find("quantum_queries_total"),      std::string::npos) << p;
    EXPECT_NE(p.find("quantum_rows_emitted_total"), std::string::npos) << p;

    q_destroy_engine(eng);
}
