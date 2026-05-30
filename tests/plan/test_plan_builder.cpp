#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "memory/MemoryManager.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "plan/PlanJson.hpp"
#include "vector/ColumnBatch.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

// Build a single batch of (k INT64, v INT64).
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

ScalarExpr col(int32_t i)    { return ColumnRefExpr{i}; }
ScalarExpr lit(int64_t v)    { return LiteralExpr{v}; }

ScalarExpr bin(BinaryOpType op, ScalarExpr l, ScalarExpr r) {
    auto e = std::make_unique<BinaryOpExpr>();
    e->op  = op;
    e->lhs = std::move(l);
    e->rhs = std::move(r);
    return ScalarExpr{std::move(e)};
}

PlanNode scanKV() {
    return ScanNode{"kv", Schema({{"k", Int64}, {"v", Int64}})};
}

} // namespace

TEST(PlanBuilder, ScanFilterProjectEndToEnd) {
    MemoryManager mm(64 * 1024 * 1024);

    // SELECT k, v+10 FROM kv WHERE v > 1
    auto filter = std::make_unique<FilterNode>();
    filter->child     = scanKV();
    filter->predicate = bin(BinaryOpType::Gt, col(1), lit(int64_t{1}));

    auto project = std::make_unique<ProjectNode>();
    project->child = PlanNode{std::move(filter)};
    project->exprs.push_back(col(0));
    project->exprs.push_back(bin(BinaryOpType::Add, col(1), lit(int64_t{10})));
    project->output_names = {"k", "v_plus_10"};

    PlanNode root{std::move(project)};

    // Round-trip through JSON to also validate the codec end-to-end.
    auto json = planToJson(root);
    auto roundtripped = planFromJson(json);

    SourceCatalog catalog;
    catalog["kv"].push_back(intIntBatch(mm, {1, 2, 3, 4}, {0, 1, 2, 3}));

    PlanBuilder pb(&mm);
    auto op = pb.build(roundtripped, catalog);

    std::vector<std::pair<int64_t, int64_t>> got;
    op->open();
    while (auto b = op->getNext()) {
        auto dense = b->selectActive(&mm);
        for (size_t i = 0; i < dense.numRows(); ++i) {
            got.emplace_back(dense.column(0).typed<int64_t>()[i],
                             dense.column(1).typed<int64_t>()[i]);
        }
    }
    op->close();

    // v > 1 keeps rows 2 and 3 (values 2 and 3). v+10 = 12, 13.
    std::vector<std::pair<int64_t, int64_t>> want = {{3, 12}, {4, 13}};
    EXPECT_EQ(got, want);
}

TEST(PlanBuilder, HashAggregateFromJson) {
    MemoryManager mm(64 * 1024 * 1024);

    const std::string js = R"json({
      "kind": "hash_agg",
      "child": {
        "kind": "scan",
        "source_id": "kv",
        "schema": [
          {"name": "k", "type": "int64"},
          {"name": "v", "type": "int64"}
        ]
      },
      "group_keys": [{"kind": "column_ref", "index": 0}],
      "aggregates": [
        {"kind": "sum", "input_type": "int64",
         "input": {"kind": "column_ref", "index": 1}},
        {"kind": "count_star"}
      ],
      "output_names": ["k", "sum_v", "cnt"]
    })json";

    PlanNode root = planFromJson(js);

    SourceCatalog catalog;
    catalog["kv"].push_back(intIntBatch(mm,
        {1, 2, 1, 3, 2}, {10, 20, 30, 40, 50}));

    PlanBuilder pb(&mm);
    auto op = pb.build(root, catalog);

    std::map<int64_t, std::pair<int64_t, int64_t>> got;
    op->open();
    while (auto b = op->getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            got[b->column(0).typed<int64_t>()[i]] = {
                b->column(1).typed<int64_t>()[i],
                b->column(2).typed<int64_t>()[i],
            };
        }
    }
    op->close();

    EXPECT_EQ(got[1], (std::pair<int64_t,int64_t>{40, 2}));
    EXPECT_EQ(got[2], (std::pair<int64_t,int64_t>{70, 2}));
    EXPECT_EQ(got[3], (std::pair<int64_t,int64_t>{40, 1}));
}

TEST(PlanBuilder, HashJoinFromJson) {
    MemoryManager mm(64 * 1024 * 1024);

    auto build = std::make_unique<HashJoinNode>();
    build->build     = ScanNode{"build", Schema({{"k", Int64}, {"v", Int64}})};
    build->probe     = ScanNode{"probe", Schema({{"k", Int64}, {"v", Int64}})};
    build->build_keys.push_back(col(0));
    build->probe_keys.push_back(col(0));
    build->join_type = JoinTypeIR::Inner;
    build->output_names = {"bk", "bv", "pk", "pv"};
    PlanNode root{std::move(build)};

    auto js = planToJson(root);
    auto root2 = planFromJson(js);

    SourceCatalog catalog;
    catalog["build"].push_back(intIntBatch(mm, {1, 2, 3}, {100, 200, 300}));
    catalog["probe"].push_back(intIntBatch(mm, {2, 3, 4}, {20, 30, 40}));

    PlanBuilder pb(&mm);
    auto op = pb.build(root2, catalog);

    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> got;
    op->open();
    while (auto b = op->getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            got.emplace_back(b->column(0).typed<int64_t>()[i],
                             b->column(1).typed<int64_t>()[i],
                             b->column(2).typed<int64_t>()[i],
                             b->column(3).typed<int64_t>()[i]);
        }
    }
    op->close();

    std::sort(got.begin(), got.end());
    decltype(got) want = {{2, 200, 2, 20}, {3, 300, 3, 30}};
    EXPECT_EQ(got, want);
}

TEST(PlanJson, ScalarExprRoundtrip) {
    auto e = bin(BinaryOpType::Add,
                 col(0),
                 bin(BinaryOpType::Mul, col(1), lit(int64_t{2})));
    auto js = scalarExprToJson(e);
    auto e2 = scalarExprFromJson(js);
    auto js2 = scalarExprToJson(e2);
    EXPECT_EQ(js, js2);
}

TEST(PlanJson, LiteralStringRoundtrip) {
    LiteralExpr lit;
    lit.value = std::string{"hello"};
    auto js = scalarExprToJson(ScalarExpr{lit});
    auto e2 = scalarExprFromJson(js);
    auto js2 = scalarExprToJson(e2);
    EXPECT_EQ(js, js2);
}
