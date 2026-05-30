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
#include "plan/PlanProto.hpp"
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
ScalarExpr lit(int64_t v) { return LiteralExpr{v}; }
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

TEST(PlanProto, ScalarExprBytesRoundtrip) {
    auto e = bin(BinaryOpType::Add,
                 col(0),
                 bin(BinaryOpType::Mul, col(1), lit(int64_t{2})));
    auto bytes = scalarExprToProtoBytes(e);
    auto e2    = scalarExprFromProtoBytes(bytes);

    // Re-encode via JSON for a stable comparison key.
    EXPECT_EQ(scalarExprToJson(e), scalarExprToJson(e2));
}

TEST(PlanProto, LiteralStringBytesRoundtrip) {
    LiteralExpr l;
    l.value = std::string{"hello world"};
    ScalarExpr e{l};

    auto bytes = scalarExprToProtoBytes(e);
    auto e2    = scalarExprFromProtoBytes(bytes);
    EXPECT_EQ(scalarExprToJson(e), scalarExprToJson(e2));
}

TEST(PlanProto, ScanFilterProjectRoundtripAndExecute) {
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

    // Round-trip through protobuf bytes.
    auto bytes = planToProtoBytes(root);
    auto root2 = planFromProtoBytes(bytes);

    SourceCatalog catalog;
    catalog["kv"].push_back(intIntBatch(mm, {1, 2, 3, 4}, {0, 1, 2, 3}));

    PlanBuilder pb(&mm);
    auto op = pb.build(root2, catalog);

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

    std::vector<std::pair<int64_t, int64_t>> want = {{3, 12}, {4, 13}};
    EXPECT_EQ(got, want);
}

TEST(PlanProto, HashAggregateRoundtripAndExecute) {
    MemoryManager mm(64 * 1024 * 1024);

    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = scanKV();
    agg->group_keys.push_back(col(0));

    AggregateSpec sumSpec;
    sumSpec.kind = AggregateKind::Sum;
    sumSpec.inputType = Int64;
    sumSpec.input = col(1);
    agg->aggregates.push_back(std::move(sumSpec));

    AggregateSpec cntSpec;
    cntSpec.kind = AggregateKind::CountStar;
    agg->aggregates.push_back(std::move(cntSpec));

    agg->output_names = {"k", "sum_v", "cnt"};
    PlanNode root{std::move(agg)};

    auto bytes = planToProtoBytes(root);
    auto root2 = planFromProtoBytes(bytes);

    SourceCatalog catalog;
    catalog["kv"].push_back(intIntBatch(mm,
        {1, 2, 1, 3, 2}, {10, 20, 30, 40, 50}));

    PlanBuilder pb(&mm);
    auto op = pb.build(root2, catalog);

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

// JSON and Protobuf should be semantically equivalent: encoding via one
// codec and decoding via the other (modulo binary representation) must
// reconstruct an IR that JSON-encodes back to the original string.
TEST(PlanProto, JsonProtoCrossCodecEquivalence) {
    auto p = std::make_unique<ProjectNode>();
    p->child = scanKV();
    p->exprs.push_back(bin(BinaryOpType::Sub, col(0), col(1)));
    p->output_names = {"diff"};
    PlanNode root{std::move(p)};

    const std::string originalJson = planToJson(root);

    // JSON -> IR -> proto -> IR -> JSON
    auto irFromJson = planFromJson(originalJson);
    auto protoBytes = planToProtoBytes(irFromJson);
    auto irFromProto = planFromProtoBytes(protoBytes);
    EXPECT_EQ(planToJson(irFromProto), originalJson);
}
