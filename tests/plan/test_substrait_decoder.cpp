#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <substrait/plan.pb.h>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "plan/SubstraitDecoder.hpp"
#include "quantum/quantum_c.h"
#include "vector/ColumnBatch.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

ColumnBatch kvBatch(MemoryManager& mm,
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

// Declare an extension function and return its anchor.
uint32_t declareFn(substrait::Plan& plan, const std::string& name) {
    const auto anchor = static_cast<uint32_t>(plan.extensions_size() + 1);
    auto* ext = plan.add_extensions()->mutable_extension_function();
    ext->set_function_anchor(anchor);
    ext->set_name(name);
    return anchor;
}

substrait::Expression fieldRef(int32_t idx) {
    substrait::Expression e;
    e.mutable_selection()->mutable_direct_reference()
        ->mutable_struct_field()->set_field(idx);
    e.mutable_selection()->mutable_root_reference();
    return e;
}

substrait::Expression i64Lit(int64_t v) {
    substrait::Expression e;
    e.mutable_literal()->set_i64(v);
    return e;
}

// ReadRel over named table "kv" with (k i64, v i64).
substrait::Rel readKv() {
    substrait::Rel rel;
    auto* read = rel.mutable_read();
    read->mutable_named_table()->add_names("kv");
    auto* ns = read->mutable_base_schema();
    ns->add_names("k");
    ns->add_names("v");
    ns->mutable_struct_()->add_types()->mutable_i64();
    ns->mutable_struct_()->add_types()->mutable_i64();
    return rel;
}

} // namespace

TEST(SubstraitDecoder, ReadFilterAggregateEndToEnd) {
    // SELECT k, SUM(v), COUNT(*) FROM kv WHERE v > 15 GROUP BY k
    substrait::Plan plan;
    const uint32_t gtFn    = declareFn(plan, "gt:any_any");
    const uint32_t sumFn   = declareFn(plan, "sum:i64");
    const uint32_t countFn = declareFn(plan, "count:");

    substrait::Rel filterRel;
    auto* filter = filterRel.mutable_filter();
    *filter->mutable_input() = readKv();
    auto* cond = filter->mutable_condition()->mutable_scalar_function();
    cond->set_function_reference(gtFn);
    *cond->add_arguments()->mutable_value() = fieldRef(1);
    *cond->add_arguments()->mutable_value() = i64Lit(15);

    substrait::Rel aggRel;
    auto* agg = aggRel.mutable_aggregate();
    *agg->mutable_input() = filterRel;
    *agg->add_groupings()->add_grouping_expressions() = fieldRef(0);
    auto* sum = agg->add_measures()->mutable_measure();
    sum->set_function_reference(sumFn);
    *sum->add_arguments()->mutable_value() = fieldRef(1);
    auto* cnt = agg->add_measures()->mutable_measure();
    cnt->set_function_reference(countFn);

    *plan.add_relations()->mutable_root()->mutable_input() = aggRel;

    // Execute through the C ABI substrait entry point.
    QEngine* eng = q_create_engine(64 * 1024 * 1024);
    MemoryManager localMm(1 << 20);  // only for building the input arrays
    const std::vector<int64_t> ks = {1, 2, 1, 3, 2};
    const std::vector<int64_t> vs = {10, 20, 30, 40, 50};
    QColumn* kCol = q_column_int64(eng, ks.data(), nullptr, ks.size());
    QColumn* vCol = q_column_int64(eng, vs.data(), nullptr, vs.size());
    QColumn* cols[2] = {kCol, vCol};
    QBatch* batch = q_batch_new(eng, cols, 2);
    QBatch* batches[1] = {batch};
    q_register_source(eng, "kv", batches, 1);

    const std::string bytes = plan.SerializeAsString();
    QResult* res = q_execute_substrait(
        eng, reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    ASSERT_NE(res, nullptr) << q_last_error();

    // v > 15 keeps (2,20),(1,30),(3,40),(2,50):
    //   k=1 -> sum 30 cnt 1; k=2 -> sum 70 cnt 2; k=3 -> sum 40 cnt 1
    std::map<int64_t, std::pair<int64_t, int64_t>> got;
    while (const QBatch* out = q_result_next(res)) {
        const int64_t* kArr = q_batch_col_int64(out, 0);
        const int64_t* sArr = q_batch_col_int64(out, 1);
        const int64_t* cArr = q_batch_col_int64(out, 2);
        for (size_t i = 0; i < q_batch_num_rows(out); ++i) {
            got[kArr[i]] = {sArr[i], cArr[i]};
        }
    }
    EXPECT_EQ(got.size(), 3u);
    EXPECT_EQ(got[1], (std::pair<int64_t, int64_t>{30, 1}));
    EXPECT_EQ(got[2], (std::pair<int64_t, int64_t>{70, 2}));
    EXPECT_EQ(got[3], (std::pair<int64_t, int64_t>{40, 1}));
    q_destroy_result(res);
    q_destroy_engine(eng);
}

TEST(SubstraitDecoder, ProjectEmitSortFetch) {
    // SELECT v+k AS s FROM kv ORDER BY s DESC LIMIT 2
    substrait::Plan plan;
    const uint32_t addFn = declareFn(plan, "add:i64_i64");

    substrait::Rel projRel;
    auto* proj = projRel.mutable_project();
    *proj->mutable_input() = readKv();
    auto* addExpr = proj->add_expressions()->mutable_scalar_function();
    addExpr->set_function_reference(addFn);
    *addExpr->add_arguments()->mutable_value() = fieldRef(0);
    *addExpr->add_arguments()->mutable_value() = fieldRef(1);
    // emit only the computed column (index 2 = nChild(2) + expr 0).
    proj->mutable_common()->mutable_emit()->add_output_mapping(2);

    substrait::Rel sortRel;
    auto* sort = sortRel.mutable_sort();
    *sort->mutable_input() = projRel;
    auto* sf = sort->add_sorts();
    *sf->mutable_expr() = fieldRef(0);
    sf->set_direction(substrait::SortField::SORT_DIRECTION_DESC_NULLS_LAST);

    substrait::Rel fetchRel;
    auto* fetch = fetchRel.mutable_fetch();
    *fetch->mutable_input() = sortRel;
    fetch->set_count(2);

    *plan.add_relations()->mutable_root()->mutable_input() = fetchRel;

    // Decode + build natively (no C ABI this time).
    MemoryManager mm(64 * 1024 * 1024);
    SourceCatalog catalog;
    catalog["kv"].push_back(kvBatch(mm, {1, 2, 3}, {10, 20, 30}));

    const std::string bytes = plan.SerializeAsString();
    PlanNode node = planFromSubstraitBytes(bytes);
    PlanBuilder pb(&mm);
    auto op = pb.build(node, catalog);
    op->open();
    std::vector<int64_t> got;
    while (auto b = op->getNext()) {
        // LimitOperator truncates the position list; densify before reading.
        auto dense = b->selectActive(&mm);
        for (size_t i = 0; i < dense.numRows(); ++i) {
            got.push_back(dense.column(0).typed<int64_t>()[i]);
        }
    }
    op->close();
    // sums: 11, 22, 33 -> desc limit 2: 33, 22
    EXPECT_EQ(got, (std::vector<int64_t>{33, 22}));
}

TEST(SubstraitDecoder, UnsupportedRelRaises) {
    substrait::Plan plan;
    substrait::Rel cross;
    cross.mutable_cross();  // CrossRel — not supported
    *plan.add_relations()->mutable_root()->mutable_input() = cross;
    const std::string bytes = plan.SerializeAsString();
    EXPECT_THROW(planFromSubstraitBytes(bytes), QuantumError);
}

TEST(SubstraitDecoder, GarbageBytesRaise) {
    EXPECT_THROW(planFromSubstraitBytes("not a substrait plan"), QuantumError);
}
