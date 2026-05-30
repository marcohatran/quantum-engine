#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/HashJoinOperator.hpp"
#include "operator/ProjectOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "vector/DictStringBuilder.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

ColumnBatch dictNameValueBatch(MemoryManager& mm,
                               const std::vector<std::string>& names,
                               const std::vector<int64_t>&     values) {
    const size_t n = names.size();
    DictStringBuilder sb(&mm);
    for (const auto& s : names) sb.append(s);
    auto nameCol = sb.build();

    auto valCol = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) valCol.typed<int64_t>()[i] = values[i];

    std::vector<ColumnVector> cols;
    cols.push_back(std::move(nameCol));
    cols.push_back(std::move(valCol));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

OperatorPtr scanNameVal(MemoryManager& mm, std::vector<ColumnBatch> src) {
    return std::make_unique<ScanOperator>(
        &mm,
        Schema({{"name", DataType{TypeId::STRING}}, {"v", Int64}}),
        std::move(src));
}

std::vector<ExpressionPtr> keyOn(size_t idx) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(idx));
    return v;
}

} // namespace

TEST(DictE2E, FilterPreservesDictLayout) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(dictNameValueBatch(mm,
        {"a", "b", "a", "c"},
        {10,  20,  30,  40}));

    auto scan = scanNameVal(mm, std::move(src));
    auto pred = std::make_unique<BinaryExpr>(
        BinaryOp::Gt,
        std::make_unique<ColumnRef>(1),
        std::make_unique<Literal>(int64_t{15}));
    FilterOperator op(&mm, std::move(scan), std::move(pred));

    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->activeRows(), 3u);
    // Filter doesn't copy columns; the name column should still be the
    // original dict-encoded one.
    EXPECT_TRUE(batch->column(0).isDictEncoded());

    // selectActive on the (post-filter) batch keeps the dict layout via the
    // dict-aware gather path.
    auto dense = batch->selectActive(&mm);
    EXPECT_TRUE(dense.column(0).isDictEncoded());
    EXPECT_EQ(dense.column(0).stringAt(0), "b");
    EXPECT_EQ(dense.column(0).stringAt(1), "a");
    EXPECT_EQ(dense.column(0).stringAt(2), "c");
}

TEST(DictE2E, StringComparisonAcrossDictAndRegular) {
    MemoryManager mm(64 * 1024 * 1024);
    DictStringBuilder dsb(&mm);
    dsb.append("apple"); dsb.append("banana"); dsb.append("apple");
    StringBuilder ssb(&mm);
    ssb.append("apple"); ssb.append("apple"); ssb.append("banana");

    std::vector<ColumnVector> cols;
    cols.push_back(dsb.build());
    cols.push_back(ssb.build());
    ColumnBatch batch(std::move(cols), PositionList::identity(3));

    BinaryExpr eq(BinaryOp::Eq,
                  std::make_unique<ColumnRef>(0),
                  std::make_unique<ColumnRef>(1));
    auto r = eq.eval(batch, &mm);
    EXPECT_EQ(r.typed<uint8_t>()[0], 1);   // apple == apple
    EXPECT_EQ(r.typed<uint8_t>()[1], 0);   // banana != apple
    EXPECT_EQ(r.typed<uint8_t>()[2], 0);   // apple != banana
}

TEST(DictE2E, GroupByDictEncodedKey) {
    MemoryManager mm(64 * 1024 * 1024);
    std::vector<ColumnBatch> src;
    src.push_back(dictNameValueBatch(mm,
        {"alice", "bob", "alice", "carol", "bob"},
        {10,      20,    30,      40,      50}));

    auto scan = scanNameVal(mm, std::move(src));
    std::vector<AggregateFunctionPtr> aggs;
    aggs.push_back(std::make_unique<SumI64>(std::make_unique<ColumnRef>(1)));

    HashAggregateOperator op(&mm, std::move(scan), keyOn(0),
                             std::move(aggs),
                             {"name", "sum_v"});
    op.open();
    auto batch = op.getNext();
    op.close();
    ASSERT_TRUE(batch.has_value());
    ASSERT_EQ(batch->numRows(), 3u);

    std::map<std::string, int64_t> got;
    for (size_t i = 0; i < batch->numRows(); ++i) {
        got[std::string(batch->column(0).stringAt(i))] =
            batch->column(1).typed<int64_t>()[i];
    }
    EXPECT_EQ(got["alice"], 40);
    EXPECT_EQ(got["bob"],   70);
    EXPECT_EQ(got["carol"], 40);
}

TEST(DictE2E, HashJoinWithDictBuildValueColumn) {
    MemoryManager mm(64 * 1024 * 1024);

    // Build side: (id INT64, name STRING dict-encoded).
    auto build = [&](const std::vector<int64_t>& ids,
                     const std::vector<std::string>& names) {
        const size_t n = ids.size();
        auto idCol = ColumnVector::make(Int64, n, &mm);
        for (size_t i = 0; i < n; ++i) idCol.typed<int64_t>()[i] = ids[i];
        DictStringBuilder sb(&mm);
        for (const auto& s : names) sb.append(s);
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(idCol));
        cols.push_back(sb.build());
        return ColumnBatch(std::move(cols), PositionList::identity(n));
    };

    std::vector<ColumnBatch> buildSrc;
    buildSrc.push_back(build({1, 2, 3}, {"alice", "bob", "carol"}));

    std::vector<ColumnBatch> probeSrc;
    auto probeBatch = ColumnBatch{};
    {
        auto idCol = ColumnVector::make(Int64, 3, &mm);
        idCol.typed<int64_t>()[0] = 2;
        idCol.typed<int64_t>()[1] = 3;
        idCol.typed<int64_t>()[2] = 4;
        auto scoreCol = ColumnVector::make(Int64, 3, &mm);
        scoreCol.typed<int64_t>()[0] = 20;
        scoreCol.typed<int64_t>()[1] = 30;
        scoreCol.typed<int64_t>()[2] = 40;
        std::vector<ColumnVector> cols;
        cols.push_back(std::move(idCol));
        cols.push_back(std::move(scoreCol));
        probeSrc.push_back(ColumnBatch(std::move(cols), PositionList::identity(3)));
    }

    auto bScan = std::make_unique<ScanOperator>(
        &mm,
        Schema({{"id", Int64}, {"name", DataType{TypeId::STRING}}}),
        std::move(buildSrc));
    auto pScan = std::make_unique<ScanOperator>(
        &mm,
        Schema({{"id", Int64}, {"score", Int64}}),
        std::move(probeSrc));

    HashJoinOperator op(&mm, std::move(bScan), std::move(pScan),
                        keyOn(0), keyOn(0),
                        {"id", "name", "pid", "score"});

    std::map<int64_t, std::pair<std::string, int64_t>> got;
    op.open();
    while (auto b = op.getNext()) {
        if (b->numRows() == 0) continue;
        for (size_t i = 0; i < b->numRows(); ++i) {
            got[b->column(0).typed<int64_t>()[i]] = {
                std::string(b->column(1).stringAt(i)),
                b->column(3).typed<int64_t>()[i],
            };
        }
    }
    op.close();

    EXPECT_EQ(got.size(), 2u);
    EXPECT_EQ(got[2], (std::pair<std::string,int64_t>{"bob",   20}));
    EXPECT_EQ(got[3], (std::pair<std::string,int64_t>{"carol", 30}));
}
