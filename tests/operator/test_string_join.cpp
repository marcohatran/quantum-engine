#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "expression/ColumnRef.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/HashJoinOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

// Build side: (id INT64, name STRING).
ColumnBatch buildBatch(MemoryManager& mm,
                       const std::vector<int64_t>& ids,
                       const std::vector<std::string>& names) {
    const size_t n = ids.size();
    auto idCol = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) idCol.typed<int64_t>()[i] = ids[i];

    StringBuilder sb(&mm);
    for (const auto& s : names) sb.append(s);
    auto nameCol = sb.build();

    std::vector<ColumnVector> cols;
    cols.push_back(std::move(idCol));
    cols.push_back(std::move(nameCol));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

// Probe side: (id INT64, score INT64).
ColumnBatch probeBatch(MemoryManager& mm,
                       const std::vector<int64_t>& ids,
                       const std::vector<int64_t>& scores) {
    const size_t n = ids.size();
    auto idCol    = ColumnVector::make(Int64, n, &mm);
    auto scoreCol = ColumnVector::make(Int64, n, &mm);
    for (size_t i = 0; i < n; ++i) {
        idCol.typed<int64_t>()[i]    = ids[i];
        scoreCol.typed<int64_t>()[i] = scores[i];
    }
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(idCol));
    cols.push_back(std::move(scoreCol));
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

std::vector<ExpressionPtr> keyOn(size_t idx) {
    std::vector<ExpressionPtr> v;
    v.push_back(std::make_unique<ColumnRef>(idx));
    return v;
}

} // namespace

TEST(HashJoin, PassesStringColumnThroughBuildSide) {
    MemoryManager mm(64 * 1024 * 1024);

    std::vector<ColumnBatch> build;
    build.push_back(buildBatch(mm, {1, 2, 3},     {"alice", "bob", "carol"}));
    build.push_back(buildBatch(mm, {4},           {"dan"}));

    std::vector<ColumnBatch> probe;
    probe.push_back(probeBatch(mm, {3, 1, 5, 4}, {30, 10, 50, 40}));

    auto bScan = std::make_unique<ScanOperator>(
        &mm, Schema({{"id", Int64}, {"name", DataType{TypeId::STRING}}}),
        std::move(build));
    auto pScan = std::make_unique<ScanOperator>(
        &mm, Schema({{"id", Int64}, {"score", Int64}}),
        std::move(probe));

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

    EXPECT_EQ(got.size(), 3u);
    EXPECT_EQ(got[1], (std::pair<std::string,int64_t>{"alice", 10}));
    EXPECT_EQ(got[3], (std::pair<std::string,int64_t>{"carol", 30}));
    EXPECT_EQ(got[4], (std::pair<std::string,int64_t>{"dan",   40}));
}
