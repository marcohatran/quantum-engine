#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/Metrics.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "plan/OperatorRegistry.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "plan/PlanProto.hpp"
#include "quantum/quantum_c.h"
#include "vector/ColumnBatch.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

// Example plugin operator: adds a constant (parsed from the extension
// payload) to INT64 column 0 of every batch from its single child.
class AddConstantOperator final : public Operator {
public:
    AddConstantOperator(MemoryManager* mm, OperatorPtr child, int64_t delta)
        : Operator(mm), child_(std::move(child)), delta_(delta) {}

    void open() override  { child_->open(); }
    void close() override { child_->close(); }
    const Schema& outputSchema() const override { return child_->outputSchema(); }

    std::optional<ColumnBatch> getNext() override {
        auto b = child_->getNext();
        if (!b.has_value()) return std::nullopt;
        auto* vals = b->column(0).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) vals[i] += delta_;
        return b;
    }

private:
    OperatorPtr child_;
    int64_t     delta_;
};

void registerAddConstant(const std::string& type) {
    OperatorRegistry::instance().registerFactory(
        type, [](ExtensionContext ctx) -> OperatorPtr {
            if (ctx.children.size() != 1) {
                raise(ErrorCode::InvalidArgument,
                      "add_constant: expected exactly 1 child");
            }
            const int64_t delta = std::stoll(ctx.payload);
            return std::make_unique<AddConstantOperator>(
                ctx.mm, std::move(ctx.children[0]), delta);
        });
}

ColumnBatch kBatch(MemoryManager& mm, const std::vector<int64_t>& ks) {
    auto k = ColumnVector::make(Int64, ks.size(), &mm);
    for (size_t i = 0; i < ks.size(); ++i) k.typed<int64_t>()[i] = ks[i];
    std::vector<ColumnVector> cols;
    cols.push_back(std::move(k));
    return ColumnBatch(std::move(cols), PositionList::identity(ks.size()));
}

PlanNode extensionOverScan(const std::string& type, const std::string& payload) {
    auto ext = std::make_unique<ExtensionNode>();
    ext->extension_type = type;
    ext->payload        = payload;
    ext->children.push_back(ScanNode{"src", Schema({{"k", Int64}})});
    return PlanNode{std::move(ext)};
}

} // namespace

TEST(OperatorRegistry, PluginOperatorBuildsAndExecutes) {
    OperatorRegistry::instance().clearForTesting();
    registerAddConstant("test.add_constant");

    MemoryManager mm(64 * 1024 * 1024);
    SourceCatalog catalog;
    catalog["src"].push_back(kBatch(mm, {1, 2, 3}));

    QueryMetrics qm;
    PlanBuilder pb(&mm, &qm);
    auto op = pb.build(extensionOverScan("test.add_constant", "100"), catalog);

    op->open();
    std::vector<int64_t> got;
    while (auto b = op->getNext()) {
        const auto* k = b->column(0).typed<int64_t>();
        for (size_t i = 0; i < b->numRows(); ++i) got.push_back(k[i]);
    }
    op->close();
    EXPECT_EQ(got, (std::vector<int64_t>{101, 102, 103}));

    // Plugin operators report through the same metrics seam: the root
    // entry carries the plugin's class name.
    ASSERT_EQ(qm.operators().size(), 2u);
    EXPECT_EQ(qm.root()->name, "AddConstantOperator");
    EXPECT_EQ(qm.root()->rows, 3u);
}

TEST(OperatorRegistry, ExtensionNodeSurvivesProtoRoundTrip) {
    OperatorRegistry::instance().clearForTesting();
    registerAddConstant("test.add_constant");

    const std::string bytes =
        planToProtoBytes(extensionOverScan("test.add_constant", "7"));
    PlanNode decoded = planFromProtoBytes(bytes);

    MemoryManager mm(64 * 1024 * 1024);
    SourceCatalog catalog;
    catalog["src"].push_back(kBatch(mm, {10}));
    PlanBuilder pb(&mm);
    auto op = pb.build(decoded, catalog);
    op->open();
    auto b = op->getNext();
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(b->column(0).typed<int64_t>()[0], 17);
    op->close();
}

TEST(OperatorRegistry, EndToEndThroughCAbi) {
    OperatorRegistry::instance().clearForTesting();
    registerAddConstant("test.add_constant");

    QEngine* eng = q_create_engine(64 * 1024 * 1024);
    const std::vector<int64_t> ks = {5, 6};
    QColumn* kCol = q_column_int64(eng, ks.data(), nullptr, ks.size());
    QColumn* cols[1] = {kCol};
    QBatch* batch = q_batch_new(eng, cols, 1);
    QBatch* batches[1] = {batch};
    q_register_source(eng, "src", batches, 1);

    const std::string bytes =
        planToProtoBytes(extensionOverScan("test.add_constant", "1000"));
    QResult* res = q_execute(eng,
                             reinterpret_cast<const uint8_t*>(bytes.data()),
                             bytes.size());
    ASSERT_NE(res, nullptr) << q_last_error();
    const QBatch* out = q_result_next(res);
    ASSERT_NE(out, nullptr);
    const int64_t* vals = q_batch_col_int64(out, 0);
    EXPECT_EQ(vals[0], 1005);
    EXPECT_EQ(vals[1], 1006);
    q_destroy_result(res);
    q_destroy_engine(eng);
}

TEST(OperatorRegistry, UnknownExtensionTypeRaises) {
    OperatorRegistry::instance().clearForTesting();
    MemoryManager mm(1 << 20);
    SourceCatalog catalog;
    catalog["src"].push_back(kBatch(mm, {1}));
    PlanBuilder pb(&mm);
    EXPECT_THROW(pb.build(extensionOverScan("vendor.unknown", ""), catalog),
                 QuantumError);
}

TEST(OperatorRegistry, DuplicateRegistrationRaises) {
    OperatorRegistry::instance().clearForTesting();
    registerAddConstant("test.dup");
    EXPECT_THROW(registerAddConstant("test.dup"), QuantumError);
}
