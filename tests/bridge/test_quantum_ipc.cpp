#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>

#include "quantum/quantum_c.h"
#include "quantum/quantum_ipc.h"

// Plan IR + proto bytes built via the in-process C++ helpers, same pattern
// as test_quantum_c.cpp.
#include "plan/Plan.hpp"
#include "plan/PlanProto.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

// Serialize one record batch of (k int64, v int64, name utf8 nullable)
// into Arrow IPC stream bytes.
std::vector<uint8_t> makeIpcSource(const std::vector<int64_t>& ks,
                                   const std::vector<int64_t>& vs,
                                   const std::vector<const char*>& names) {
    arrow::Int64Builder kb, vb;
    arrow::StringBuilder nb;
    for (auto k : ks) EXPECT_TRUE(kb.Append(k).ok());
    for (auto v : vs) EXPECT_TRUE(vb.Append(v).ok());
    for (const char* s : names) {
        if (s) EXPECT_TRUE(nb.Append(s).ok());
        else   EXPECT_TRUE(nb.AppendNull().ok());
    }
    std::shared_ptr<arrow::Array> ka, va, na;
    EXPECT_TRUE(kb.Finish(&ka).ok());
    EXPECT_TRUE(vb.Finish(&va).ok());
    EXPECT_TRUE(nb.Finish(&na).ok());

    auto schema = arrow::schema({arrow::field("k", arrow::int64()),
                                 arrow::field("v", arrow::int64()),
                                 arrow::field("name", arrow::utf8())});
    auto rb = arrow::RecordBatch::Make(schema,
                                       static_cast<int64_t>(ks.size()),
                                       {ka, va, na});

    auto sink   = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(sink, schema).ValueOrDie();
    EXPECT_TRUE(writer->WriteRecordBatch(*rb).ok());
    EXPECT_TRUE(writer->Close().ok());
    auto buf = sink->Finish().ValueOrDie();
    return std::vector<uint8_t>(buf->data(), buf->data() + buf->size());
}

std::shared_ptr<arrow::RecordBatch> parseIpc(const uint8_t* data, size_t len) {
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(
        std::make_shared<arrow::io::BufferReader>(
            std::make_shared<arrow::Buffer>(
                data, static_cast<int64_t>(len)))).ValueOrDie();
    auto rb = reader->Next().ValueOrDie();
    EXPECT_NE(rb, nullptr);
    return rb;
}

} // namespace

TEST(QuantumIpc, RegisterExecuteFetchRoundTrip) {
    QEngine* eng = q_create_engine(64 * 1024 * 1024);
    ASSERT_NE(eng, nullptr);

    // Source via IPC: (k, v, name) with a null name.
    auto ipc = makeIpcSource({1, 2, 1, 3, 2},
                             {10, 20, 30, 40, 50},
                             {"a", "b", nullptr, "c", "b"});
    ASSERT_EQ(q_register_source_ipc(eng, "kv", ipc.data(), ipc.size()), 0)
        << q_last_error();

    // SELECT k, SUM(v) FROM kv GROUP BY k — same plan shape as the
    // existing C ABI test, built via the IR helpers.
    auto agg = std::make_unique<HashAggregateNode>();
    agg->child = ScanNode{"kv", Schema({{"k", Int64}, {"v", Int64},
                                        {"name", DataType{TypeId::STRING}}})};
    agg->group_keys.push_back(ColumnRefExpr{0});
    AggregateSpec spec;
    spec.kind      = AggregateKind::Sum;
    spec.input     = ColumnRefExpr{1};
    spec.inputType = Int64;
    agg->aggregates.push_back(std::move(spec));
    agg->output_names = {"k", "sum_v"};
    PlanNode root{std::move(agg)};
    const std::string bytes = planToProtoBytes(root);
    QResult* res = q_execute(eng,
                             reinterpret_cast<const uint8_t*>(bytes.data()),
                             bytes.size());
    ASSERT_NE(res, nullptr) << q_last_error();

    std::map<int64_t, int64_t> got;
    while (true) {
        uint8_t* out  = nullptr;
        size_t   olen = 0;
        const int rc  = q_result_next_ipc(res, &out, &olen);
        ASSERT_GE(rc, 0) << q_last_error();
        if (rc == 0) break;
        // The parsed RecordBatch aliases the IPC buffer (zero-copy read);
        // q_ipc_free only after the arrays have been consumed.
        auto rb = parseIpc(out, olen);
        const auto& kArr = static_cast<const arrow::Int64Array&>(*rb->column(0));
        const auto& sArr = static_cast<const arrow::Int64Array&>(*rb->column(1));
        for (int64_t i = 0; i < rb->num_rows(); ++i) {
            got[kArr.Value(i)] = sArr.Value(i);
        }
        rb.reset();
        q_ipc_free(out);
    }
    q_destroy_result(res);
    q_destroy_engine(eng);

    EXPECT_EQ(got.size(), 3u);
    EXPECT_EQ(got[1], 40);
    EXPECT_EQ(got[2], 70);
    EXPECT_EQ(got[3], 40);
}

TEST(QuantumIpc, StringsAndNullsSurviveRoundTrip) {
    QEngine* eng = q_create_engine(64 * 1024 * 1024);
    ASSERT_NE(eng, nullptr);

    auto ipc = makeIpcSource({1, 2, 3},
                             {10, 20, 30},
                             {"alice", nullptr, "carol"});
    ASSERT_EQ(q_register_source_ipc(eng, "src", ipc.data(), ipc.size()), 0)
        << q_last_error();

    // Bare scan: everything comes back, including the null string.
    PlanNode root{ScanNode{"src", Schema({{"k", Int64}, {"v", Int64},
                                          {"name", DataType{TypeId::STRING}}})}};
    const std::string bytes = planToProtoBytes(root);

    QResult* res = q_execute(eng,
                             reinterpret_cast<const uint8_t*>(bytes.data()),
                             bytes.size());
    ASSERT_NE(res, nullptr) << q_last_error();

    uint8_t* out  = nullptr;
    size_t   olen = 0;
    ASSERT_EQ(q_result_next_ipc(res, &out, &olen), 1) << q_last_error();
    auto rb = parseIpc(out, olen);

    ASSERT_EQ(rb->num_rows(), 3);
    const auto& names = static_cast<const arrow::StringArray&>(*rb->column(2));
    EXPECT_EQ(names.GetString(0), "alice");
    EXPECT_TRUE(names.IsNull(1));
    EXPECT_EQ(names.GetString(2), "carol");
    rb.reset();
    q_ipc_free(out);

    uint8_t* out2 = nullptr;
    size_t   olen2 = 0;
    EXPECT_EQ(q_result_next_ipc(res, &out2, &olen2), 0);  // EOS

    q_destroy_result(res);
    q_destroy_engine(eng);
}
