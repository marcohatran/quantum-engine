#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "quantum/quantum_arrow.h"
#include "quantum/quantum_c.h"

namespace {

// Build a tiny QColumn(INT64) directly and export it as Arrow.
TEST(QuantumArrow, Int64ExportZeroCopy) {
    QEngine* eng = q_create_engine(64 * 1024);
    ASSERT_NE(eng, nullptr);

    const std::vector<int64_t> data = {10, 20, 30, 40};
    const std::vector<uint8_t> val  = {1,  0,  1,  1};   // row 1 is NULL
    QColumn* col = q_column_int64(eng, data.data(), val.data(), data.size());
    ASSERT_NE(col, nullptr);

    ArrowSchema schema{};
    ArrowArray  array{};
    ASSERT_EQ(q_column_export_arrow(col, &schema, &array), 0)
        << q_last_error();

    EXPECT_STREQ(schema.format, "l");
    EXPECT_EQ(array.length, 4);
    EXPECT_EQ(array.null_count, 1);
    EXPECT_EQ(array.n_buffers, 2);

    // Buffers must be zero-copy views — pointers identical to underlying
    // buffer addresses.
    const int64_t* values = static_cast<const int64_t*>(array.buffers[1]);
    EXPECT_EQ(values[0], 10);
    EXPECT_EQ(values[2], 30);
    EXPECT_EQ(values[3], 40);

    // Validity bit at index 1 should be 0.
    const uint8_t* validity = static_cast<const uint8_t*>(array.buffers[0]);
    EXPECT_EQ((validity[0] >> 1) & 1U, 0u);

    array.release(&array);
    schema.release(&schema);
    q_destroy_engine(eng);
}

TEST(QuantumArrow, Int64ImportRoundtrip) {
    QEngine* eng = q_create_engine(64 * 1024);
    ASSERT_NE(eng, nullptr);

    const std::vector<int64_t> data = {7, 8, 9, 10, 11};
    const std::vector<uint8_t> val  = {1, 1, 0, 1, 1};
    QColumn* original = q_column_int64(eng, data.data(), val.data(), data.size());
    ASSERT_NE(original, nullptr);

    ArrowSchema schema{};
    ArrowArray  array{};
    ASSERT_EQ(q_column_export_arrow(original, &schema, &array), 0);

    QColumn* imported = q_column_import_arrow(eng, &schema, &array);
    ASSERT_NE(imported, nullptr) << q_last_error();

    array.release(&array);
    schema.release(&schema);

    // Now wrap imported in a batch and inspect via the existing C API.
    QColumn* cols[1] = {imported};
    QBatch* batch = q_batch_new(eng, cols, 1);
    ASSERT_NE(batch, nullptr);

    EXPECT_EQ(q_batch_num_rows(batch), 5u);
    EXPECT_EQ(q_batch_col_type(batch, 0), Q_TYPE_INT64);
    const int64_t* v = q_batch_col_int64(batch, 0);
    EXPECT_EQ(v[0], 7);
    EXPECT_EQ(v[4], 11);
    EXPECT_EQ(q_batch_col_valid(batch, 0, 0), 1);
    EXPECT_EQ(q_batch_col_valid(batch, 0, 2), 0);

    q_destroy_engine(eng);
}

TEST(QuantumArrow, StringExportZeroCopyAndRoundtrip) {
    QEngine* eng = q_create_engine(64 * 1024);
    const std::vector<std::string> names = {"alice", "bob", "carol"};
    std::vector<const char*> ptrs;
    std::vector<int32_t>     lens;
    for (const auto& s : names) {
        ptrs.push_back(s.data());
        lens.push_back(static_cast<int32_t>(s.size()));
    }
    QColumn* col = q_column_string(eng, ptrs.data(), lens.data(), nullptr, names.size());
    ASSERT_NE(col, nullptr);

    ArrowSchema schema{};
    ArrowArray  array{};
    ASSERT_EQ(q_column_export_arrow(col, &schema, &array), 0);

    EXPECT_STREQ(schema.format, "u");
    EXPECT_EQ(array.length, 3);
    EXPECT_EQ(array.n_buffers, 3);

    const int32_t* offsets = static_cast<const int32_t*>(array.buffers[1]);
    const char*    bytes   = static_cast<const char*>(array.buffers[2]);
    EXPECT_EQ(std::string_view(bytes + offsets[0], offsets[1] - offsets[0]), "alice");
    EXPECT_EQ(std::string_view(bytes + offsets[1], offsets[2] - offsets[1]), "bob");
    EXPECT_EQ(std::string_view(bytes + offsets[2], offsets[3] - offsets[2]), "carol");

    // Roundtrip through import.
    QColumn* imported = q_column_import_arrow(eng, &schema, &array);
    ASSERT_NE(imported, nullptr);
    array.release(&array);
    schema.release(&schema);

    QColumn* cols[1] = {imported};
    QBatch* batch = q_batch_new(eng, cols, 1);
    for (size_t i = 0; i < 3; ++i) {
        const char* p = nullptr;
        int64_t n = q_batch_col_string_at(batch, 0, i, &p);
        EXPECT_EQ(std::string_view(p, static_cast<size_t>(n)), names[i]);
    }
    q_destroy_engine(eng);
}

TEST(QuantumArrow, UnsupportedTypeReturnsError) {
    QEngine* eng = q_create_engine(1024);
    // We don't currently expose a non-int64 / non-string column builder via
    // the C API. So instead, build an Arrow array with an unsupported format
    // and try to import it.
    ArrowSchema schema{};
    schema.format = "f";   // float32, not supported in this slice
    ArrowArray array{};
    array.length = 0;
    array.n_buffers = 0;

    QColumn* col = q_column_import_arrow(eng, &schema, &array);
    EXPECT_EQ(col, nullptr);
    EXPECT_NE(std::string(q_last_error()).find("format"), std::string::npos);

    q_destroy_engine(eng);
}

} // namespace
