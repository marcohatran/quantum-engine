// Arrow IPC transport — see include/quantum/quantum_ipc.h.

#include "quantum/quantum_ipc.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>

#include <fmt/format.h>

#include "bridge/BridgeError.hpp"
#include "bridge/Handles.hpp"
#include "io/ArrowConvert.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/ColumnVector.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

namespace {

template <typename F>
auto guardedIpc(F&& f, decltype(f()) onError) noexcept -> decltype(f()) {
    try {
        bridge::setLastError("");
        return f();
    } catch (const QuantumError& e) {
        bridge::setLastError(e.what());
    } catch (const std::exception& e) {
        bridge::setLastError(e.what());
    } catch (...) {
        bridge::setLastError("unknown C++ exception");
    }
    return onError;
}

// ---------------------------------------------------------------------------
// ColumnBatch -> Arrow RecordBatch (positions must be identity)
// ---------------------------------------------------------------------------

std::shared_ptr<arrow::RecordBatch> toArrow(const ColumnBatch& batch) {
    const size_t nRows = batch.numRows();
    const size_t nCols = batch.numColumns();

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(nCols);
    arrays.reserve(nCols);

    // The engine's column layout IS Arrow's: LSB-first 1=valid validity
    // bitmaps, contiguous fixed-width values, int32 offsets + bytes for
    // strings. Wrap the buffers (non-owning — the IPC writer copies them
    // into the output stream while `batch` is alive) instead of rebuilding
    // arrays row by row.
    auto wrap = [](const void* data, size_t size) {
        return std::make_shared<arrow::Buffer>(
            static_cast<const uint8_t*>(data), static_cast<int64_t>(size));
    };

    auto finishOrThrow = [](arrow::ArrayBuilder& b) {
        std::shared_ptr<arrow::Array> a;
        auto st = b.Finish(&a);
        if (!st.ok()) {
            raise(ErrorCode::Internal,
                  "q_result_next_ipc: array finish failed: {}", st.ToString());
        }
        return a;
    };

    for (size_t c = 0; c < nCols; ++c) {
        const auto& col      = batch.column(c);
        const auto  name     = fmt::format("c{}", c);
        const auto  validity = wrap(col.validity().data(),
                                    col.validity().sizeBytes());
        switch (col.type().id) {
            case TypeId::INT64: {
                auto data = arrow::ArrayData::Make(
                    arrow::int64(), static_cast<int64_t>(nRows),
                    {validity, wrap(col.values().data(), nRows * sizeof(int64_t))},
                    arrow::kUnknownNullCount);
                fields.push_back(arrow::field(name, arrow::int64()));
                arrays.push_back(arrow::MakeArray(data));
                break;
            }
            case TypeId::INT32: {
                auto data = arrow::ArrayData::Make(
                    arrow::int32(), static_cast<int64_t>(nRows),
                    {validity, wrap(col.values().data(), nRows * sizeof(int32_t))},
                    arrow::kUnknownNullCount);
                fields.push_back(arrow::field(name, arrow::int32()));
                arrays.push_back(arrow::MakeArray(data));
                break;
            }
            case TypeId::FLOAT64: {
                auto data = arrow::ArrayData::Make(
                    arrow::float64(), static_cast<int64_t>(nRows),
                    {validity, wrap(col.values().data(), nRows * sizeof(double))},
                    arrow::kUnknownNullCount);
                fields.push_back(arrow::field(name, arrow::float64()));
                arrays.push_back(arrow::MakeArray(data));
                break;
            }
            case TypeId::STRING: {
                if (!col.isDictEncoded()) {
                    const auto* off = col.offsets().as<int32_t>();
                    auto data = arrow::ArrayData::Make(
                        arrow::utf8(), static_cast<int64_t>(nRows),
                        {validity,
                         wrap(off, (nRows + 1) * sizeof(int32_t)),
                         wrap(col.values().data(),
                              static_cast<size_t>(off[nRows]))},
                        arrow::kUnknownNullCount);
                    fields.push_back(arrow::field(name, arrow::utf8()));
                    arrays.push_back(arrow::MakeArray(data));
                    break;
                }
                // Dict-encoded: decode through the builder (rare at the
                // result boundary; dict export is a follow-up).
                arrow::StringBuilder b;
                for (size_t r = 0; r < nRows; ++r) {
                    if (col.validity().isValid(r)) {
                        const auto sv = col.stringAt(r);
                        (void)b.Append(sv.data(), static_cast<int32_t>(sv.size()));
                    } else {
                        (void)b.AppendNull();
                    }
                }
                fields.push_back(arrow::field(name, arrow::utf8()));
                arrays.push_back(finishOrThrow(b));
                break;
            }
            case TypeId::BOOL: {
                // Engine bools are 1 byte/row; Arrow's are bit-packed —
                // builder path.
                arrow::BooleanBuilder b;
                for (size_t r = 0; r < nRows; ++r) {
                    if (col.validity().isValid(r)) (void)b.Append(col.typed<uint8_t>()[r] != 0);
                    else                           (void)b.AppendNull();
                }
                fields.push_back(arrow::field(name, arrow::boolean()));
                arrays.push_back(finishOrThrow(b));
                break;
            }
            default:
                raise(ErrorCode::NotImplemented,
                      "q_result_next_ipc: unsupported column type {}",
                      col.type().toString());
        }
    }
    return arrow::RecordBatch::Make(arrow::schema(fields),
                                    static_cast<int64_t>(nRows), arrays);
}

} // namespace

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" int q_register_source_ipc(QEngine*       engine,
                                     const char*    source_id,
                                     const uint8_t* data,
                                     size_t         len) {
    return guardedIpc([&]() -> int {
        if (!engine || !source_id || !data) {
            raise(ErrorCode::InvalidArgument, "q_register_source_ipc: null input");
        }
        auto buffer = std::make_shared<arrow::io::BufferReader>(
            std::make_shared<arrow::Buffer>(data, static_cast<int64_t>(len)));
        auto reader_r = arrow::ipc::RecordBatchStreamReader::Open(buffer);
        if (!reader_r.ok()) {
            raise(ErrorCode::InvalidArgument,
                  "q_register_source_ipc: IPC open failed: {}",
                  reader_r.status().ToString());
        }
        auto reader = reader_r.ValueOrDie();

        auto& sink = engine->catalog[source_id];
        while (true) {
            auto rb_r = reader->Next();
            if (!rb_r.ok()) {
                raise(ErrorCode::InvalidArgument,
                      "q_register_source_ipc: IPC read failed: {}",
                      rb_r.status().ToString());
            }
            auto rb = rb_r.ValueOrDie();
            if (!rb) break;
            std::vector<ColumnVector> cols;
            cols.reserve(static_cast<size_t>(rb->num_columns()));
            for (int c = 0; c < rb->num_columns(); ++c) {
                cols.push_back(io::columnFromArrow(*rb->column(c), engine->mm.get()));
            }
            sink.emplace_back(std::move(cols),
                              PositionList::identity(
                                  static_cast<size_t>(rb->num_rows())));
        }
        return 0;
    }, -1);
}

extern "C" int q_result_next_ipc(QResult* result,
                                 uint8_t** out_data, size_t* out_len) {
    return guardedIpc([&]() -> int {
        if (!result || !out_data || !out_len) {
            raise(ErrorCode::InvalidArgument, "q_result_next_ipc: null input");
        }
        auto next = result->op->getNext();
        if (!next.has_value()) return 0;

        // Densify non-identity position lists at the boundary, same as
        // q_result_next — the IPC consumer iterates rows 0..n-1.
        ColumnBatch batch = std::move(*next);
        if (!batch.positions().isIdentity()) {
            batch = batch.selectActive(result->mm);
        }

        auto rb = toArrow(batch);
        auto sink_r = arrow::io::BufferOutputStream::Create();
        if (!sink_r.ok()) {
            raise(ErrorCode::Internal, "q_result_next_ipc: sink create failed");
        }
        auto sink = sink_r.ValueOrDie();
        auto writer_r = arrow::ipc::MakeStreamWriter(sink, rb->schema());
        if (!writer_r.ok()) {
            raise(ErrorCode::Internal,
                  "q_result_next_ipc: writer create failed: {}",
                  writer_r.status().ToString());
        }
        auto writer = writer_r.ValueOrDie();
        auto st = writer->WriteRecordBatch(*rb);
        if (st.ok()) st = writer->Close();
        if (!st.ok()) {
            raise(ErrorCode::Internal,
                  "q_result_next_ipc: IPC write failed: {}", st.ToString());
        }
        auto buf_r = sink->Finish();
        if (!buf_r.ok()) {
            raise(ErrorCode::Internal, "q_result_next_ipc: sink finish failed");
        }
        auto buf = buf_r.ValueOrDie();

        auto* out = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(buf->size())));
        if (!out) raise(ErrorCode::OutOfMemory, "q_result_next_ipc: malloc failed");
        std::memcpy(out, buf->data(), static_cast<size_t>(buf->size()));
        *out_data = out;
        *out_len  = static_cast<size_t>(buf->size());
        return 1;
    }, -1);
}

extern "C" void q_ipc_free(uint8_t* buf) { std::free(buf); }
