// Arrow C Data Interface bridge — see include/quantum/quantum_arrow.h.

#include "quantum/quantum_arrow.h"
#include "quantum/quantum_c.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "bridge/BridgeError.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/ColumnVector.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;

// QEngine and QColumn are defined in quantum_c.cpp as friend-private types
// of the C ABI. Forward-declare equivalent layouts so the Arrow bridge can
// reach into them. (Keeping them as anonymous structs in quantum_c.cpp is
// the right module discipline; sharing the definition here is acceptable
// because both files together comprise the bridge layer.)
struct QEngine {
    std::unique_ptr<MemoryManager> mm;
    // (other fields private to quantum_c.cpp are not used here)
};

struct QColumn {
    ColumnVector vec;
};

namespace {

// Arrow expects an "is-aligned" validity bitmap as buffer 0 — but with
// null_count == 0 the buffer pointer may be NULL by spec. Our internal
// ValidityBitmask is always allocated (memset 0xFF on construction) so we
// pass it through unconditionally for safety.

// Arrow C Data Interface format strings (subset).
constexpr const char* kFormatInt64 = "l";
constexpr const char* kFormatUtf8  = "u";

// Release callbacks. Quantum data lives in MemoryManager Buffers that are
// freed when the engine dies; from the Arrow consumer's point of view there
// is nothing to release per-array beyond freeing the per-array scratch state
// allocated below.

struct ExportPrivateData {
    // 2 buffer slots for INT64 (validity, values),
    // 3 slots for UTF8 (validity, offsets, values).
    std::vector<const void*> buffers;
};

void releaseArray(ArrowArray* array) {
    if (!array || !array->release) return;
    auto* priv = static_cast<ExportPrivateData*>(array->private_data);
    delete priv;
    array->buffers      = nullptr;
    array->n_buffers    = 0;
    array->private_data = nullptr;
    array->release      = nullptr;
}

void releaseSchema(ArrowSchema* schema) {
    if (!schema || !schema->release) return;
    // We allocate `format` as a static C string (kFormat...), so don't free.
    schema->release = nullptr;
}

template <typename F>
auto guardedArrow(F&& f, decltype(f()) onError) noexcept -> decltype(f()) {
    try {
        bridge::setLastError("");
        return f();
    } catch (const QuantumError& e) { bridge::setLastError(e.what()); }
      catch (const std::exception& e) { bridge::setLastError(e.what()); }
      catch (...) { bridge::setLastError("unknown C++ exception"); }
    return onError;
}

} // namespace

extern "C" int q_column_export_arrow(const QColumn*  col,
                                     ArrowSchema*    out_schema,
                                     ArrowArray*     out_array) {
    return guardedArrow([&]() -> int {
        if (!col || !out_schema || !out_array) {
            raise(ErrorCode::InvalidArgument, "q_column_export_arrow: null input");
        }

        const auto& v = col->vec;
        auto* priv = new ExportPrivateData();

        const char* format = nullptr;
        if (v.type().id == TypeId::INT64) {
            format = kFormatInt64;
            priv->buffers = {
                v.validity().data(),   // validity bitmap
                v.values().data(),     // int64 values
            };
            out_array->n_buffers = 2;
        } else if (v.type().id == TypeId::STRING) {
            if (v.isDictEncoded()) {
                delete priv;
                raise(ErrorCode::NotImplemented,
                      "q_column_export_arrow: dict-encoded STRING export not supported in Phase 4.4 (use a plain STRING column)");
            }
            format = kFormatUtf8;
            priv->buffers = {
                v.validity().data(),   // validity bitmap
                v.offsets().data(),    // int32 offsets
                v.values().data(),     // char data
            };
            out_array->n_buffers = 3;
        } else {
            delete priv;
            raise(ErrorCode::NotImplemented,
                  "q_column_export_arrow: type {} not supported yet",
                  v.type().toString());
        }

        // Schema
        out_schema->format       = format;
        out_schema->name         = nullptr;
        out_schema->metadata     = nullptr;
        out_schema->flags        = 0;
        out_schema->n_children   = 0;
        out_schema->children     = nullptr;
        out_schema->dictionary   = nullptr;
        out_schema->release      = releaseSchema;
        out_schema->private_data = nullptr;

        // Array
        out_array->length       = static_cast<int64_t>(v.numRows());
        out_array->null_count   = static_cast<int64_t>(v.validity().nullCount());
        out_array->offset       = 0;
        out_array->n_children   = 0;
        out_array->buffers      = priv->buffers.data();
        out_array->children     = nullptr;
        out_array->dictionary   = nullptr;
        out_array->release      = releaseArray;
        out_array->private_data = priv;
        return 0;
    }, -1);
}

extern "C" QColumn* q_column_import_arrow(QEngine*           engine,
                                          const ArrowSchema* schema,
                                          const ArrowArray*  array) {
    return guardedArrow([&]() -> QColumn* {
        if (!engine || !schema || !array) {
            raise(ErrorCode::InvalidArgument, "q_column_import_arrow: null input");
        }
        if (!schema->format) {
            raise(ErrorCode::InvalidArgument, "q_column_import_arrow: schema has no format");
        }
        if (array->offset != 0) {
            raise(ErrorCode::NotImplemented,
                  "q_column_import_arrow: nonzero array offset not yet supported");
        }

        const std::string_view fmt(schema->format);
        const auto n = static_cast<size_t>(array->length);

        if (fmt == kFormatInt64) {
            if (array->n_buffers < 2) {
                raise(ErrorCode::InvalidArgument,
                      "q_column_import_arrow: INT64 array needs 2 buffers, got {}",
                      array->n_buffers);
            }
            auto col = ColumnVector::make(Int64, n, engine->mm.get());
            if (n > 0) {
                std::memcpy(col.values().data(),
                            array->buffers[1],
                            n * sizeof(int64_t));
            }
            col.validity().setAllValid();
            if (array->buffers[0] != nullptr && array->null_count > 0) {
                // Arrow validity is a bit-packed bitmap, little-endian; the
                // engine's ValidityBitmask has the same shape so we copy then
                // mask off bits beyond `n` (handled by setValid loop here).
                const auto* src = static_cast<const uint8_t*>(array->buffers[0]);
                for (size_t i = 0; i < n; ++i) {
                    const bool v = ((src[i / 8] >> (i % 8)) & 1U) != 0;
                    col.validity().setValid(i, v);
                }
            }
            return new QColumn{std::move(col)};
        }

        if (fmt == kFormatUtf8) {
            if (array->n_buffers < 3) {
                raise(ErrorCode::InvalidArgument,
                      "q_column_import_arrow: UTF8 array needs 3 buffers, got {}",
                      array->n_buffers);
            }
            const auto* validity = static_cast<const uint8_t*>(array->buffers[0]);
            const auto* offsets  = static_cast<const int32_t*>(array->buffers[1]);
            const auto* values   = static_cast<const char*>(array->buffers[2]);

            StringBuilder sb(engine->mm.get());
            for (size_t i = 0; i < n; ++i) {
                const bool valid = (validity == nullptr) ? true
                    : ((validity[i / 8] >> (i % 8)) & 1U) != 0;
                if (!valid) { sb.appendNull(); continue; }
                const size_t beg = static_cast<size_t>(offsets[i]);
                const size_t end = static_cast<size_t>(offsets[i + 1]);
                sb.append(std::string_view(values + beg, end - beg));
            }
            return new QColumn{sb.build()};
        }

        raise(ErrorCode::NotImplemented,
              "q_column_import_arrow: format '{}' not supported", fmt);
    }, nullptr);
}
