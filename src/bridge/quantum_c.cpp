// Implementation of the C ABI declared in include/quantum/quantum_c.h.
//
// Each opaque handle is a thin wrapper that owns one C++ object. Exceptions
// from the C++ side are converted to a recorded thread-local error string +
// a NULL return / no-op; callers query q_last_error() for diagnostics.

#include "quantum/quantum_c.h"

#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "bridge/BridgeError.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/Operator.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"
#include "plan/PlanProto.hpp"
#include "vector/ColumnBatch.hpp"
#include "vector/ColumnVector.hpp"
#include "vector/StringBuilder.hpp"

using namespace quantum;
using namespace quantum::plan;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

struct QEngine {
    std::unique_ptr<MemoryManager> mm;
    SourceCatalog                  catalog;
};

struct QColumn {
    ColumnVector vec;
};

struct QBatch {
    ColumnBatch batch;
};

struct QResult {
    OperatorPtr                op;
    std::optional<ColumnBatch> current;
    QBatch                     viewHandle;  // reused across q_result_next calls
};

namespace {

thread_local std::string g_lastError;

template <typename F>
auto guarded(F&& f, decltype(f()) onError) noexcept -> decltype(f()) {
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

} // namespace

namespace quantum::bridge {
void setLastError(std::string msg) noexcept { g_lastError = std::move(msg); }
const char* lastError() noexcept { return g_lastError.c_str(); }
} // namespace quantum::bridge

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

extern "C" QEngine* q_create_engine(size_t budget_bytes) {
    return guarded([&]() -> QEngine* {
        auto* eng = new QEngine();
        eng->mm = std::make_unique<MemoryManager>(budget_bytes);
        return eng;
    }, nullptr);
}

extern "C" void q_destroy_engine(QEngine* engine) {
    delete engine;
}

extern "C" const char* q_last_error(void) {
    return quantum::bridge::lastError();
}

// ---------------------------------------------------------------------------
// Column / batch construction
// ---------------------------------------------------------------------------

extern "C" QColumn* q_column_int64(QEngine*       engine,
                                   const int64_t* data,
                                   const uint8_t* validity,
                                   size_t         num_rows) {
    return guarded([&]() -> QColumn* {
        if (!engine || (!data && num_rows > 0)) {
            raise(ErrorCode::InvalidArgument, "q_column_int64: null input");
        }
        auto col = ColumnVector::make(Int64, num_rows, engine->mm.get());
        if (num_rows > 0) {
            std::memcpy(col.values().data(), data, num_rows * sizeof(int64_t));
        }
        col.validity().setAllValid();
        if (validity != nullptr) {
            for (size_t i = 0; i < num_rows; ++i) {
                col.validity().setValid(i, validity[i] != 0);
            }
        }
        auto* out = new QColumn{std::move(col)};
        return out;
    }, nullptr);
}

extern "C" QColumn* q_column_string(QEngine*           engine,
                                    const char* const* values,
                                    const int32_t*     lens,
                                    const uint8_t*     validity,
                                    size_t             num_rows) {
    return guarded([&]() -> QColumn* {
        if (!engine || (!values && num_rows > 0) || (!lens && num_rows > 0)) {
            raise(ErrorCode::InvalidArgument, "q_column_string: null input");
        }
        StringBuilder sb(engine->mm.get());
        for (size_t i = 0; i < num_rows; ++i) {
            if (validity != nullptr && validity[i] == 0) {
                sb.appendNull();
            } else {
                sb.append(std::string_view(values[i],
                                           static_cast<size_t>(lens[i])));
            }
        }
        auto* out = new QColumn{sb.build()};
        return out;
    }, nullptr);
}

extern "C" QBatch* q_batch_new(QEngine* engine, QColumn** cols, size_t num_cols) {
    return guarded([&]() -> QBatch* {
        if (!engine || (!cols && num_cols > 0)) {
            raise(ErrorCode::InvalidArgument, "q_batch_new: null input");
        }
        std::vector<ColumnVector> moved;
        moved.reserve(num_cols);
        size_t rows = 0;
        for (size_t i = 0; i < num_cols; ++i) {
            if (cols[i] == nullptr) {
                raise(ErrorCode::InvalidArgument, "q_batch_new: column {} is null", i);
            }
            if (i == 0) rows = cols[i]->vec.numRows();
            else if (cols[i]->vec.numRows() != rows) {
                raise(ErrorCode::InvalidArgument,
                      "q_batch_new: column {} has {} rows, expected {}",
                      i, cols[i]->vec.numRows(), rows);
            }
            moved.push_back(std::move(cols[i]->vec));
            delete cols[i];
            cols[i] = nullptr;
        }
        auto* out = new QBatch{ColumnBatch(std::move(moved), PositionList::identity(rows))};
        return out;
    }, nullptr);
}

extern "C" void q_register_source(QEngine*    engine,
                                  const char* source_id,
                                  QBatch**    batches,
                                  size_t      num_batches) {
    guarded([&]() -> int {
        if (!engine || !source_id) {
            raise(ErrorCode::InvalidArgument, "q_register_source: null input");
        }
        auto& sink = engine->catalog[source_id];
        sink.reserve(sink.size() + num_batches);
        for (size_t i = 0; i < num_batches; ++i) {
            if (batches[i] == nullptr) {
                raise(ErrorCode::InvalidArgument,
                      "q_register_source: batch {} is null", i);
            }
            sink.push_back(std::move(batches[i]->batch));
            delete batches[i];
            batches[i] = nullptr;
        }
        return 0;
    }, 0);
}

// ---------------------------------------------------------------------------
// Plan execution
// ---------------------------------------------------------------------------

extern "C" QResult* q_execute(QEngine*       engine,
                              const uint8_t* plan_bytes,
                              size_t         plan_len) {
    return guarded([&]() -> QResult* {
        if (!engine || (!plan_bytes && plan_len > 0)) {
            raise(ErrorCode::InvalidArgument, "q_execute: null input");
        }
        auto node = planFromProtoBytes(std::string_view(
            reinterpret_cast<const char*>(plan_bytes), plan_len));
        PlanBuilder pb(engine->mm.get());
        auto op = pb.build(node, engine->catalog);
        op->open();
        auto* out = new QResult();
        out->op = std::move(op);
        return out;
    }, nullptr);
}

extern "C" void q_destroy_result(QResult* result) {
    if (result && result->op) {
        try { result->op->close(); } catch (...) {}
    }
    delete result;
}

extern "C" const QBatch* q_result_next(QResult* result) {
    return guarded([&]() -> const QBatch* {
        if (!result) {
            raise(ErrorCode::InvalidArgument, "q_result_next: null result");
        }
        result->current = result->op->getNext();
        if (!result->current.has_value()) return nullptr;
        result->viewHandle.batch = std::move(*result->current);
        result->current.reset();
        return &result->viewHandle;
    }, nullptr);
}

// ---------------------------------------------------------------------------
// Batch / column inspection
// ---------------------------------------------------------------------------

extern "C" size_t q_batch_num_rows(const QBatch* b) {
    return b ? b->batch.numRows() : 0;
}

extern "C" size_t q_batch_num_cols(const QBatch* b) {
    return b ? b->batch.numColumns() : 0;
}

extern "C" q_type_t q_batch_col_type(const QBatch* b, size_t col) {
    if (!b) return Q_TYPE_INT64;
    return static_cast<q_type_t>(b->batch.column(col).type().id);
}

extern "C" const int64_t* q_batch_col_int64(const QBatch* b, size_t col) {
    if (!b) return nullptr;
    const auto& c = b->batch.column(col);
    if (c.type().id != TypeId::INT64) return nullptr;
    return c.typed<int64_t>();
}

extern "C" const double* q_batch_col_float64(const QBatch* b, size_t col) {
    if (!b) return nullptr;
    const auto& c = b->batch.column(col);
    if (c.type().id != TypeId::FLOAT64) return nullptr;
    return c.typed<double>();
}

extern "C" int q_batch_col_valid(const QBatch* b, size_t col, size_t row) {
    if (!b) return 0;
    return b->batch.column(col).validity().isValid(row) ? 1 : 0;
}

extern "C" int64_t q_batch_col_string_at(const QBatch* b, size_t col, size_t row,
                                         const char** out_ptr) {
    if (!b || !out_ptr) return -1;
    const auto& c = b->batch.column(col);
    if (c.type().id != TypeId::STRING) return -1;
    auto sv = c.stringAt(row);
    *out_ptr = sv.data();
    return static_cast<int64_t>(sv.size());
}
