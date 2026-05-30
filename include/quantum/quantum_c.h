// Quantum Engine C ABI.
//
// Stable extern "C" entry points for embedding the engine in another
// language. The JVM bridge (Phase 4.4) calls these via JNI / JNA / Panama;
// other hosts can use them directly.
//
// Lifetime model:
//   - QEngine owns a MemoryManager and a SourceCatalog. All other handles
//     created against an engine are released when the engine is destroyed.
//   - QColumn and QBatch handles are owned-by-engine; do not free them
//     individually. QResult is the exception — caller must q_destroy_result
//     when done iterating.
//   - All handles are opaque; callers may only pass them back into the C API.
//
// Error model: functions that can fail return either a NULL handle or a
// negative status code. q_last_error returns a human-readable description of
// the most recent failure on this thread (thread-local).

#ifndef QUANTUM_QUANTUM_C_H
#define QUANTUM_QUANTUM_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QEngine QEngine;
typedef struct QColumn QColumn;
typedef struct QBatch  QBatch;
typedef struct QResult QResult;

// Column type codes — match TypeId in common/DataType.hpp.
typedef enum {
  Q_TYPE_BOOL    = 0,
  Q_TYPE_INT32   = 1,
  Q_TYPE_INT64   = 2,
  Q_TYPE_FLOAT64 = 3,
  Q_TYPE_STRING  = 4
} q_type_t;

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

// Create a new engine with the given memory budget (bytes).
QEngine* q_create_engine(size_t budget_bytes);

// Free the engine and every handle created against it.
void q_destroy_engine(QEngine* engine);

// Human-readable description of the most recent failure on this thread.
// Returns an empty string when no error has been recorded.
const char* q_last_error(void);

// ---------------------------------------------------------------------------
// Column / batch construction
//
// All inputs are deep-copied into engine-owned memory; caller retains
// ownership of the source buffers.
// ---------------------------------------------------------------------------

// Build an int64 column. `validity` is a per-row uint8 (0 or 1); pass NULL
// for all-valid.
QColumn* q_column_int64(QEngine* engine,
                        const int64_t* data,
                        const uint8_t* validity,
                        size_t         num_rows);

// Build a STRING column. `values[i]` is the bytes for row i with length
// `lens[i]`. Strings need not be NUL-terminated. `validity` per-row.
QColumn* q_column_string(QEngine*           engine,
                         const char* const* values,
                         const int32_t*     lens,
                         const uint8_t*     validity,
                         size_t             num_rows);

// Build a batch from `num_cols` engine-owned columns (consumed). The columns
// must all share the same num_rows; the engine takes ownership of the column
// pointers — do not use them after this call.
QBatch* q_batch_new(QEngine* engine, QColumn** cols, size_t num_cols);

// Register a vector of batches under `source_id`. The engine takes ownership
// of every passed-in QBatch* (consumed). Subsequent plan executions will see
// this source when their ScanNode.source_id matches.
void q_register_source(QEngine*    engine,
                       const char* source_id,
                       QBatch**    batches,
                       size_t      num_batches);

// ---------------------------------------------------------------------------
// Plan execution
// ---------------------------------------------------------------------------

// Execute a protobuf-serialized PlanNode against the registered sources.
// Returns NULL on error; q_last_error() describes the failure.
QResult* q_execute(QEngine*       engine,
                   const uint8_t* plan_bytes,
                   size_t         plan_len);

void q_destroy_result(QResult* result);

// Advance to the next output batch. Returns NULL at end-of-stream. The
// returned pointer is owned by the QResult and is invalidated by the next
// call to q_result_next or by q_destroy_result.
const QBatch* q_result_next(QResult* result);

// ---------------------------------------------------------------------------
// Batch / column inspection
// ---------------------------------------------------------------------------

size_t   q_batch_num_rows(const QBatch* b);
size_t   q_batch_num_cols(const QBatch* b);
q_type_t q_batch_col_type(const QBatch* b, size_t col);

// Pointer to the contiguous int64 values for column `col`. Returns NULL if
// the column is not INT64.
const int64_t* q_batch_col_int64(const QBatch* b, size_t col);

// Row-level validity check (1 = valid, 0 = NULL).
int q_batch_col_valid(const QBatch* b, size_t col, size_t row);

// Read a STRING cell. Returns the length and writes the bytes pointer into
// *out_ptr. Returns -1 if the column is not STRING.
int64_t q_batch_col_string_at(const QBatch* b, size_t col, size_t row,
                              const char** out_ptr);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_QUANTUM_C_H
