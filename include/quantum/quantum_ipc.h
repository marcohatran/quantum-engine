// Arrow IPC transport for Quantum Engine.
//
// Moves whole batches across the C ABI as Arrow IPC streams instead of
// per-column / per-row calls. One JNA crossing per direction: the host
// serializes its source data to an IPC stream (schema + record batches)
// and registers it in one call; each result batch comes back as one IPC
// stream buffer the host parses with its own Arrow library.
//
// Supported column types both directions: int64, int32, float64, bool,
// utf8 (and large_utf8 on input).

#ifndef QUANTUM_QUANTUM_IPC_H
#define QUANTUM_QUANTUM_IPC_H

#include <stddef.h>
#include <stdint.h>

#include "quantum/quantum_c.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register `source_id` from an Arrow IPC stream (all record batches in the
// stream become the source's batches, deep-copied into engine memory).
// Returns 0 on success, -1 on error (q_last_error() for details).
int q_register_source_ipc(QEngine*       engine,
                          const char*    source_id,
                          const uint8_t* data,
                          size_t         len);

// Serialize the next result batch as a self-contained Arrow IPC stream
// (schema + one record batch; columns named c0..cN). On success returns 1
// and sets *out_data / *out_len — the caller owns the buffer and must
// release it with q_ipc_free. Returns 0 at end-of-stream, -1 on error.
int q_result_next_ipc(QResult* result, uint8_t** out_data, size_t* out_len);

void q_ipc_free(uint8_t* buf);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_QUANTUM_IPC_H
