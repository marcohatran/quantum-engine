// Arrow C Data Interface bridge for Quantum Engine.
//
// Defines the canonical ArrowSchema + ArrowArray structs (exact wire-level
// match for https://arrow.apache.org/docs/format/CDataInterface.html) plus
// two convenience functions for moving QColumn data across the boundary:
//
//   q_column_export_arrow  — populate an ArrowSchema + ArrowArray that
//                            point INTO an existing QColumn's buffers.
//                            Zero-copy. The QColumn must outlive the array.
//                            The release callbacks are valid no-ops; calling
//                            them is required for forward-compat with Arrow
//                            consumers but doesn't free anything here.
//
//   q_column_import_arrow  — deep-copy an Arrow array into a new QColumn
//                            owned by the engine. The caller may release
//                            the Arrow array immediately after this returns.
//
// Phase 4.4 supports `int64` ("l") and `utf8` ("u"). Other fixed-width
// types and dict-encoded export are follow-ups.

#ifndef QUANTUM_QUANTUM_ARROW_H
#define QUANTUM_QUANTUM_ARROW_H

#include <stddef.h>
#include <stdint.h>

#include "quantum/quantum_c.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Arrow C Data Interface (verbatim from the Arrow specification)
// =============================================================================

struct ArrowSchema {
    // Array type description.
    const char* format;
    const char* name;
    const char* metadata;
    int64_t     flags;
    int64_t     n_children;
    struct ArrowSchema** children;
    struct ArrowSchema*  dictionary;

    // Release callback.
    void (*release)(struct ArrowSchema*);
    // Opaque producer-specific data.
    void* private_data;
};

struct ArrowArray {
    // Array data description.
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void**       buffers;
    struct ArrowArray**children;
    struct ArrowArray* dictionary;

    // Release callback.
    void (*release)(struct ArrowArray*);
    // Opaque producer-specific data.
    void* private_data;
};

// =============================================================================
// Conversion entry points
// =============================================================================

// Zero-copy export. Out parameters must point to caller-allocated ArrowSchema
// and ArrowArray; on return they hold valid (release-callable) descriptors
// that reference the QColumn's underlying buffers. The QColumn must outlive
// any access via the returned descriptors.
//
// Returns 0 on success, -1 on error (q_last_error() for details).
int q_column_export_arrow(const QColumn*       col,
                          struct ArrowSchema*  out_schema,
                          struct ArrowArray*   out_array);

// Deep-copy import. Returns NULL on error (q_last_error() for details).
// The caller may release(*array) and release(*schema) immediately after.
QColumn* q_column_import_arrow(QEngine*                   engine,
                               const struct ArrowSchema*  schema,
                               const struct ArrowArray*   array);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_QUANTUM_ARROW_H
