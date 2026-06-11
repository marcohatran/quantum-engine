#pragma once

namespace arrow {
class Array;
class RecordBatch;
}

#include "vector/ColumnBatch.hpp"
#include "vector/ColumnVector.hpp"

namespace quantum {
class MemoryManager;
}

namespace quantum::io {

// Deep-copy one Arrow array into an engine ColumnVector, validity
// included. Null-free numeric arrays memcpy their value buffer; null-free
// string arrays copy offsets + bytes wholesale. Supported Arrow types:
// BOOL, INT32, INT64, DOUBLE, STRING, LARGE_STRING.
ColumnVector columnFromArrow(const arrow::Array& arr, MemoryManager* mm);

// Convert a whole RecordBatch (identity positions).
ColumnBatch batchFromArrow(const arrow::RecordBatch& rb, MemoryManager* mm);

} // namespace quantum::io
