#pragma once

#include "vector/ColumnVector.hpp"
#include "vector/PositionList.hpp"

namespace quantum {
class MemoryManager;
}

namespace quantum::kernels {

enum class LogicalOp { And, Or };

// Element-wise Kleene-logic AND / OR on two BOOL ColumnVectors. Output
// is a BOOL ColumnVector storing one uint8_t (0/1) per row.
//
// SQL three-valued logic:
//   AND:  T AND T = T,  T AND F = F,  T AND NULL = NULL,
//         F AND NULL = F,  NULL AND NULL = NULL.
//   OR:   T OR  T = T,  T OR  F = T,  F OR  F  = F,
//         T OR NULL = T, F OR NULL = NULL, NULL OR NULL = NULL.
ColumnVector applyLogical(LogicalOp           op,
                          const ColumnVector& lhs,
                          const ColumnVector& rhs,
                          const PositionList& pos,
                          MemoryManager*      mm);

} // namespace quantum::kernels
