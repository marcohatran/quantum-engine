#pragma once

#include "vector/ColumnBatch.hpp"
#include "vector/ColumnVector.hpp"

namespace quantum::kernels {

enum class CmpOp { Eq, Lt, Gt };

// Element-wise comparison. Output is a BOOL ColumnVector storing one
// uint8_t per row (1/0). Inactive slots are left undefined and invalid.
ColumnVector applyComparison(CmpOp                  op,
                             DataType               operandType,
                             const ColumnVector&    lhs,
                             const ColumnVector&    rhs,
                             const PositionList&    pos,
                             MemoryManager*         mm);

} // namespace quantum::kernels
