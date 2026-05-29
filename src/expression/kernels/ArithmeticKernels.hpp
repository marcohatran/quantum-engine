#pragma once

#include "vector/ColumnBatch.hpp"
#include "vector/ColumnVector.hpp"

namespace quantum::kernels {

enum class ArithOp { Add, Sub, Mul };

// Element-wise arithmetic. Inputs are evaluated children of a BinaryExpr —
// they have length batch.numRows(); the active rows are given by
// batch.positions(). Outputs are written into the same set of active slots;
// other slots are left undefined and marked invalid.
ColumnVector applyArithmetic(ArithOp                op,
                             DataType               type,
                             const ColumnVector&    lhs,
                             const ColumnVector&    rhs,
                             const PositionList&    pos,
                             MemoryManager*         mm);

} // namespace quantum::kernels
