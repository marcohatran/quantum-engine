#include "expression/ColumnRef.hpp"

#include <fmt/format.h>

#include "common/Status.hpp"

namespace quantum {

ColumnVector ColumnRef::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    if (index_ >= batch.numColumns()) {
        raise(ErrorCode::InvalidArgument,
              "ColumnRef: index {} out of range (numColumns={})",
              index_, batch.numColumns());
    }
    (void)mm;
    // Zero-copy: the caller consumes the result within the lifetime of
    // `batch` (the expression-eval contract); cloning here cost a full
    // column copy per batch on every filter/aggregate input.
    return ColumnVector::view(batch.column(index_));
}

std::string ColumnRef::toString() const {
    return fmt::format("ColumnRef({})", index_);
}

} // namespace quantum
