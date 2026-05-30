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
    return batch.column(index_).clone(mm);
}

std::string ColumnRef::toString() const {
    return fmt::format("ColumnRef({})", index_);
}

} // namespace quantum
