#include "expression/ColumnRef.hpp"

#include <cstring>

#include <fmt/format.h>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

ColumnVector ColumnRef::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    if (index_ >= batch.numColumns()) {
        raise(ErrorCode::InvalidArgument,
              "ColumnRef: index {} out of range (numColumns={})",
              index_, batch.numColumns());
    }

    const ColumnVector& src = batch.column(index_);
    ColumnVector out = ColumnVector::make(src.type(), src.numRows(), mm);
    std::memcpy(out.values().data(),   src.values().data(),   src.values().size());
    std::memcpy(out.validity().data(), src.validity().data(), src.validity().sizeBytes());
    return out;
}

std::string ColumnRef::toString() const {
    return fmt::format("ColumnRef({})", index_);
}

} // namespace quantum
