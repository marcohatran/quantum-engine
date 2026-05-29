#pragma once

#include <memory>
#include <optional>

#include "common/DataType.hpp"
#include "common/Macros.hpp"
#include "vector/ColumnBatch.hpp"

namespace quantum {

class MemoryManager;

// Pull-based execution operator. Lifecycle: open() once, getNext() until empty,
// close() once.
class Operator {
public:
    virtual ~Operator() = default;

    virtual void                       open()  = 0;
    virtual std::optional<ColumnBatch> getNext() = 0;
    virtual void                       close() = 0;
    virtual const Schema&              outputSchema() const = 0;

protected:
    explicit Operator(MemoryManager* mm) : mm_(mm) {}
    MemoryManager* mm_;
};

using OperatorPtr = std::unique_ptr<Operator>;

} // namespace quantum
