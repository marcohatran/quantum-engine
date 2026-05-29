#pragma once

#include <cstddef>
#include <vector>

#include "common/Macros.hpp"
#include "vector/ColumnVector.hpp"
#include "vector/PositionList.hpp"

namespace quantum {

class MemoryManager;

// A batch of column-aligned data. Carries a PositionList so filters can
// remove indices without physically compacting any values buffer.
class ColumnBatch {
public:
    ColumnBatch() noexcept = default;
    ColumnBatch(std::vector<ColumnVector> cols, PositionList pos) noexcept;

    QUANTUM_DISALLOW_COPY(ColumnBatch);
    ColumnBatch(ColumnBatch&&) noexcept            = default;
    ColumnBatch& operator=(ColumnBatch&&) noexcept = default;

    size_t numColumns() const noexcept { return cols_.size(); }
    size_t numRows()    const noexcept;
    size_t activeRows() const noexcept { return positions_.size(); }

    ColumnVector&       column(size_t i)       noexcept { return cols_[i]; }
    const ColumnVector& column(size_t i) const noexcept { return cols_[i]; }

    const PositionList& positions() const noexcept { return positions_; }
    void setPositions(PositionList p) noexcept { positions_ = std::move(p); }

    std::vector<ColumnVector>&       columns()       noexcept { return cols_; }
    const std::vector<ColumnVector>& columns() const noexcept { return cols_; }

    // Materializes the position list: allocates dense column vectors and
    // copies the active rows into them. Used at output boundaries / tests.
    ColumnBatch selectActive(MemoryManager* mm) const;

private:
    std::vector<ColumnVector> cols_;
    PositionList              positions_;
};

} // namespace quantum
