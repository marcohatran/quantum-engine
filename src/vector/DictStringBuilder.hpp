#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/Macros.hpp"
#include "vector/ColumnVector.hpp"

namespace quantum {

class MemoryManager;

// Constructs a dictionary-encoded STRING ColumnVector row-by-row.
//
// append(s) interns s into the dictionary (deduping via unordered_map) and
// pushes the assigned int32 code. appendNull() pushes a placeholder code
// (the value is undefined; only the validity bit matters). build() copies
// the dictionary bytes / offsets and the codes into off-heap Buffers and
// returns the dict-encoded ColumnVector.
//
// This is the canonical way to produce dict-encoded columns in Phase 3 —
// e.g. when ingesting a low-cardinality string column or normalizing during
// scan. Operators that pass dict-encoded data through gather / clone keep
// the existing dict (no re-encoding needed).
class DictStringBuilder {
public:
    explicit DictStringBuilder(MemoryManager* mm);

    QUANTUM_DISALLOW_COPY(DictStringBuilder);

    void append(std::string_view s);
    void appendNull();

    size_t numRows()      const noexcept { return codes_.size(); }
    size_t dictSize()     const noexcept { return dictOffsets_.size() - 1; }

    ColumnVector build();

private:
    int32_t internCode(std::string_view s);

    MemoryManager*       mm_;
    std::vector<int32_t> codes_;
    std::vector<bool>    validity_;

    // Dictionary state: bytes_ + dictOffsets_, indexed by code. dictIndex_
    // maps a string to its assigned code; we hold the strings here too so
    // string_view keys into the map remain valid as bytes_ grows.
    std::vector<char>                       bytes_;
    std::vector<int32_t>                    dictOffsets_;
    std::unordered_map<std::string, int32_t> dictIndex_;
};

} // namespace quantum
