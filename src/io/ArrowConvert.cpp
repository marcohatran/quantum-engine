#include "io/ArrowConvert.hpp"

#include <cstring>

#include <arrow/api.h>

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum::io {

namespace {

template <typename CType, typename ArrowArrayT>
ColumnVector fixedFromArrow(const ArrowArrayT& a, DataType type,
                            MemoryManager* mm) {
    const auto n = static_cast<size_t>(a.length());
    ColumnVector out = ColumnVector::make(type, n, mm);
    out.validity().setAllValid();
    if (a.null_count() == 0) {
        // raw_values() is already adjusted for the array offset.
        std::memcpy(out.values().data(), a.raw_values(), n * sizeof(CType));
        return out;
    }
    auto* dest = out.typed<CType>();
    for (size_t i = 0; i < n; ++i) {
        const auto j = static_cast<int64_t>(i);
        dest[i] = a.IsNull(j) ? CType{} : static_cast<CType>(a.Value(j));
        out.validity().setValid(i, !a.IsNull(j));
    }
    return out;
}

template <typename ArrowStringT>
ColumnVector stringFromArrow(const ArrowStringT& a, MemoryManager* mm) {
    const auto n = static_cast<size_t>(a.length());
    StringBuilder sb(mm);
    for (size_t i = 0; i < n; ++i) {
        const auto j = static_cast<int64_t>(i);
        if (a.IsNull(j)) {
            sb.appendNull();
        } else {
            const auto sv = a.GetView(j);
            sb.append(std::string_view(sv.data(), sv.size()));
        }
    }
    return sb.build();
}

// Null-free, offset-free STRING: copy the int32 offsets + byte buffers
// wholesale instead of re-appending row by row.
ColumnVector stringFromArrowFast(const arrow::StringArray& a,
                                 MemoryManager* mm) {
    const auto     n        = static_cast<size_t>(a.length());
    const int32_t* srcOff   = a.raw_value_offsets();
    const int32_t  base     = srcOff[0];
    const auto     numBytes = static_cast<size_t>(srcOff[n] - base);

    Buffer offsets = mm->allocate(roundUp((n + 1) * sizeof(int32_t), kCacheLineBytes));
    Buffer values  = mm->allocate(roundUp(numBytes, kCacheLineBytes));
    auto*  outOff  = offsets.as<int32_t>();
    if (base == 0) {
        std::memcpy(outOff, srcOff, (n + 1) * sizeof(int32_t));
    } else {
        for (size_t i = 0; i <= n; ++i) outOff[i] = srcOff[i] - base;
    }
    if (numBytes > 0) {
        std::memcpy(values.data(), a.raw_data() + base, numBytes);
    }
    ColumnVector out = ColumnVector::makeVariableWidth(
        DataType{TypeId::STRING}, n, std::move(offsets), std::move(values), mm);
    out.validity().setAllValid();
    return out;
}

} // namespace

ColumnVector columnFromArrow(const arrow::Array& arr, MemoryManager* mm) {
    switch (arr.type_id()) {
        case arrow::Type::INT64:
            return fixedFromArrow<int64_t, arrow::Int64Array>(
                static_cast<const arrow::Int64Array&>(arr), Int64, mm);
        case arrow::Type::INT32:
            return fixedFromArrow<int32_t, arrow::Int32Array>(
                static_cast<const arrow::Int32Array&>(arr), Int32, mm);
        case arrow::Type::DOUBLE:
            return fixedFromArrow<double, arrow::DoubleArray>(
                static_cast<const arrow::DoubleArray&>(arr), Float64, mm);
        case arrow::Type::BOOL: {
            // Bit-packed in Arrow — always per-row.
            const auto& a = static_cast<const arrow::BooleanArray&>(arr);
            const auto  n = static_cast<size_t>(a.length());
            ColumnVector out = ColumnVector::make(Bool, n, mm);
            out.validity().setAllValid();
            for (size_t i = 0; i < n; ++i) {
                const auto j = static_cast<int64_t>(i);
                out.typed<uint8_t>()[i] = (!a.IsNull(j) && a.Value(j)) ? 1 : 0;
                out.validity().setValid(i, !a.IsNull(j));
            }
            return out;
        }
        case arrow::Type::STRING: {
            const auto& a = static_cast<const arrow::StringArray&>(arr);
            if (a.null_count() == 0) return stringFromArrowFast(a, mm);
            return stringFromArrow(a, mm);
        }
        case arrow::Type::LARGE_STRING:
            return stringFromArrow(
                static_cast<const arrow::LargeStringArray&>(arr), mm);
        default:
            raise(ErrorCode::NotImplemented,
                  "ArrowConvert: unsupported Arrow type {}",
                  arr.type()->ToString());
    }
}

ColumnBatch batchFromArrow(const arrow::RecordBatch& rb, MemoryManager* mm) {
    std::vector<ColumnVector> cols;
    cols.reserve(static_cast<size_t>(rb.num_columns()));
    for (int c = 0; c < rb.num_columns(); ++c) {
        cols.push_back(columnFromArrow(*rb.column(c), mm));
    }
    return ColumnBatch(std::move(cols),
                       PositionList::identity(static_cast<size_t>(rb.num_rows())));
}

} // namespace quantum::io
