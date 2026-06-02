#include "operator/SortOperator.hpp"

#include <algorithm>
#include <cstring>
#include <string_view>
#include <utility>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

SortOperator::SortOperator(MemoryManager*               mm,
                           OperatorPtr                  child,
                           std::vector<ExpressionPtr>   keys,
                           std::vector<bool>            ascending,
                           std::vector<bool>            nullsFirst)
    : Operator(mm),
      child_(std::move(child)),
      keys_(std::move(keys)),
      ascending_(std::move(ascending)),
      nullsFirst_(std::move(nullsFirst)) {
    if (keys_.empty()) {
        raise(ErrorCode::InvalidArgument, "SortOperator: at least one sort key required");
    }
    if (ascending_.size() != keys_.size() || nullsFirst_.size() != keys_.size()) {
        raise(ErrorCode::InvalidArgument,
              "SortOperator: ascending/nullsFirst arrays must match key count");
    }
}

void SortOperator::open()  { child_->open(); }
void SortOperator::close() { child_->close(); }

namespace {

// Compare two indices into key columns. Returns negative if a<b, 0 if equal,
// positive if a>b. Handles null per nullsFirst.
int compareKey(const ColumnVector& col, size_t aIdx, size_t bIdx,
               bool nullsFirst) {
    const bool aValid = col.validity().isValid(aIdx);
    const bool bValid = col.validity().isValid(bIdx);
    if (!aValid && !bValid) return 0;
    if (!aValid) return nullsFirst ? -1 : 1;
    if (!bValid) return nullsFirst ? 1 : -1;

    switch (col.type().id) {
        case TypeId::INT64: {
            const int64_t a = col.typed<int64_t>()[aIdx];
            const int64_t b = col.typed<int64_t>()[bIdx];
            return (a < b) ? -1 : (a > b) ? 1 : 0;
        }
        case TypeId::INT32: {
            const int32_t a = col.typed<int32_t>()[aIdx];
            const int32_t b = col.typed<int32_t>()[bIdx];
            return (a < b) ? -1 : (a > b) ? 1 : 0;
        }
        case TypeId::FLOAT64: {
            const double a = col.typed<double>()[aIdx];
            const double b = col.typed<double>()[bIdx];
            return (a < b) ? -1 : (a > b) ? 1 : 0;
        }
        case TypeId::BOOL: {
            const uint8_t a = col.typed<uint8_t>()[aIdx];
            const uint8_t b = col.typed<uint8_t>()[bIdx];
            return (a < b) ? -1 : (a > b) ? 1 : 0;
        }
        case TypeId::STRING: {
            const std::string_view a = col.stringAt(aIdx);
            const std::string_view b = col.stringAt(bIdx);
            if (a < b) return -1;
            if (a > b) return  1;
            return 0;
        }
        default:
            raise(ErrorCode::NotImplemented,
                  "SortOperator: unsupported key type {}", col.type().toString());
    }
}

} // namespace

void SortOperator::materializeAndSort() {
    // Drain the child, evaluating each key per batch, and accumulate
    // dense columns covering ALL active rows from ALL batches.
    std::vector<ColumnBatch> inputBatches;
    while (auto b = child_->getNext()) {
        if (b->activeRows() == 0) continue;
        inputBatches.push_back(std::move(*b));
    }

    size_t totalRows = 0;
    for (const auto& b : inputBatches) totalRows += b.activeRows();

    if (inputBatches.empty()) {
        // Empty input — sorted_ stays empty.
        return;
    }

    const size_t nCols = inputBatches.front().numColumns();
    const auto&  childSchema = child_->outputSchema();

    // Concatenate all batches into one dense batch (active rows only).
    // For each column, walk every batch and copy active rows. Build the
    // output batch column-by-column.
    std::vector<ColumnVector> outCols;
    outCols.reserve(nCols);
    for (size_t c = 0; c < nCols; ++c) {
        const DataType ty = childSchema.field(c).type;
        if (ty.id == TypeId::STRING) {
            StringBuilder sb(mm_);
            for (const auto& b : inputBatches) {
                const auto& col = b.column(c);
                const auto& pos = b.positions();
                for (size_t k = 0; k < pos.size(); ++k) {
                    const auto i = static_cast<size_t>(pos.at(k));
                    if (!col.validity().isValid(i)) {
                        sb.appendNull();
                    } else {
                        sb.append(col.stringAt(i));
                    }
                }
            }
            outCols.push_back(sb.build());
        } else {
            ColumnVector out = ColumnVector::make(ty, totalRows, mm_);
            out.validity().setAllValid();
            size_t writeIdx = 0;
            for (const auto& b : inputBatches) {
                const auto& col = b.column(c);
                const auto& pos = b.positions();
                for (size_t k = 0; k < pos.size(); ++k) {
                    const auto i = static_cast<size_t>(pos.at(k));
                    out.validity().setValid(writeIdx, col.validity().isValid(i));
                    switch (ty.id) {
                        case TypeId::INT64:
                            out.typed<int64_t>()[writeIdx] = col.typed<int64_t>()[i]; break;
                        case TypeId::INT32:
                            out.typed<int32_t>()[writeIdx] = col.typed<int32_t>()[i]; break;
                        case TypeId::FLOAT64:
                            out.typed<double>()[writeIdx] = col.typed<double>()[i]; break;
                        case TypeId::BOOL:
                            out.typed<uint8_t>()[writeIdx] = col.typed<uint8_t>()[i]; break;
                        default:
                            raise(ErrorCode::NotImplemented,
                                  "SortOperator: unsupported col type {}", ty.toString());
                    }
                    ++writeIdx;
                }
            }
            outCols.push_back(std::move(out));
        }
    }

    ColumnBatch dense(std::move(outCols), PositionList::identity(totalRows));

    // Evaluate keys against the dense batch — each key becomes a ColumnVector
    // indexed 0..totalRows-1.
    std::vector<ColumnVector> keyCols;
    keyCols.reserve(keys_.size());
    for (const auto& k : keys_) keyCols.push_back(k->eval(dense, mm_));

    // Sort an index permutation 0..totalRows-1 using the keys.
    std::vector<int32_t> perm(totalRows);
    for (size_t i = 0; i < totalRows; ++i) perm[i] = static_cast<int32_t>(i);
    std::sort(perm.begin(), perm.end(), [&](int32_t a, int32_t b) {
        for (size_t kIdx = 0; kIdx < keys_.size(); ++kIdx) {
            const int cmp = compareKey(keyCols[kIdx],
                                       static_cast<size_t>(a),
                                       static_cast<size_t>(b),
                                       nullsFirst_[kIdx]);
            if (cmp == 0) continue;
            const bool asc = ascending_[kIdx];
            if (asc) return cmp < 0;
            else     return cmp > 0;
        }
        return false;
    });

    // Permute the dense batch's columns according to perm. We do this by
    // building a fresh batch with rows in `perm` order.
    std::vector<ColumnVector> sortedCols;
    sortedCols.reserve(nCols);
    for (size_t c = 0; c < nCols; ++c) {
        const DataType ty = childSchema.field(c).type;
        const auto& src   = dense.column(c);
        if (ty.id == TypeId::STRING) {
            StringBuilder sb(mm_);
            for (size_t k = 0; k < totalRows; ++k) {
                const auto i = static_cast<size_t>(perm[k]);
                if (!src.validity().isValid(i)) sb.appendNull();
                else                            sb.append(src.stringAt(i));
            }
            sortedCols.push_back(sb.build());
        } else {
            ColumnVector out = ColumnVector::make(ty, totalRows, mm_);
            out.validity().setAllValid();
            for (size_t k = 0; k < totalRows; ++k) {
                const auto i = static_cast<size_t>(perm[k]);
                out.validity().setValid(k, src.validity().isValid(i));
                switch (ty.id) {
                    case TypeId::INT64:
                        out.typed<int64_t>()[k] = src.typed<int64_t>()[i]; break;
                    case TypeId::INT32:
                        out.typed<int32_t>()[k] = src.typed<int32_t>()[i]; break;
                    case TypeId::FLOAT64:
                        out.typed<double>()[k] = src.typed<double>()[i]; break;
                    case TypeId::BOOL:
                        out.typed<uint8_t>()[k] = src.typed<uint8_t>()[i]; break;
                    default: break;
                }
            }
            sortedCols.push_back(std::move(out));
        }
    }
    sorted_ = ColumnBatch(std::move(sortedCols), PositionList::identity(totalRows));
}

std::optional<ColumnBatch> SortOperator::getNext() {
    if (emitted_) return std::nullopt;
    materializeAndSort();
    emitted_ = true;
    if (sorted_.numColumns() == 0 || sorted_.numRows() == 0) return std::nullopt;
    return std::move(sorted_);
}

} // namespace quantum
