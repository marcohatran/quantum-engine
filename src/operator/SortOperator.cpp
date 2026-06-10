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
                           std::vector<bool>            nullsFirst,
                           size_t                       spillTriggerRows,
                           double                       spillFraction)
    : Operator(mm),
      child_(std::move(child)),
      keys_(std::move(keys)),
      ascending_(std::move(ascending)),
      nullsFirst_(std::move(nullsFirst)),
      spillTriggerRows_(spillTriggerRows),
      spillFraction_(spillFraction) {
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

// Spilled-run value encoding: [uint8 valid][payload]. Payload by type:
// INT64/FLOAT64 8 bytes, INT32 4, BOOL 1, STRING int32 len + bytes. Null
// values carry no payload.
void serializeValue(const ColumnVector& col, size_t row,
                    std::vector<uint8_t>& out) {
    const uint8_t valid = col.validity().isValid(row) ? 1 : 0;
    out.push_back(valid);
    if (!valid) return;
    switch (col.type().id) {
        case TypeId::INT64: {
            const int64_t v = col.typed<int64_t>()[row];
            const auto* p = reinterpret_cast<const uint8_t*>(&v);
            out.insert(out.end(), p, p + sizeof(v));
            break;
        }
        case TypeId::INT32: {
            const int32_t v = col.typed<int32_t>()[row];
            const auto* p = reinterpret_cast<const uint8_t*>(&v);
            out.insert(out.end(), p, p + sizeof(v));
            break;
        }
        case TypeId::FLOAT64: {
            const double v = col.typed<double>()[row];
            const auto* p = reinterpret_cast<const uint8_t*>(&v);
            out.insert(out.end(), p, p + sizeof(v));
            break;
        }
        case TypeId::BOOL:
            out.push_back(col.typed<uint8_t>()[row]);
            break;
        case TypeId::STRING: {
            const std::string_view s = col.stringAt(row);
            const auto len = static_cast<int32_t>(s.size());
            const auto* p = reinterpret_cast<const uint8_t*>(&len);
            out.insert(out.end(), p, p + sizeof(len));
            out.insert(out.end(), s.begin(), s.end());
            break;
        }
        default:
            raise(ErrorCode::NotImplemented,
                  "SortOperator: unsupported spill type {}", col.type().toString());
    }
}

template <typename T>
T readPod(const uint8_t*& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

} // namespace

ColumnBatch SortOperator::densifyBuffer() {
    size_t totalRows = 0;
    for (const auto& b : buffer_) totalRows += b.activeRows();

    const auto&  childSchema = child_->outputSchema();
    const size_t nCols       = childSchema.numFields();

    // Concatenate all buffered batches into one dense batch (active rows
    // only), column by column.
    std::vector<ColumnVector> outCols;
    outCols.reserve(nCols);
    for (size_t c = 0; c < nCols; ++c) {
        const DataType ty = childSchema.field(c).type;
        if (ty.id == TypeId::STRING) {
            StringBuilder sb(mm_);
            for (const auto& b : buffer_) {
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
            for (const auto& b : buffer_) {
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
    return ColumnBatch(std::move(outCols), PositionList::identity(totalRows));
}

std::vector<int32_t> SortOperator::sortedPermutation(
        const ColumnBatch&         dense,
        std::vector<ColumnVector>& keyColsOut) {
    const size_t totalRows = dense.numRows();
    keyColsOut.clear();
    keyColsOut.reserve(keys_.size());
    for (const auto& k : keys_) keyColsOut.push_back(k->eval(dense, mm_));

    std::vector<int32_t> perm(totalRows);
    for (size_t i = 0; i < totalRows; ++i) perm[i] = static_cast<int32_t>(i);
    std::sort(perm.begin(), perm.end(), [&](int32_t a, int32_t b) {
        for (size_t kIdx = 0; kIdx < keys_.size(); ++kIdx) {
            const int cmp = compareKey(keyColsOut[kIdx],
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
    return perm;
}

ColumnBatch SortOperator::permute(const ColumnBatch&          dense,
                                  const std::vector<int32_t>& perm) {
    const auto&  childSchema = child_->outputSchema();
    const size_t nCols       = childSchema.numFields();
    const size_t totalRows   = perm.size();

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
    return ColumnBatch(std::move(sortedCols), PositionList::identity(totalRows));
}

void SortOperator::spillRun() {
    if (bufferedRows_ == 0) return;

    ColumnBatch dense = densifyBuffer();
    buffer_.clear();
    bufferedRows_ = 0;

    std::vector<ColumnVector> keyCols;
    const std::vector<int32_t> perm = sortedPermutation(dense, keyCols);
    if (keyTypes_.empty()) {
        for (const auto& kc : keyCols) keyTypes_.push_back(kc.type());
    }

    const auto&  childSchema = child_->outputSchema();
    const size_t nCols       = childSchema.numFields();

    SpillFile run(/*stateBytes=*/0, "quantum_sort_run");
    std::vector<uint8_t> keyBuf, rowBuf;
    for (const int32_t idx : perm) {
        const auto row = static_cast<size_t>(idx);
        keyBuf.clear();
        rowBuf.clear();
        for (const auto& kc : keyCols) serializeValue(kc, row, keyBuf);
        for (size_t c = 0; c < nCols; ++c) {
            serializeValue(dense.column(c), row, rowBuf);
        }
        run.write(keyBuf.data(), static_cast<int32_t>(keyBuf.size()),
                  rowBuf.data(), rowBuf.size());
    }
    run.finishWriting();
    runs_.push_back(std::move(run));
}

void SortOperator::build() {
    while (auto b = child_->getNext()) {
        if (b->activeRows() == 0) continue;
        bufferedRows_ += b->activeRows();
        buffer_.push_back(std::move(*b));

        const bool overRows = spillTriggerRows_ > 0 &&
                              bufferedRows_ >= spillTriggerRows_;
        const bool overMem  = spillFraction_ > 0.0 &&
                              mm_->usedFraction() >= spillFraction_;
        if (overRows || overMem) spillRun();
    }

    if (runs_.empty()) {
        // In-memory path: sort the whole buffer and emit it as one batch.
        if (bufferedRows_ > 0) {
            ColumnBatch dense = densifyBuffer();
            buffer_.clear();
            std::vector<ColumnVector> keyCols;
            sorted_ = permute(dense, sortedPermutation(dense, keyCols));
        }
        return;
    }

    // External path: spill the tail buffer as the final run, then open a
    // reader + head per run for the k-way merge.
    spillRun();
    readers_.reserve(runs_.size());
    heads_.resize(runs_.size());
    for (size_t i = 0; i < runs_.size(); ++i) {
        readers_.push_back(runs_[i].reader());
        advanceHead(i);
    }
    merging_ = true;
}

void SortOperator::advanceHead(size_t runIdx) {
    auto& head = heads_[runIdx];
    if (!readers_[runIdx].next()) {
        head.clear();  // exhausted
        return;
    }
    head.resize(keyTypes_.size());
    const uint8_t* p = readers_[runIdx].key();
    for (size_t k = 0; k < keyTypes_.size(); ++k) {
        KeyValue& kv = head[k];
        kv.valid = readPod<uint8_t>(p) != 0;
        if (!kv.valid) continue;
        switch (keyTypes_[k].id) {
            case TypeId::INT64:   kv.i64 = readPod<int64_t>(p); break;
            case TypeId::INT32:   kv.i64 = readPod<int32_t>(p); break;
            case TypeId::BOOL:    kv.i64 = readPod<uint8_t>(p); break;
            case TypeId::FLOAT64: kv.f64 = readPod<double>(p);  break;
            case TypeId::STRING: {
                const auto len = readPod<int32_t>(p);
                kv.str.assign(reinterpret_cast<const char*>(p),
                              static_cast<size_t>(len));
                p += len;
                break;
            }
            default:
                raise(ErrorCode::NotImplemented,
                      "SortOperator: unsupported merge key type {}",
                      keyTypes_[k].toString());
        }
    }
}

// Returns negative when `a` sorts before `b` in the requested output order
// (ascending/descending and nulls placement already applied).
int SortOperator::compareHeads(const std::vector<KeyValue>& a,
                               const std::vector<KeyValue>& b) const {
    for (size_t k = 0; k < keyTypes_.size(); ++k) {
        const KeyValue& x = a[k];
        const KeyValue& y = b[k];
        int cmp = 0;
        if (!x.valid && !y.valid)      cmp = 0;
        else if (!x.valid)             cmp = nullsFirst_[k] ? -1 : 1;
        else if (!y.valid)             cmp = nullsFirst_[k] ? 1 : -1;
        else {
            switch (keyTypes_[k].id) {
                case TypeId::INT64:
                case TypeId::INT32:
                case TypeId::BOOL:
                    cmp = (x.i64 < y.i64) ? -1 : (x.i64 > y.i64) ? 1 : 0;
                    break;
                case TypeId::FLOAT64:
                    cmp = (x.f64 < y.f64) ? -1 : (x.f64 > y.f64) ? 1 : 0;
                    break;
                case TypeId::STRING:
                    cmp = (x.str < y.str) ? -1 : (x.str > y.str) ? 1 : 0;
                    break;
                default: break;
            }
        }
        if (cmp == 0) continue;
        return ascending_[k] ? cmp : -cmp;
    }
    return 0;
}

std::optional<ColumnBatch> SortOperator::nextMergedBatch() {
    constexpr size_t kMergeBatchRows = 4096;

    const auto&  childSchema = child_->outputSchema();
    const size_t nCols       = childSchema.numFields();

    // Per-column staging for one output batch.
    struct Stage {
        std::vector<uint8_t>     valid;
        std::vector<int64_t>     i64;
        std::vector<int32_t>     i32;
        std::vector<double>      f64;
        std::vector<uint8_t>     b8;
        std::vector<std::string> str;
    };
    std::vector<Stage> stages(nCols);

    size_t rows = 0;
    while (rows < kMergeBatchRows) {
        // Linear scan for the smallest head — the run count is the number
        // of spills, small enough that a heap isn't worth the bookkeeping.
        int best = -1;
        for (size_t i = 0; i < heads_.size(); ++i) {
            if (heads_[i].empty()) continue;
            if (best < 0 || compareHeads(heads_[i], heads_[static_cast<size_t>(best)]) < 0) {
                best = static_cast<int>(i);
            }
        }
        if (best < 0) break;  // all runs exhausted

        const uint8_t* p = readers_[static_cast<size_t>(best)].state();
        for (size_t c = 0; c < nCols; ++c) {
            Stage&        st    = stages[c];
            const uint8_t valid = readPod<uint8_t>(p);
            st.valid.push_back(valid);
            switch (childSchema.field(c).type.id) {
                case TypeId::INT64:
                    st.i64.push_back(valid ? readPod<int64_t>(p) : 0); break;
                case TypeId::INT32:
                    st.i32.push_back(valid ? readPod<int32_t>(p) : 0); break;
                case TypeId::FLOAT64:
                    st.f64.push_back(valid ? readPod<double>(p) : 0.0); break;
                case TypeId::BOOL:
                    st.b8.push_back(valid ? readPod<uint8_t>(p) : 0); break;
                case TypeId::STRING: {
                    if (valid) {
                        const auto len = readPod<int32_t>(p);
                        st.str.emplace_back(reinterpret_cast<const char*>(p),
                                            static_cast<size_t>(len));
                        p += len;
                    } else {
                        st.str.emplace_back();
                    }
                    break;
                }
                default:
                    raise(ErrorCode::NotImplemented,
                          "SortOperator: unsupported merge col type {}",
                          childSchema.field(c).type.toString());
            }
        }
        ++rows;
        advanceHead(static_cast<size_t>(best));
    }

    if (rows == 0) return std::nullopt;

    std::vector<ColumnVector> cols;
    cols.reserve(nCols);
    for (size_t c = 0; c < nCols; ++c) {
        const DataType ty = childSchema.field(c).type;
        Stage&         st = stages[c];
        if (ty.id == TypeId::STRING) {
            StringBuilder sb(mm_);
            for (size_t r = 0; r < rows; ++r) {
                if (st.valid[r]) sb.append(st.str[r]);
                else             sb.appendNull();
            }
            cols.push_back(sb.build());
        } else {
            ColumnVector out = ColumnVector::make(ty, rows, mm_);
            out.validity().setAllValid();
            for (size_t r = 0; r < rows; ++r) {
                out.validity().setValid(r, st.valid[r] != 0);
                switch (ty.id) {
                    case TypeId::INT64:   out.typed<int64_t>()[r] = st.i64[r]; break;
                    case TypeId::INT32:   out.typed<int32_t>()[r] = st.i32[r]; break;
                    case TypeId::FLOAT64: out.typed<double>()[r]  = st.f64[r]; break;
                    case TypeId::BOOL:    out.typed<uint8_t>()[r] = st.b8[r];  break;
                    default: break;
                }
            }
            cols.push_back(std::move(out));
        }
    }
    return ColumnBatch(std::move(cols), PositionList::identity(rows));
}

std::optional<ColumnBatch> SortOperator::getNext() {
    if (!built_) {
        build();
        built_ = true;
    }
    if (merging_) return nextMergedBatch();

    if (emitted_) return std::nullopt;
    emitted_ = true;
    if (sorted_.numColumns() == 0 || sorted_.numRows() == 0) return std::nullopt;
    return std::move(sorted_);
}

} // namespace quantum
