#include "operator/ExchangeOperator.hpp"

#include <cstring>
#include <string_view>
#include <utility>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

ExchangeOperator::ExchangeOperator(MemoryManager*             mm,
                                   OperatorPtr                child,
                                   std::vector<ExpressionPtr> keys,
                                   size_t                     numPartitions)
    : Operator(mm),
      child_(std::move(child)),
      keys_(std::move(keys)),
      numPartitions_(numPartitions) {
    if (keys_.empty()) {
        raise(ErrorCode::InvalidArgument,
              "ExchangeOperator: at least one partition key required");
    }
    if (numPartitions_ == 0) {
        raise(ErrorCode::InvalidArgument,
              "ExchangeOperator: numPartitions must be >= 1");
    }
}

void ExchangeOperator::open()  { child_->open(); }
void ExchangeOperator::close() { child_->close(); }

namespace {

constexpr uint64_t kNullHash = 0x9E3779B97F4A7C15ULL;

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t hashKeyAt(const ColumnVector& col, size_t row) {
    if (!col.validity().isValid(row)) return kNullHash;
    switch (col.type().id) {
        case TypeId::INT64:
            return splitmix64(static_cast<uint64_t>(col.typed<int64_t>()[row]));
        case TypeId::INT32:
            return splitmix64(static_cast<uint64_t>(
                static_cast<int64_t>(col.typed<int32_t>()[row])));
        case TypeId::BOOL:
            return splitmix64(col.typed<uint8_t>()[row]);
        case TypeId::FLOAT64: {
            uint64_t bits = 0;
            const double v = col.typed<double>()[row];
            std::memcpy(&bits, &v, sizeof(bits));
            return splitmix64(bits);
        }
        case TypeId::STRING: {
            const std::string_view s = col.stringAt(row);
            uint64_t h = 1469598103934665603ULL;  // FNV-1a
            for (const char c : s) {
                h ^= static_cast<uint8_t>(c);
                h *= 1099511628211ULL;
            }
            return h;
        }
        default:
            raise(ErrorCode::NotImplemented,
                  "ExchangeOperator: unsupported key type {}",
                  col.type().toString());
    }
}

// Gather `rows` of `src` into a fresh dense batch (SortOperator::permute's
// pattern, with an arbitrary index list).
ColumnBatch gatherRows(const ColumnBatch&          src,
                       const Schema&               schema,
                       const std::vector<int32_t>& rows,
                       MemoryManager*              mm) {
    const size_t nCols = schema.numFields();
    const size_t nOut  = rows.size();

    std::vector<ColumnVector> cols;
    cols.reserve(nCols);
    for (size_t c = 0; c < nCols; ++c) {
        const DataType ty  = schema.field(c).type;
        const auto&    col = src.column(c);
        if (ty.id == TypeId::STRING) {
            StringBuilder sb(mm);
            for (const int32_t r : rows) {
                const auto i = static_cast<size_t>(r);
                if (!col.validity().isValid(i)) sb.appendNull();
                else                            sb.append(col.stringAt(i));
            }
            cols.push_back(sb.build());
        } else {
            ColumnVector out = ColumnVector::make(ty, nOut, mm);
            out.validity().setAllValid();
            for (size_t k = 0; k < nOut; ++k) {
                const auto i = static_cast<size_t>(rows[k]);
                out.validity().setValid(k, col.validity().isValid(i));
                switch (ty.id) {
                    case TypeId::INT64:
                        out.typed<int64_t>()[k] = col.typed<int64_t>()[i]; break;
                    case TypeId::INT32:
                        out.typed<int32_t>()[k] = col.typed<int32_t>()[i]; break;
                    case TypeId::FLOAT64:
                        out.typed<double>()[k] = col.typed<double>()[i]; break;
                    case TypeId::BOOL:
                        out.typed<uint8_t>()[k] = col.typed<uint8_t>()[i]; break;
                    default:
                        raise(ErrorCode::NotImplemented,
                              "ExchangeOperator: unsupported column type {}",
                              ty.toString());
                }
            }
            cols.push_back(std::move(out));
        }
    }
    return ColumnBatch(std::move(cols), PositionList::identity(nOut));
}

} // namespace

void ExchangeOperator::materialize() {
    if (materialized_) return;
    materialized_ = true;
    partitions_.clear();
    partitions_.resize(numPartitions_);

    const Schema& schema = child_->outputSchema();
    std::vector<std::vector<int32_t>> rowsByPartition(numPartitions_);

    while (auto batch = child_->getNext()) {
        if (batch->activeRows() == 0) continue;

        // Key columns cover batch->numRows(); only active rows are routed.
        std::vector<ColumnVector> keyCols;
        keyCols.reserve(keys_.size());
        for (const auto& k : keys_) keyCols.push_back(k->eval(*batch, mm_));

        for (auto& v : rowsByPartition) v.clear();
        const auto& pos = batch->positions();
        for (size_t k = 0; k < pos.size(); ++k) {
            const auto i = static_cast<size_t>(pos.at(k));
            uint64_t h = 0;
            for (const auto& kc : keyCols) {
                h = splitmix64(h ^ hashKeyAt(kc, i));
            }
            rowsByPartition[h % numPartitions_].push_back(
                static_cast<int32_t>(i));
        }

        for (size_t p = 0; p < numPartitions_; ++p) {
            if (rowsByPartition[p].empty()) continue;
            partitions_[p].push_back(
                gatherRows(*batch, schema, rowsByPartition[p], mm_));
        }
    }
}

std::vector<std::vector<ColumnBatch>> ExchangeOperator::takePartitions() {
    materialize();
    return std::move(partitions_);
}

std::optional<ColumnBatch> ExchangeOperator::getNext() {
    materialize();
    while (emitPartition_ < partitions_.size()) {
        auto& part = partitions_[emitPartition_];
        if (emitBatch_ < part.size()) {
            return std::move(part[emitBatch_++]);
        }
        ++emitPartition_;
        emitBatch_ = 0;
    }
    return std::nullopt;
}

} // namespace quantum
