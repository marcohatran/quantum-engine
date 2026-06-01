#include "io/PartitionedFileReader.hpp"

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

namespace {

ColumnVector broadcastConstant(const Field&          field,
                               const PartitionValue& value,
                               size_t                numRows,
                               MemoryManager*        mm) {
    switch (field.type.id) {
        case TypeId::INT64: {
            if (!std::holds_alternative<int64_t>(value)) {
                raise(ErrorCode::TypeMismatch,
                      "PartitionedFileReader: partition '{}' expects INT64 value",
                      field.name);
            }
            const int64_t v = std::get<int64_t>(value);
            auto col = ColumnVector::make(Int64, numRows, mm);
            col.validity().setAllValid();
            auto* dest = col.typed<int64_t>();
            for (size_t i = 0; i < numRows; ++i) dest[i] = v;
            return col;
        }
        case TypeId::FLOAT64: {
            if (!std::holds_alternative<double>(value)) {
                raise(ErrorCode::TypeMismatch,
                      "PartitionedFileReader: partition '{}' expects FLOAT64 value",
                      field.name);
            }
            const double v = std::get<double>(value);
            auto col = ColumnVector::make(Float64, numRows, mm);
            col.validity().setAllValid();
            auto* dest = col.typed<double>();
            for (size_t i = 0; i < numRows; ++i) dest[i] = v;
            return col;
        }
        case TypeId::BOOL: {
            if (!std::holds_alternative<bool>(value)) {
                raise(ErrorCode::TypeMismatch,
                      "PartitionedFileReader: partition '{}' expects BOOL value",
                      field.name);
            }
            const bool v = std::get<bool>(value);
            auto col = ColumnVector::make(Bool, numRows, mm);
            col.validity().setAllValid();
            auto* dest = col.typed<uint8_t>();
            for (size_t i = 0; i < numRows; ++i) dest[i] = v ? 1U : 0U;
            return col;
        }
        case TypeId::STRING: {
            if (!std::holds_alternative<std::string>(value)) {
                raise(ErrorCode::TypeMismatch,
                      "PartitionedFileReader: partition '{}' expects STRING value",
                      field.name);
            }
            const std::string& v = std::get<std::string>(value);
            StringBuilder sb(mm);
            for (size_t i = 0; i < numRows; ++i) sb.append(v);
            return sb.build();
        }
        default:
            raise(ErrorCode::NotImplemented,
                  "PartitionedFileReader: unsupported partition type for '{}'",
                  field.name);
    }
}

} // namespace

PartitionedFileReader::PartitionedFileReader(
    FileFormatReaderPtr           inner,
    std::vector<Field>            partitionFields,
    std::vector<PartitionValue>   partitionValues,
    MemoryManager*                mm)
    : inner_(std::move(inner)),
      partitionFields_(std::move(partitionFields)),
      partitionValues_(std::move(partitionValues)),
      mm_(mm) {
    if (partitionFields_.size() != partitionValues_.size()) {
        raise(ErrorCode::InvalidArgument,
              "PartitionedFileReader: {} fields but {} values",
              partitionFields_.size(), partitionValues_.size());
    }
    std::vector<Field> outFields(inner_->schema().fields());
    for (const auto& f : partitionFields_) outFields.push_back(f);
    schema_ = Schema(std::move(outFields));
}

void PartitionedFileReader::open()  { inner_->open(); }
void PartitionedFileReader::close() { inner_->close(); }

std::optional<ColumnBatch> PartitionedFileReader::nextBatch() {
    auto batchOpt = inner_->nextBatch();
    if (!batchOpt.has_value()) return std::nullopt;

    ColumnBatch innerBatch = std::move(*batchOpt);
    const size_t n = innerBatch.numRows();

    // Move inner columns into the output, then append broadcast partition cols.
    std::vector<ColumnVector> outCols;
    outCols.reserve(innerBatch.numColumns() + partitionFields_.size());
    for (auto& c : innerBatch.columns()) outCols.push_back(std::move(c));
    for (size_t i = 0; i < partitionFields_.size(); ++i) {
        outCols.push_back(broadcastConstant(partitionFields_[i],
                                            partitionValues_[i],
                                            n, mm_));
    }

    // Preserve the inner batch's position list — filters above might still
    // be using it.
    return ColumnBatch(std::move(outCols), innerBatch.positions());
}

} // namespace quantum
