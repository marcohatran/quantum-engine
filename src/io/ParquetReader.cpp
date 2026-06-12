#include "io/ParquetReader.hpp"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>
#include <parquet/statistics.h>

#include <fmt/format.h>

#include "common/Status.hpp"
#include "io/ArrowConvert.hpp"
#include "io/FileSystem.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

namespace {

DataType arrowTypeToDataType(const arrow::DataType& t) {
    switch (t.id()) {
        case arrow::Type::BOOL:   return Bool;
        case arrow::Type::INT32:  return Int32;
        case arrow::Type::INT64:  return Int64;
        case arrow::Type::DOUBLE: return Float64;
        case arrow::Type::STRING: return DataType{TypeId::STRING};
        case arrow::Type::LARGE_STRING: return DataType{TypeId::STRING};
        default:
            raise(ErrorCode::NotImplemented,
                  "ParquetReader: Arrow type {} not supported", t.ToString());
    }
}

} // namespace

ParquetReader::ParquetReader(std::string       path,
                             MemoryManager*    mm,
                             FileFormatOptions options)
    : path_(std::move(path)), mm_(mm), options_(options) {
    // Open the footer only — schema must be available before open()
    // (downstream operators compute their output schema at construction
    // time), but no data is decoded until batches are pulled.
    auto infile = io::openReadable(path_);

    parquet::ArrowReaderProperties props;
    props.set_batch_size(static_cast<int64_t>(options_.batchSize));

    parquet::arrow::FileReaderBuilder builder;
    auto st = builder.Open(infile);
    if (!st.ok()) {
        raise(ErrorCode::InvalidArgument,
              "ParquetReader: parquet open failed: {}", st.ToString());
    }
    builder.memory_pool(arrow::default_memory_pool());
    builder.properties(props);
    auto reader_result = builder.Build();
    if (!reader_result.ok()) {
        raise(ErrorCode::InvalidArgument,
              "ParquetReader: reader build failed: {}",
              reader_result.status().ToString());
    }
    reader_ = std::move(reader_result).ValueOrDie();

    std::shared_ptr<arrow::Schema> arrowSchema;
    auto schema_st = reader_->GetSchema(&arrowSchema);
    if (!schema_st.ok()) {
        raise(ErrorCode::InvalidArgument,
              "ParquetReader: schema read failed: {}", schema_st.ToString());
    }
    std::vector<Field> fields;
    if (options_.columns.empty()) {
        fields.reserve(static_cast<size_t>(arrowSchema->num_fields()));
        for (const auto& f : arrowSchema->fields()) {
            fields.push_back(Field{f->name(), arrowTypeToDataType(*f->type())});
        }
    } else {
        // Projected read: schema (and every decoded batch) carries only the
        // selected columns, in the given ascending order.
        fields.reserve(options_.columns.size());
        for (const int32_t idx : options_.columns) {
            if (idx < 0 || idx >= arrowSchema->num_fields()) {
                raise(ErrorCode::InvalidArgument,
                      "ParquetReader: projected column {} out of range "
                      "(file has {} columns)", idx, arrowSchema->num_fields());
            }
            const auto& f = arrowSchema->field(idx);
            fields.push_back(Field{f->name(), arrowTypeToDataType(*f->type())});
        }
    }
    schema_ = Schema(std::move(fields));
}

ParquetReader::~ParquetReader() = default;

namespace {

// Could a row group whose column-chunk min/max is [min, max] contain a
// row satisfying `value <op> lit`? Missing/odd stats answer true.
template <typename T>
bool rangeMayMatch(StatsPredicate::Op op, T min, T max, T lit) {
    switch (op) {
        case StatsPredicate::Op::Eq:  return min <= lit && lit <= max;
        case StatsPredicate::Op::Lt:  return min < lit;
        case StatsPredicate::Op::Lte: return min <= lit;
        case StatsPredicate::Op::Gt:  return max > lit;
        case StatsPredicate::Op::Gte: return max >= lit;
    }
    return true;
}

bool rowGroupMayMatch(const parquet::RowGroupMetaData&  rg,
                      const std::vector<StatsPredicate>& preds,
                      const std::vector<int32_t>&        projection) {
    for (const auto& pred : preds) {
        const int fileIdx = projection.empty()
            ? pred.column
            : projection[static_cast<size_t>(pred.column)];
        if (fileIdx < 0 || fileIdx >= rg.num_columns()) continue;
        auto chunk = rg.ColumnChunk(fileIdx);
        auto stats = chunk->statistics();
        if (!stats || !stats->HasMinMax()) continue;
        bool may = true;
        if (stats->physical_type() == parquet::Type::INT64 &&
            std::holds_alternative<int64_t>(pred.literal)) {
            const auto& s =
                static_cast<const parquet::Int64Statistics&>(*stats);
            may = rangeMayMatch(pred.op, s.min(), s.max(),
                                std::get<int64_t>(pred.literal));
        } else if (stats->physical_type() == parquet::Type::DOUBLE &&
                   std::holds_alternative<double>(pred.literal)) {
            const auto& s =
                static_cast<const parquet::DoubleStatistics&>(*stats);
            may = rangeMayMatch(pred.op, s.min(), s.max(),
                                std::get<double>(pred.literal));
        }
        if (!may) return false;
    }
    return true;
}

} // namespace

void ParquetReader::open() {
    auto rb_result = [&] {
        if (options_.columns.empty() && options_.statsPredicates.empty()) {
            return reader_->GetRecordBatchReader();
        }
        // Projected / stats-skipping read. Flat schemas make field index ==
        // leaf column index.
        const auto* meta = reader_->parquet_reader()->metadata().get();
        std::vector<int> rowGroups;
        rowGroups.reserve(static_cast<size_t>(meta->num_row_groups()));
        for (int i = 0; i < meta->num_row_groups(); ++i) {
            if (options_.statsPredicates.empty() ||
                rowGroupMayMatch(*meta->RowGroup(i),
                                 options_.statsPredicates,
                                 options_.columns)) {
                rowGroups.push_back(i);
            }
        }
        if (options_.columns.empty()) {
            return reader_->GetRecordBatchReader(rowGroups);
        }
        return reader_->GetRecordBatchReader(
            rowGroups, std::vector<int>(options_.columns.begin(),
                                        options_.columns.end()));
    }();
    if (!rb_result.ok()) {
        raise(ErrorCode::InvalidArgument,
              "ParquetReader: record batch reader failed: {}",
              rb_result.status().ToString());
    }
    batchReader_ = std::move(rb_result).ValueOrDie();
    open_ = true;
}

void ParquetReader::close() {
    batchReader_.reset();
    open_ = false;
}

std::optional<ColumnBatch> ParquetReader::nextBatch() {
    if (!open_) raise(ErrorCode::InvalidArgument, "ParquetReader: nextBatch before open");

    std::shared_ptr<arrow::RecordBatch> rb;
    auto st = batchReader_->ReadNext(&rb);
    if (!st.ok()) {
        raise(ErrorCode::Internal,
              "ParquetReader: batch read failed: {}", st.ToString());
    }
    if (!rb || rb->num_rows() == 0) return std::nullopt;
    return io::batchFromArrow(*rb, mm_);
}

} // namespace quantum
