#include "io/ParquetReader.hpp"

#include <cstring>
#include <memory>
#include <string_view>
#include <utility>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>

#include <fmt/format.h>

#include "common/Status.hpp"
#include "io/FileSystem.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

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

template <typename CType, typename ArrowArrayT>
void copyFixedColumn(const arrow::ChunkedArray& chunkedArr,
                     size_t                     rowOffset,
                     size_t                     n,
                     ColumnVector&              out) {
    auto* dest = out.typed<CType>();
    out.validity().setAllValid();

    // Skip whole chunks arithmetically — a per-row skip scan here makes
    // nextBatch O(table) per batch, i.e. O(table^2 / batch) per file.
    size_t chunkStart = 0;  // global row index of the current chunk's first row
    size_t written    = 0;
    for (int c = 0; c < chunkedArr.num_chunks() && written < n; ++c) {
        const auto& arr = static_cast<const ArrowArrayT&>(*chunkedArr.chunk(c));
        const size_t chunkLen = static_cast<size_t>(arr.length());
        if (chunkStart + chunkLen <= rowOffset) {  // entirely before the slice
            chunkStart += chunkLen;
            continue;
        }
        const size_t begin = rowOffset > chunkStart ? rowOffset - chunkStart : 0;
        const size_t take  = std::min(chunkLen - begin, n - written);
        for (size_t i = 0; i < take; ++i) {
            const auto j = static_cast<int64_t>(begin + i);
            dest[written + i] = static_cast<CType>(arr.Value(j));
            out.validity().setValid(written + i, !arr.IsNull(j));
        }
        written    += take;
        chunkStart += chunkLen;
    }
}

ColumnVector copyStringColumn(const arrow::ChunkedArray& chunkedArr,
                              size_t                     rowOffset,
                              size_t                     n,
                              MemoryManager*             mm) {
    StringBuilder sb(mm);

    // Chunk-skipping slice copy, same shape as copyFixedColumn.
    auto copySlice = [&](const auto& sArr, size_t begin, size_t take) {
        for (size_t i = 0; i < take; ++i) {
            const auto j = static_cast<int64_t>(begin + i);
            if (sArr.IsNull(j)) sb.appendNull();
            else {
                const auto sv = sArr.GetView(j);
                sb.append(std::string_view(sv.data(), sv.size()));
            }
        }
    };

    size_t chunkStart = 0;
    size_t written    = 0;
    for (int c = 0; c < chunkedArr.num_chunks() && written < n; ++c) {
        const auto* arr = chunkedArr.chunk(c).get();
        const size_t chunkLen = static_cast<size_t>(arr->length());
        if (chunkStart + chunkLen <= rowOffset) {
            chunkStart += chunkLen;
            continue;
        }
        const size_t begin = rowOffset > chunkStart ? rowOffset - chunkStart : 0;
        const size_t take  = std::min(chunkLen - begin, n - written);
        if (arr->type_id() == arrow::Type::STRING) {
            copySlice(static_cast<const arrow::StringArray&>(*arr), begin, take);
        } else if (arr->type_id() == arrow::Type::LARGE_STRING) {
            copySlice(static_cast<const arrow::LargeStringArray&>(*arr), begin, take);
        } else {
            raise(ErrorCode::Internal,
                  "copyStringColumn: unexpected chunk type {}", arr->type()->ToString());
        }
        written    += take;
        chunkStart += chunkLen;
    }
    return sb.build();
}

} // namespace

ParquetReader::ParquetReader(std::string       path,
                             MemoryManager*    mm,
                             FileFormatOptions options)
    : path_(std::move(path)), mm_(mm), options_(options) {
    // Eagerly load the file. We need the schema available before open()
    // because downstream operators (ProjectOperator) call outputSchema()
    // at construction time — they can't wait until execution begins.
    auto infile = io::openReadable(path_);

    auto reader_result = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
    if (!reader_result.ok()) {
        raise(ErrorCode::InvalidArgument,
              "ParquetReader: parquet open failed: {}", reader_result.status().ToString());
    }
    auto reader = std::move(reader_result).ValueOrDie();

    auto read_status = reader->ReadTable(&table_);
    if (!read_status.ok()) {
        raise(ErrorCode::InvalidArgument,
              "ParquetReader: ReadTable failed: {}", read_status.ToString());
    }

    std::vector<Field> fields;
    fields.reserve(static_cast<size_t>(table_->num_columns()));
    for (int c = 0; c < table_->num_columns(); ++c) {
        const auto& f = table_->schema()->field(c);
        fields.push_back(Field{f->name(), arrowTypeToDataType(*f->type())});
    }
    schema_ = Schema(std::move(fields));
}

ParquetReader::~ParquetReader() = default;

void ParquetReader::open() {
    cursor_ = 0;
    open_   = true;
}

void ParquetReader::close() {
    open_ = false;
}

std::optional<ColumnBatch> ParquetReader::nextBatch() {
    if (!open_) raise(ErrorCode::InvalidArgument, "ParquetReader: nextBatch before open");
    if (!table_) return std::nullopt;
    const size_t total = static_cast<size_t>(table_->num_rows());
    if (cursor_ >= total) return std::nullopt;

    const size_t n = std::min(options_.batchSize, total - cursor_);
    const size_t numCols = static_cast<size_t>(table_->num_columns());

    std::vector<ColumnVector> cols;
    cols.reserve(numCols);

    for (size_t c = 0; c < numCols; ++c) {
        const auto& chunked = *table_->column(static_cast<int>(c));
        const auto  qType   = schema_.field(c).type;

        switch (qType.id) {
            case TypeId::BOOL: {
                auto col = ColumnVector::make(Bool, n, mm_);
                col.validity().setAllValid();
                auto* dest = col.typed<uint8_t>();
                size_t globalRow = 0, written = 0;
                for (int ci = 0; ci < chunked.num_chunks() && written < n; ++ci) {
                    const auto& arr = static_cast<const arrow::BooleanArray&>(*chunked.chunk(ci));
                    const size_t chunkLen = static_cast<size_t>(arr.length());
                    for (size_t i = 0; i < chunkLen && written < n; ++i) {
                        if (globalRow >= cursor_) {
                            dest[written] = arr.Value(static_cast<int64_t>(i)) ? 1U : 0U;
                            col.validity().setValid(written, !arr.IsNull(static_cast<int64_t>(i)));
                            ++written;
                        }
                        ++globalRow;
                    }
                }
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::INT32: {
                auto col = ColumnVector::make(Int32, n, mm_);
                copyFixedColumn<int32_t, arrow::Int32Array>(chunked, cursor_, n, col);
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::INT64: {
                auto col = ColumnVector::make(Int64, n, mm_);
                copyFixedColumn<int64_t, arrow::Int64Array>(chunked, cursor_, n, col);
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::FLOAT64: {
                auto col = ColumnVector::make(Float64, n, mm_);
                copyFixedColumn<double, arrow::DoubleArray>(chunked, cursor_, n, col);
                cols.push_back(std::move(col));
                break;
            }
            case TypeId::STRING: {
                cols.push_back(copyStringColumn(chunked, cursor_, n, mm_));
                break;
            }
        }
    }
    cursor_ += n;
    return ColumnBatch(std::move(cols), PositionList::identity(n));
}

} // namespace quantum
