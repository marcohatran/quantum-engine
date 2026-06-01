#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/Status.hpp"
#include "io/FileFormat.hpp"

namespace quantum {

// Concatenates the batches from a sequence of FileFormatReaders. Each
// underlying reader is opened lazily (only when we advance to it) and
// closed before moving on, so the steady-state file-handle count is one.
//
// All inputs are assumed to share the same schema; we report the schema
// of the first reader. Mismatched schemas would surface as crashes
// downstream — Spark normally enforces uniform schemas across a Parquet
// directory's part files.
class ConcatFileReader final : public FileFormatReader {
public:
    explicit ConcatFileReader(std::vector<FileFormatReaderPtr> readers)
        : readers_(std::move(readers)) {
        if (readers_.empty()) {
            raise(ErrorCode::InvalidArgument,
                  "ConcatFileReader: requires at least one underlying reader");
        }
    }

    void open() override {
        cursor_ = 0;
        if (!readers_.empty()) readers_[0]->open();
    }

    std::optional<ColumnBatch> nextBatch() override {
        while (cursor_ < readers_.size()) {
            auto batch = readers_[cursor_]->nextBatch();
            if (batch.has_value()) return batch;
            // Current reader exhausted — close it and advance.
            readers_[cursor_]->close();
            ++cursor_;
            if (cursor_ < readers_.size()) readers_[cursor_]->open();
        }
        return std::nullopt;
    }

    void close() override {
        if (cursor_ < readers_.size()) readers_[cursor_]->close();
    }

    const Schema& schema() const override { return readers_.front()->schema(); }

private:
    std::vector<FileFormatReaderPtr> readers_;
    size_t                           cursor_ = 0;
};

} // namespace quantum
