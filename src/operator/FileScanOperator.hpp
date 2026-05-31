#pragma once

#include "io/FileFormat.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Operator wrapping any FileFormatReader. Plugs into the existing operator
// tree so Filter / Project / HashAgg / HashJoin can sit above an on-disk
// data source.
class FileScanOperator final : public Operator {
public:
    FileScanOperator(MemoryManager* mm, FileFormatReaderPtr reader)
        : Operator(mm), reader_(std::move(reader)) {}

    void                       open()  override { reader_->open(); }
    std::optional<ColumnBatch> getNext() override { return reader_->nextBatch(); }
    void                       close() override { reader_->close(); }
    const Schema&              outputSchema() const override { return reader_->schema(); }

private:
    FileFormatReaderPtr reader_;
};

} // namespace quantum
