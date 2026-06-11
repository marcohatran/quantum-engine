#include "operator/ProjectOperator.hpp"

#include "common/Status.hpp"

namespace quantum {

ProjectOperator::ProjectOperator(MemoryManager* mm,
                                 OperatorPtr child,
                                 std::vector<ExpressionPtr> exprs,
                                 std::vector<std::string>   outputNames)
    : Operator(mm),
      child_(std::move(child)),
      exprs_(std::move(exprs)) {
    if (outputNames.size() != exprs_.size()) {
        raise(ErrorCode::InvalidArgument,
              "ProjectOperator: {} expressions vs {} names",
              exprs_.size(), outputNames.size());
    }

    const Schema& inSchema = child_->outputSchema();
    std::vector<Field> fields;
    fields.reserve(exprs_.size());
    for (size_t i = 0; i < exprs_.size(); ++i) {
        fields.push_back(Field{std::move(outputNames[i]), exprs_[i]->resultType(inSchema)});
    }
    outSchema_ = Schema(std::move(fields));
}

void ProjectOperator::open()  { child_->open(); }
void ProjectOperator::close() { child_->close(); }

std::optional<ColumnBatch> ProjectOperator::getNext() {
    auto in = child_->getNext();
    if (!in) return std::nullopt;

    std::vector<ColumnVector> outCols;
    outCols.reserve(exprs_.size());
    for (const auto& e : exprs_) {
        // The output batch outlives `in`, so views (bare ColumnRef
        // projections) must be materialized; computed columns are
        // already owned and move through.
        outCols.push_back(e->eval(*in, mm_).ensureOwned(mm_));
    }

    PositionList outPos = in->positions().isIdentity()
        ? PositionList::identity(in->numRows())
        : PositionList::fromIndices(std::vector<int32_t>(
              in->positions().indices().begin(),
              in->positions().indices().end()));

    return ColumnBatch(std::move(outCols), std::move(outPos));
}

} // namespace quantum
