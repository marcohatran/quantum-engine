#include "operator/ParallelHashAggregateOperator.hpp"

#include <utility>

#include "common/Status.hpp"
#include "exec/TaskScheduler.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/ScanOperator.hpp"

namespace quantum {

ParallelHashAggregateOperator::ParallelHashAggregateOperator(
        MemoryManager*                    mm,
        std::unique_ptr<ExchangeOperator> exchange,
        std::vector<PartitionAggSpec>     perPartition)
    : Operator(mm),
      exchange_(std::move(exchange)),
      perPartition_(std::move(perPartition)) {
    if (perPartition_.empty()) {
        raise(ErrorCode::InvalidArgument,
              "ParallelHashAggregate: at least one partition spec required");
    }
    if (perPartition_.size() != exchange_->numPartitions()) {
        raise(ErrorCode::InvalidArgument,
              "ParallelHashAggregate: {} partition specs but exchange has {} "
              "partitions", perPartition_.size(), exchange_->numPartitions());
    }

    // Output schema mirrors HashAggregateOperator's: group key types from
    // the key expressions against the child schema, then aggregate output
    // types, named by outputNames.
    const auto&  spec     = perPartition_.front();
    const Schema& inSchema = exchange_->outputSchema();
    if (spec.outputNames.size() != spec.keys.size() + spec.aggs.size()) {
        raise(ErrorCode::InvalidArgument,
              "ParallelHashAggregate: {} output names, expected {}",
              spec.outputNames.size(), spec.keys.size() + spec.aggs.size());
    }
    std::vector<Field> fields;
    fields.reserve(spec.outputNames.size());
    for (size_t i = 0; i < spec.keys.size(); ++i) {
        fields.push_back(Field{spec.outputNames[i],
                               spec.keys[i]->resultType(inSchema)});
    }
    for (size_t i = 0; i < spec.aggs.size(); ++i) {
        fields.push_back(Field{spec.outputNames[spec.keys.size() + i],
                               spec.aggs[i]->outputType()});
    }
    outSchema_ = Schema(std::move(fields));
}

void ParallelHashAggregateOperator::open()  { exchange_->open(); }
void ParallelHashAggregateOperator::close() { exchange_->close(); }

std::optional<ColumnBatch> ParallelHashAggregateOperator::getNext() {
    if (!ran_) {
        ran_ = true;

        auto partitions = exchange_->takePartitions();
        const size_t P  = partitions.size();
        const Schema childSchema = exchange_->outputSchema();

        std::vector<std::vector<ColumnBatch>> outputs(P);
        TaskScheduler::global().parallelFor(P, [&](size_t p) {
            auto& spec = perPartition_[p];
            auto scan = std::make_unique<ScanOperator>(
                mm_, childSchema, std::move(partitions[p]));
            HashAggregateOperator agg(mm_, std::move(scan),
                                      std::move(spec.keys),
                                      std::move(spec.aggs),
                                      spec.outputNames);
            agg.open();
            while (auto b = agg.getNext()) {
                if (b->activeRows() == 0) continue;
                outputs[p].push_back(std::move(*b));
            }
            agg.close();
        });

        for (auto& out : outputs) {
            for (auto& b : out) results_.push_back(std::move(b));
        }
    }

    if (emit_ < results_.size()) return std::move(results_[emit_++]);
    return std::nullopt;
}

} // namespace quantum
