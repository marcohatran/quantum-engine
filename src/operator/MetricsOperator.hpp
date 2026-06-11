#pragma once

#include <chrono>
#include <optional>
#include <utility>

#include "common/Metrics.hpp"
#include "operator/Operator.hpp"

namespace quantum {

// Transparent decorator that times open/getNext/close and counts emitted
// rows + batches into an OperatorMetrics owned by a QueryMetrics. The
// wrapped operator is untouched; PlanBuilder wraps every node it builds
// when a metrics sink is configured, so the whole plan reports uniformly.
// Per-batch overhead is two steady_clock reads — negligible at 4k-row
// batch granularity.
class MetricsOperator final : public Operator {
public:
    MetricsOperator(OperatorPtr inner, OperatorMetrics* m)
        : Operator(nullptr), inner_(std::move(inner)), m_(m) {}

    void open() override {
        const auto t0 = std::chrono::steady_clock::now();
        inner_->open();
        m_->openNanos += nanosSince(t0);
    }

    std::optional<ColumnBatch> getNext() override {
        const auto t0 = std::chrono::steady_clock::now();
        auto batch = inner_->getNext();
        m_->nextNanos += nanosSince(t0);
        if (batch.has_value()) {
            m_->batches += 1;
            m_->rows    += batch->activeRows();
        }
        return batch;
    }

    void close() override {
        const auto t0 = std::chrono::steady_clock::now();
        inner_->close();
        m_->closeNanos += nanosSince(t0);
    }

    const Schema& outputSchema() const override { return inner_->outputSchema(); }

private:
    static uint64_t nanosSince(std::chrono::steady_clock::time_point t0) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
    }

    OperatorPtr      inner_;
    OperatorMetrics* m_;
};

} // namespace quantum
