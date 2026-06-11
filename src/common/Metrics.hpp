#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace quantum {

// Per-operator execution counters, filled in by MetricsOperator (the
// decorator PlanBuilder wraps around every built operator when a
// QueryMetrics sink is supplied). Single-threaded like the operators.
struct OperatorMetrics {
    std::string name;       // demangled operator class name
    uint64_t    rows    = 0;
    uint64_t    batches = 0;
    uint64_t    openNanos  = 0;
    uint64_t    nextNanos  = 0;  // cumulative across all getNext calls
    uint64_t    closeNanos = 0;
};

// One query's operator metrics, in plan-build order (children precede
// parents — the build is bottom-up — so the plan root is the LAST entry).
class QueryMetrics {
public:
    OperatorMetrics* add(std::string name);

    const std::vector<std::unique_ptr<OperatorMetrics>>& operators() const noexcept {
        return ops_;
    }
    const OperatorMetrics* root() const noexcept {
        return ops_.empty() ? nullptr : ops_.back().get();
    }

    // Structured query trace:
    //   {"operators":[{"operator":...,"rows":...,"batches":...,
    //                  "open_ns":...,"next_ns":...,"close_ns":...}, ...]}
    std::string toJson() const;

private:
    std::vector<std::unique_ptr<OperatorMetrics>> ops_;
};

// Process-wide cumulative counters, rendered in Prometheus text format.
namespace metrics {

void recordSpill() noexcept;                              // any operator spill
void recordQuery(uint64_t rowsOut, uint64_t execNanos) noexcept;

// Prometheus text exposition of the cumulative counters:
//   quantum_queries_total, quantum_rows_emitted_total,
//   quantum_spills_total, quantum_exec_nanos_total
std::string prometheusText();

void resetForTesting() noexcept;

} // namespace metrics

} // namespace quantum
