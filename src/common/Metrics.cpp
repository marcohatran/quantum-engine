#include "common/Metrics.hpp"

#include <atomic>

#include <nlohmann/json.hpp>

namespace quantum {

OperatorMetrics* QueryMetrics::add(std::string name) {
    ops_.push_back(std::make_unique<OperatorMetrics>());
    ops_.back()->name = std::move(name);
    return ops_.back().get();
}

std::string QueryMetrics::toJson() const {
    nlohmann::json ops = nlohmann::json::array();
    for (const auto& m : ops_) {
        ops.push_back({
            {"operator", m->name},
            {"rows",     m->rows},
            {"batches",  m->batches},
            {"open_ns",  m->openNanos},
            {"next_ns",  m->nextNanos},
            {"close_ns", m->closeNanos},
        });
    }
    return nlohmann::json{{"operators", std::move(ops)}}.dump();
}

namespace metrics {

namespace {
std::atomic<uint64_t> g_queries{0};
std::atomic<uint64_t> g_rows{0};
std::atomic<uint64_t> g_spills{0};
std::atomic<uint64_t> g_execNanos{0};
} // namespace

void recordSpill() noexcept {
    g_spills.fetch_add(1, std::memory_order_relaxed);
}

void recordQuery(uint64_t rowsOut, uint64_t execNanos) noexcept {
    g_queries.fetch_add(1, std::memory_order_relaxed);
    g_rows.fetch_add(rowsOut, std::memory_order_relaxed);
    g_execNanos.fetch_add(execNanos, std::memory_order_relaxed);
}

std::string prometheusText() {
    std::string out;
    auto counter = [&](const char* name, const char* help, uint64_t v) {
        out += "# HELP ";  out += name; out += ' '; out += help; out += '\n';
        out += "# TYPE ";  out += name; out += " counter\n";
        out += name; out += ' '; out += std::to_string(v); out += '\n';
    };
    counter("quantum_queries_total",      "Queries executed by the native engine.",
            g_queries.load(std::memory_order_relaxed));
    counter("quantum_rows_emitted_total", "Rows emitted by query plan roots.",
            g_rows.load(std::memory_order_relaxed));
    counter("quantum_spills_total",       "Operator spill events (HashAggregate + Sort).",
            g_spills.load(std::memory_order_relaxed));
    counter("quantum_exec_nanos_total",   "Cumulative wall time inside plan roots (ns).",
            g_execNanos.load(std::memory_order_relaxed));
    return out;
}

void resetForTesting() noexcept {
    g_queries.store(0);
    g_rows.store(0);
    g_spills.store(0);
    g_execNanos.store(0);
}

} // namespace metrics

} // namespace quantum
