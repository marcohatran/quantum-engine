#pragma once

// Shared definitions of the C-ABI opaque handle types. quantum_c.cpp,
// quantum_arrow.cpp and quantum_ipc.cpp together form the bridge layer;
// they include this header so every entry point can reach the underlying
// C++ objects without fragile per-file mirror declarations.

#include <memory>
#include <optional>
#include <string>

#include "common/Metrics.hpp"
#include "memory/MemoryManager.hpp"
#include "operator/Operator.hpp"
#include "plan/PlanBuilder.hpp"
#include "vector/ColumnBatch.hpp"
#include "vector/ColumnVector.hpp"

struct QEngine {
    std::unique_ptr<quantum::MemoryManager> mm;
    quantum::plan::SourceCatalog            catalog;
};

struct QColumn {
    quantum::ColumnVector vec;
};

struct QBatch {
    quantum::ColumnBatch batch;
};

struct QResult {
    quantum::OperatorPtr                op;
    quantum::MemoryManager*             mm = nullptr;  // borrowed from QEngine
    std::optional<quantum::ColumnBatch> current;
    QBatch                              viewHandle;  // reused across q_result_next calls
    std::unique_ptr<quantum::QueryMetrics> metrics;  // per-operator counters
    std::string                            traceJson;  // q_result_trace_json storage
};
