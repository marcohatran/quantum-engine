#pragma once

#include <string>
#include <string_view>

#include "plan/Plan.hpp"

namespace quantum::plan {

// JSON wire format for the plan IR. Schema (informal):
//
//   PlanNode: {"kind": "scan" | "filter" | "project" | "hash_agg" | "hash_join", ...}
//   ScalarExpr: {"kind": "literal" | "column_ref" | "binary" | "like", ...}
//
// Used by the JNI bridge and by tests. Not designed for human authoring at
// scale; Protobuf will replace this for the production JVM bridge in a later
// Phase 4 slice.

std::string planToJson(const PlanNode& node);
PlanNode    planFromJson(std::string_view json);

std::string scalarExprToJson(const ScalarExpr& expr);
ScalarExpr  scalarExprFromJson(std::string_view json);

} // namespace quantum::plan
