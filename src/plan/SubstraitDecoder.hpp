#pragma once

#include <string_view>

#include "plan/Plan.hpp"

namespace quantum::plan {

// Decode a serialized `substrait.Plan` (the cross-engine IR; protos
// vendored under third_party/substrait, version in VERSION) into the
// engine's PlanNode IR, so external producers — Ibis, Isthmus/Calcite,
// DuckDB — can target the engine directly.
//
// Supported relations: ReadRel(NamedTable), FilterRel, ProjectRel
// (incl. `emit` remaps), AggregateRel (single grouping set of field
// references; sum/min/max/avg/count measures, COUNT(DISTINCT) via the
// DISTINCT invocation), FetchRel (offset 0), SortRel. Expressions:
// field selections, bool/i32/i64/fp64/string literals, and scalar
// functions by extension name: lt/gt/lte/gte/equal, and/or/not,
// is_null, add/subtract/multiply. Everything else raises NotImplemented
// with the offending construct named.
PlanNode planFromSubstraitBytes(std::string_view bytes);

} // namespace quantum::plan
