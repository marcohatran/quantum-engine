#pragma once

#include "plan/Plan.hpp"

namespace quantum::plan {

// Scan column pruning: for every HashAggregateNode whose child chain is
// Filter* -> FileScanNode (parquet, unpartitioned, not already projected),
// compute the set of file columns referenced by the aggregate's group
// keys, aggregate inputs, and the chain's filter predicates; record it in
// FileScanNode::projected_columns and remap every ColumnRef in the chain
// to the pruned positions. The scan then decodes only the used columns —
// on wide tables this skips most of the Parquet decode cost.
//
// Nodes above the aggregate reference its output space and are
// unaffected. Plain (non-aggregate) roots are never pruned: their scan
// schema is host-visible. Mutates `root` in place; idempotent.
void pruneScanColumns(PlanNode& root);

} // namespace quantum::plan
