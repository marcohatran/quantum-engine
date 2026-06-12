#pragma once

#include "plan/Plan.hpp"

namespace quantum::plan {

// Row-group statistics pushdown: for every Filter* -> FileScanNode
// (parquet, unpartitioned) chain, split the filter predicates into AND
// conjuncts and copy every `ColumnRef <cmp> Literal(int64|double)`
// comparison (either operand order) into FileScanNode::stat_predicates.
// The Parquet reader uses them to skip row groups whose column-chunk
// min/max statistics cannot match. The FilterNodes stay in the plan —
// skipping is conservative, surviving row groups still need row-level
// filtering.
//
// Runs after pruneScanColumns: predicate ColumnRefs are already in the
// scan's projected output space, which is what ScanPredicate::column
// means. Mutates `root` in place; idempotent.
void pushScanPredicates(PlanNode& root);

} // namespace quantum::plan
