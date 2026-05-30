#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "expression/Expression.hpp"
#include "operator/Operator.hpp"
#include "plan/Plan.hpp"

namespace quantum {

class MemoryManager;
class ColumnBatch;

namespace plan {

// A SourceCatalog supplies the actual data for each ScanNode in a plan. The
// key is the ScanNode::source_id; the value is a vector of ColumnBatch moved
// into the constructed ScanOperator.
using SourceCatalog = std::unordered_map<std::string, std::vector<ColumnBatch>>;

// Walks a PlanNode IR and produces an Operator tree owned by the caller.
// Scan nodes consume their entries from `catalog` (moved out). Expressions
// are translated to ExpressionPtr; aggregates to AggregateFunctionPtr.
class PlanBuilder {
public:
    explicit PlanBuilder(MemoryManager* mm) : mm_(mm) {}

    // Builds the operator tree. The returned operator owns its sub-tree.
    // After this call, `catalog` may have been partially drained.
    OperatorPtr build(const PlanNode& node, SourceCatalog& catalog);

    // Translate a scalar expression IR into the engine's Expression tree.
    ExpressionPtr buildExpr(const ScalarExpr& expr);

private:
    MemoryManager* mm_;
};

} // namespace plan
} // namespace quantum
