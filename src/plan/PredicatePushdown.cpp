#include "plan/PredicatePushdown.hpp"

#include <cstdlib>
#include <string_view>
#include <variant>
#include <vector>

namespace quantum::plan {

namespace {

bool isComparison(BinaryOpType op) {
    switch (op) {
        case BinaryOpType::Eq:
        case BinaryOpType::Lt:
        case BinaryOpType::Gt:
        case BinaryOpType::Lte:
        case BinaryOpType::Gte:
            return true;
        default:
            return false;
    }
}

// Mirror the comparison for `lit <op> col` -> `col <op'> lit`.
BinaryOpType flip(BinaryOpType op) {
    switch (op) {
        case BinaryOpType::Lt:  return BinaryOpType::Gt;
        case BinaryOpType::Gt:  return BinaryOpType::Lt;
        case BinaryOpType::Lte: return BinaryOpType::Gte;
        case BinaryOpType::Gte: return BinaryOpType::Lte;
        default:                return op;  // Eq is symmetric
    }
}

bool numericLiteral(const LiteralExpr& lit) {
    return std::holds_alternative<int64_t>(lit.value) ||
           std::holds_alternative<double>(lit.value);
}

// Collect `col <cmp> literal` conjuncts of `expr` into `out`.
void collectConjuncts(const ScalarExpr&           expr,
                      std::vector<ScanPredicate>& out) {
    const auto* bin = std::get_if<std::unique_ptr<BinaryOpExpr>>(&expr);
    if (bin == nullptr) return;
    const BinaryOpExpr& e = **bin;
    if (e.op == BinaryOpType::And) {
        collectConjuncts(e.lhs, out);
        collectConjuncts(e.rhs, out);
        return;
    }
    if (!isComparison(e.op)) return;
    const auto* lhsCol = std::get_if<ColumnRefExpr>(&e.lhs);
    const auto* rhsCol = std::get_if<ColumnRefExpr>(&e.rhs);
    const auto* lhsLit = std::get_if<LiteralExpr>(&e.lhs);
    const auto* rhsLit = std::get_if<LiteralExpr>(&e.rhs);
    if (lhsCol != nullptr && rhsLit != nullptr && numericLiteral(*rhsLit)) {
        out.push_back(ScanPredicate{lhsCol->index, e.op, *rhsLit});
    } else if (lhsLit != nullptr && rhsCol != nullptr &&
               numericLiteral(*lhsLit)) {
        out.push_back(ScanPredicate{rhsCol->index, flip(e.op), *lhsLit});
    }
}

// Descend Filter* chains: collect predicates until the chain ends at a
// parquet FileScanNode, then attach them.
void tryPushChain(PlanNode& filterNode) {
    std::vector<const FilterNode*> filters;
    PlanNode* cursor = &filterNode;
    FileScanNode* scan = nullptr;
    while (scan == nullptr) {
        if (auto* f = std::get_if<std::unique_ptr<FilterNode>>(cursor)) {
            filters.push_back(f->get());
            cursor = &(*f)->child;
        } else if (auto* fs = std::get_if<FileScanNode>(cursor)) {
            scan = fs;
        } else {
            return;
        }
    }
    if (scan->format != "parquet") return;
    if (scan->partition_columns.numFields() > 0) return;  // refs may be
                                                          // partition cols
    if (!scan->stat_predicates.empty()) return;           // already pushed

    std::vector<ScanPredicate> preds;
    for (const FilterNode* f : filters) collectConjuncts(f->predicate, preds);
    scan->stat_predicates = std::move(preds);
}

void walk(PlanNode& node) {
    std::visit([&](auto& n) {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<FilterNode>>) {
            tryPushChain(node);
            walk(n->child);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ProjectNode>>) {
            walk(n->child);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashAggregateNode>>) {
            walk(n->child);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashJoinNode>>) {
            walk(n->build);
            walk(n->probe);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<LimitNode>>) {
            walk(n->child);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<SortNode>>) {
            walk(n->child);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<UnionAllNode>>) {
            for (auto& c : n->children) walk(c);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ExchangeNode>>) {
            walk(n->child);
        }
        // ScanNode / FileScanNode / ExtensionNode: leaves.
    }, node);
}

} // namespace

void pushScanPredicates(PlanNode& root) {
    static const bool enabled = [] {
        const char* env = std::getenv("QUANTUM_STATS_SKIP");
        return env == nullptr || std::string_view(env) != "0";
    }();
    if (enabled) walk(root);
}

} // namespace quantum::plan
