#include "plan/ColumnPruning.hpp"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace quantum::plan {

namespace {

void collectRefs(const ScalarExpr& expr, std::set<int32_t>& out) {
    std::visit([&](const auto& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, ColumnRefExpr>) {
            out.insert(e.index);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<BinaryOpExpr>>) {
            collectRefs(e->lhs, out);
            collectRefs(e->rhs, out);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<LikeOpExpr>>) {
            collectRefs(e->input, out);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<UnaryOpExpr>>) {
            collectRefs(e->input, out);
        }
    }, expr);
}

void remapRefs(ScalarExpr& expr,
               const std::unordered_map<int32_t, int32_t>& remap) {
    std::visit([&](auto& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, ColumnRefExpr>) {
            e.index = remap.at(e.index);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<BinaryOpExpr>>) {
            remapRefs(e->lhs, remap);
            remapRefs(e->rhs, remap);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<LikeOpExpr>>) {
            remapRefs(e->input, remap);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<UnaryOpExpr>>) {
            remapRefs(e->input, remap);
        }
    }, expr);
}

// Try to prune the scan under one aggregate. The supported shape is
// HashAggregate -> Filter* -> FileScan(parquet, unpartitioned, not yet
// projected); anything else leaves the plan untouched.
void tryPruneAggregate(HashAggregateNode& agg) {
    std::vector<FilterNode*> filters;
    PlanNode* cursor = &agg.child;
    FileScanNode* scan = nullptr;
    while (scan == nullptr) {
        if (auto* f = std::get_if<std::unique_ptr<FilterNode>>(cursor)) {
            filters.push_back(f->get());
            cursor = &(*f)->child;
        } else if (auto* fs = std::get_if<FileScanNode>(cursor)) {
            scan = fs;
        } else {
            return;  // unsupported chain (join, union, scan-by-id, ...)
        }
    }
    if (scan->format != "parquet") return;
    if (scan->partition_columns.numFields() > 0) return;  // index space shifts
    if (!scan->projected_columns.empty()) return;         // already pruned

    std::set<int32_t> used;
    for (const auto& k : agg.group_keys) collectRefs(k, used);
    for (const auto& a : agg.aggregates) {
        if (a.input.has_value()) collectRefs(*a.input, used);
    }
    for (const FilterNode* f : filters) collectRefs(f->predicate, used);
    // No referenced columns (pure COUNT(*)): a zero-column scan has no row
    // count carrier in ColumnBatch — keep the full scan.
    if (used.empty()) return;

    scan->projected_columns.assign(used.begin(), used.end());  // ascending
    std::unordered_map<int32_t, int32_t> remap;
    for (size_t i = 0; i < scan->projected_columns.size(); ++i) {
        remap[scan->projected_columns[i]] = static_cast<int32_t>(i);
    }
    // Scan schema (when a host supplied one) must match the pruned output.
    if (scan->schema.has_value()) {
        std::vector<Field> fields;
        fields.reserve(scan->projected_columns.size());
        for (const int32_t idx : scan->projected_columns) {
            if (idx < 0 ||
                static_cast<size_t>(idx) >= scan->schema->numFields()) {
                scan->projected_columns.clear();
                return;  // refs outside the declared schema — don't touch
            }
            fields.push_back(scan->schema->fields()[static_cast<size_t>(idx)]);
        }
        scan->schema = Schema(std::move(fields));
    }
    for (auto& k : agg.group_keys) remapRefs(k, remap);
    for (auto& a : agg.aggregates) {
        if (a.input.has_value()) remapRefs(*a.input, remap);
    }
    for (FilterNode* f : filters) remapRefs(f->predicate, remap);
}

void walk(PlanNode& node) {
    std::visit([&](auto& n) {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<FilterNode>>) {
            walk(n->child);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ProjectNode>>) {
            walk(n->child);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashAggregateNode>>) {
            tryPruneAggregate(*n);
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

void pruneScanColumns(PlanNode& root) {
    static const bool enabled = [] {
        const char* env = std::getenv("QUANTUM_PRUNE_COLUMNS");
        return env == nullptr || std::string_view(env) != "0";
    }();
    if (enabled) walk(root);
}

} // namespace quantum::plan
