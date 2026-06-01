#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/DataType.hpp"

namespace quantum::plan {

// ============================================================================
// Scalar expressions.
//
// Tagged union over the expression types the engine currently supports.
// Recursive shapes (BinaryOpExpr / LikeOpExpr) are held by unique_ptr to keep
// the variant size bounded and allow forward declarations.
// ============================================================================

struct LiteralExpr {
    using Value = std::variant<bool, int64_t, double, std::string>;
    Value value;
};

struct ColumnRefExpr {
    int32_t index = 0;
};

struct BinaryOpExpr;
struct LikeOpExpr;

using ScalarExpr = std::variant<
    LiteralExpr,
    ColumnRefExpr,
    std::unique_ptr<BinaryOpExpr>,
    std::unique_ptr<LikeOpExpr>
>;

enum class BinaryOpType { Add, Sub, Mul, Eq, Lt, Gt };

struct BinaryOpExpr {
    BinaryOpType op;
    ScalarExpr   lhs;
    ScalarExpr   rhs;
};

struct LikeOpExpr {
    ScalarExpr  input;
    std::string pattern;
};

// ============================================================================
// Aggregate spec.
//
// Each entry names the aggregate kind, the numeric variant for typed
// aggregates (input type), and the optional input expression. CountStar is
// the only aggregate with no input.
// ============================================================================

enum class AggregateKind { Sum, CountStar, Min, Max, Avg };

struct AggregateSpec {
    AggregateKind             kind;
    DataType                  inputType;  // ignored for CountStar
    std::optional<ScalarExpr> input;
};

// ============================================================================
// Plan nodes.
// ============================================================================

struct ScanNode {
    std::string source_id;   // looked up in the SourceCatalog at build time
    Schema      schema;
};

// FileScan: native engine reads the file(s) directly via io::FileSystem.
// `format` selects the reader ("parquet" or "delta").
struct FileScanNode {
    std::vector<std::string>   paths;
    std::string                format;
    std::optional<Schema>      schema;  // unset => derive from file metadata
};

struct FilterNode;
struct ProjectNode;
struct HashAggregateNode;
struct HashJoinNode;

using PlanNode = std::variant<
    ScanNode,
    FileScanNode,
    std::unique_ptr<FilterNode>,
    std::unique_ptr<ProjectNode>,
    std::unique_ptr<HashAggregateNode>,
    std::unique_ptr<HashJoinNode>
>;

struct FilterNode {
    PlanNode    child;
    ScalarExpr  predicate;
};

struct ProjectNode {
    PlanNode                  child;
    std::vector<ScalarExpr>   exprs;
    std::vector<std::string>  output_names;
};

struct HashAggregateNode {
    PlanNode                    child;
    std::vector<ScalarExpr>     group_keys;
    std::vector<AggregateSpec>  aggregates;
    std::vector<std::string>    output_names;
};

enum class JoinTypeIR { Inner, LeftOuter, RightOuter, FullOuter };

struct HashJoinNode {
    PlanNode                    build;
    PlanNode                    probe;
    std::vector<ScalarExpr>     build_keys;
    std::vector<ScalarExpr>     probe_keys;
    JoinTypeIR                  join_type;
    std::vector<std::string>    output_names;
};

} // namespace quantum::plan
