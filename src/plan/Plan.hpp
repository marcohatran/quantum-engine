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
struct UnaryOpExpr;

using ScalarExpr = std::variant<
    LiteralExpr,
    ColumnRefExpr,
    std::unique_ptr<BinaryOpExpr>,
    std::unique_ptr<LikeOpExpr>,
    std::unique_ptr<UnaryOpExpr>
>;

enum class BinaryOpType { Add, Sub, Mul, Eq, Lt, Gt, Lte, Gte, And, Or };
enum class UnaryOpType  { Not, IsNull };

struct BinaryOpExpr {
    BinaryOpType op;
    ScalarExpr   lhs;
    ScalarExpr   rhs;
};

struct LikeOpExpr {
    ScalarExpr  input;
    std::string pattern;
};

struct UnaryOpExpr {
    UnaryOpType op;
    ScalarExpr  input;
};

// ============================================================================
// Aggregate spec.
//
// Each entry names the aggregate kind, the numeric variant for typed
// aggregates (input type), and the optional input expression. CountStar is
// the only aggregate with no input.
// ============================================================================

enum class AggregateKind { Sum, CountStar, Min, Max, Avg, Count, CountDistinct };

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
//
// Hive-style partition columns are carried by `partition_columns` (field
// names + types) and `partition_values` (one row per path, each row one
// LiteralExpr per partition column). When non-empty, each ParquetReader is
// wrapped in a PartitionedFileReader that synthesizes the partition columns
// as constants appended to every batch. Empty => no partition columns.
struct FileScanNode {
    std::vector<std::string>                paths;
    std::string                             format;
    std::optional<Schema>                   schema;
    Schema                                  partition_columns;
    std::vector<std::vector<LiteralExpr>>   partition_values;  // [path_idx][col_idx]
};

struct FilterNode;
struct ProjectNode;
struct HashAggregateNode;
struct HashJoinNode;
struct LimitNode;

using PlanNode = std::variant<
    ScanNode,
    FileScanNode,
    std::unique_ptr<FilterNode>,
    std::unique_ptr<ProjectNode>,
    std::unique_ptr<HashAggregateNode>,
    std::unique_ptr<HashJoinNode>,
    std::unique_ptr<LimitNode>
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

struct LimitNode {
    PlanNode child;
    int64_t  limit = 0;
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
