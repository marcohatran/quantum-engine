#include "plan/PlanJson.hpp"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/Status.hpp"

namespace quantum::plan {

using json = nlohmann::json;

namespace {

// ============================================================================
// Type codec
// ============================================================================

std::string typeToString(DataType t) {
    switch (t.id) {
        case TypeId::BOOL:    return "bool";
        case TypeId::INT32:   return "int32";
        case TypeId::INT64:   return "int64";
        case TypeId::FLOAT64: return "float64";
        case TypeId::STRING:  return "string";
    }
    raise(ErrorCode::Internal, "PlanJson: unknown TypeId");
}

DataType typeFromString(std::string_view s) {
    if (s == "bool")    return Bool;
    if (s == "int32")   return Int32;
    if (s == "int64")   return Int64;
    if (s == "float64") return Float64;
    if (s == "string")  return DataType{TypeId::STRING};
    raise(ErrorCode::InvalidArgument, "PlanJson: unknown type '{}'", s);
}

// ============================================================================
// Forward decls for recursion
// ============================================================================

json exprToJson(const ScalarExpr& e);
ScalarExpr exprFromJson(const json& j);
json nodeToJson(const PlanNode& n);
PlanNode nodeFromJson(const json& j);

// ============================================================================
// ScalarExpr <-> JSON
// ============================================================================

json literalToJson(const LiteralExpr& lit) {
    json out = {{"kind", "literal"}};
    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>)         { out["type"] = "bool";    out["value"] = v; }
        else if constexpr (std::is_same_v<T, int64_t>) { out["type"] = "int64";   out["value"] = v; }
        else if constexpr (std::is_same_v<T, double>)  { out["type"] = "float64"; out["value"] = v; }
        else if constexpr (std::is_same_v<T, std::string>) {
            out["type"] = "string"; out["value"] = v;
        }
    }, lit.value);
    return out;
}

LiteralExpr literalFromJson(const json& j) {
    LiteralExpr out;
    const std::string type = j.at("type").get<std::string>();
    if      (type == "bool")    out.value = j.at("value").get<bool>();
    else if (type == "int64")   out.value = j.at("value").get<int64_t>();
    else if (type == "float64") out.value = j.at("value").get<double>();
    else if (type == "string")  out.value = j.at("value").get<std::string>();
    else raise(ErrorCode::InvalidArgument, "PlanJson: bad literal type '{}'", type);
    return out;
}

const char* binaryOpToString(BinaryOpType op) {
    switch (op) {
        case BinaryOpType::Add: return "add";
        case BinaryOpType::Sub: return "sub";
        case BinaryOpType::Mul: return "mul";
        case BinaryOpType::Eq:  return "eq";
        case BinaryOpType::Lt:  return "lt";
        case BinaryOpType::Gt:  return "gt";
    }
    raise(ErrorCode::Internal, "PlanJson: unknown BinaryOpType");
}

BinaryOpType binaryOpFromString(std::string_view s) {
    if (s == "add") return BinaryOpType::Add;
    if (s == "sub") return BinaryOpType::Sub;
    if (s == "mul") return BinaryOpType::Mul;
    if (s == "eq")  return BinaryOpType::Eq;
    if (s == "lt")  return BinaryOpType::Lt;
    if (s == "gt")  return BinaryOpType::Gt;
    raise(ErrorCode::InvalidArgument, "PlanJson: unknown binary op '{}'", s);
}

json exprToJson(const ScalarExpr& e) {
    return std::visit([](const auto& x) -> json {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, LiteralExpr>) {
            return literalToJson(x);
        } else if constexpr (std::is_same_v<T, ColumnRefExpr>) {
            return json{{"kind", "column_ref"}, {"index", x.index}};
        } else if constexpr (std::is_same_v<T, std::unique_ptr<BinaryOpExpr>>) {
            return json{
                {"kind", "binary"},
                {"op",   binaryOpToString(x->op)},
                {"lhs",  exprToJson(x->lhs)},
                {"rhs",  exprToJson(x->rhs)},
            };
        } else if constexpr (std::is_same_v<T, std::unique_ptr<LikeOpExpr>>) {
            return json{
                {"kind",    "like"},
                {"input",   exprToJson(x->input)},
                {"pattern", x->pattern},
            };
        }
    }, e);
}

ScalarExpr exprFromJson(const json& j) {
    const std::string kind = j.at("kind").get<std::string>();
    if (kind == "literal") return literalFromJson(j);
    if (kind == "column_ref") {
        return ColumnRefExpr{j.at("index").get<int32_t>()};
    }
    if (kind == "binary") {
        auto e = std::make_unique<BinaryOpExpr>();
        e->op  = binaryOpFromString(j.at("op").get<std::string>());
        e->lhs = exprFromJson(j.at("lhs"));
        e->rhs = exprFromJson(j.at("rhs"));
        return ScalarExpr{std::move(e)};
    }
    if (kind == "like") {
        auto e = std::make_unique<LikeOpExpr>();
        e->input   = exprFromJson(j.at("input"));
        e->pattern = j.at("pattern").get<std::string>();
        return ScalarExpr{std::move(e)};
    }
    raise(ErrorCode::InvalidArgument, "PlanJson: unknown expr kind '{}'", kind);
}

// ============================================================================
// Schema / Aggregate <-> JSON
// ============================================================================

json schemaToJson(const Schema& s) {
    json arr = json::array();
    for (const auto& f : s.fields()) {
        arr.push_back(json{{"name", f.name}, {"type", typeToString(f.type)}});
    }
    return arr;
}

Schema schemaFromJson(const json& j) {
    std::vector<Field> fields;
    fields.reserve(j.size());
    for (const auto& fj : j) {
        fields.push_back(Field{
            fj.at("name").get<std::string>(),
            typeFromString(fj.at("type").get<std::string>())});
    }
    return Schema(std::move(fields));
}

const char* aggKindToString(AggregateKind k) {
    switch (k) {
        case AggregateKind::Sum:       return "sum";
        case AggregateKind::CountStar: return "count_star";
        case AggregateKind::Min:       return "min";
        case AggregateKind::Max:       return "max";
        case AggregateKind::Avg:       return "avg";
    }
    raise(ErrorCode::Internal, "PlanJson: unknown AggregateKind");
}

AggregateKind aggKindFromString(std::string_view s) {
    if (s == "sum")        return AggregateKind::Sum;
    if (s == "count_star") return AggregateKind::CountStar;
    if (s == "min")        return AggregateKind::Min;
    if (s == "max")        return AggregateKind::Max;
    if (s == "avg")        return AggregateKind::Avg;
    raise(ErrorCode::InvalidArgument, "PlanJson: unknown aggregate kind '{}'", s);
}

json aggregateToJson(const AggregateSpec& a) {
    json out = {{"kind", aggKindToString(a.kind)}};
    if (a.kind != AggregateKind::CountStar) {
        out["input_type"] = typeToString(a.inputType);
        if (a.input.has_value()) out["input"] = exprToJson(*a.input);
    }
    return out;
}

AggregateSpec aggregateFromJson(const json& j) {
    AggregateSpec spec;
    spec.kind = aggKindFromString(j.at("kind").get<std::string>());
    if (spec.kind != AggregateKind::CountStar) {
        spec.inputType = typeFromString(j.at("input_type").get<std::string>());
        spec.input     = exprFromJson(j.at("input"));
    } else {
        spec.inputType = Int64;
    }
    return spec;
}

const char* joinTypeToString(JoinTypeIR t) {
    switch (t) {
        case JoinTypeIR::Inner:      return "inner";
        case JoinTypeIR::LeftOuter:  return "left_outer";
        case JoinTypeIR::RightOuter: return "right_outer";
        case JoinTypeIR::FullOuter:  return "full_outer";
    }
    raise(ErrorCode::Internal, "PlanJson: unknown JoinTypeIR");
}

JoinTypeIR joinTypeFromString(std::string_view s) {
    if (s == "inner")       return JoinTypeIR::Inner;
    if (s == "left_outer")  return JoinTypeIR::LeftOuter;
    if (s == "right_outer") return JoinTypeIR::RightOuter;
    if (s == "full_outer")  return JoinTypeIR::FullOuter;
    raise(ErrorCode::InvalidArgument, "PlanJson: unknown join type '{}'", s);
}

// ============================================================================
// PlanNode <-> JSON
// ============================================================================

json exprListToJson(const std::vector<ScalarExpr>& v) {
    json arr = json::array();
    for (const auto& e : v) arr.push_back(exprToJson(e));
    return arr;
}

std::vector<ScalarExpr> exprListFromJson(const json& j) {
    std::vector<ScalarExpr> out;
    out.reserve(j.size());
    for (const auto& e : j) out.push_back(exprFromJson(e));
    return out;
}

json nodeToJson(const PlanNode& n) {
    return std::visit([](const auto& x) -> json {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, ScanNode>) {
            return json{
                {"kind",      "scan"},
                {"source_id", x.source_id},
                {"schema",    schemaToJson(x.schema)},
            };
        } else if constexpr (std::is_same_v<T, std::unique_ptr<FilterNode>>) {
            return json{
                {"kind",      "filter"},
                {"child",     nodeToJson(x->child)},
                {"predicate", exprToJson(x->predicate)},
            };
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ProjectNode>>) {
            return json{
                {"kind",         "project"},
                {"child",        nodeToJson(x->child)},
                {"exprs",        exprListToJson(x->exprs)},
                {"output_names", x->output_names},
            };
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashAggregateNode>>) {
            json aggs = json::array();
            for (const auto& a : x->aggregates) aggs.push_back(aggregateToJson(a));
            return json{
                {"kind",         "hash_agg"},
                {"child",        nodeToJson(x->child)},
                {"group_keys",   exprListToJson(x->group_keys)},
                {"aggregates",   aggs},
                {"output_names", x->output_names},
            };
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashJoinNode>>) {
            return json{
                {"kind",         "hash_join"},
                {"build",        nodeToJson(x->build)},
                {"probe",        nodeToJson(x->probe)},
                {"build_keys",   exprListToJson(x->build_keys)},
                {"probe_keys",   exprListToJson(x->probe_keys)},
                {"join_type",    joinTypeToString(x->join_type)},
                {"output_names", x->output_names},
            };
        }
    }, n);
}

PlanNode nodeFromJson(const json& j) {
    const std::string kind = j.at("kind").get<std::string>();
    if (kind == "scan") {
        return ScanNode{
            j.at("source_id").get<std::string>(),
            schemaFromJson(j.at("schema"))};
    }
    if (kind == "filter") {
        auto n = std::make_unique<FilterNode>();
        n->child     = nodeFromJson(j.at("child"));
        n->predicate = exprFromJson(j.at("predicate"));
        return PlanNode{std::move(n)};
    }
    if (kind == "project") {
        auto n = std::make_unique<ProjectNode>();
        n->child        = nodeFromJson(j.at("child"));
        n->exprs        = exprListFromJson(j.at("exprs"));
        n->output_names = j.at("output_names").get<std::vector<std::string>>();
        return PlanNode{std::move(n)};
    }
    if (kind == "hash_agg") {
        auto n = std::make_unique<HashAggregateNode>();
        n->child      = nodeFromJson(j.at("child"));
        n->group_keys = exprListFromJson(j.at("group_keys"));
        for (const auto& aj : j.at("aggregates")) {
            n->aggregates.push_back(aggregateFromJson(aj));
        }
        n->output_names = j.at("output_names").get<std::vector<std::string>>();
        return PlanNode{std::move(n)};
    }
    if (kind == "hash_join") {
        auto n = std::make_unique<HashJoinNode>();
        n->build        = nodeFromJson(j.at("build"));
        n->probe        = nodeFromJson(j.at("probe"));
        n->build_keys   = exprListFromJson(j.at("build_keys"));
        n->probe_keys   = exprListFromJson(j.at("probe_keys"));
        n->join_type    = joinTypeFromString(j.at("join_type").get<std::string>());
        n->output_names = j.at("output_names").get<std::vector<std::string>>();
        return PlanNode{std::move(n)};
    }
    raise(ErrorCode::InvalidArgument, "PlanJson: unknown node kind '{}'", kind);
}

} // namespace

std::string planToJson(const PlanNode& node) {
    return nodeToJson(node).dump();
}

PlanNode planFromJson(std::string_view js) {
    auto j = json::parse(js);
    return nodeFromJson(j);
}

std::string scalarExprToJson(const ScalarExpr& expr) {
    return exprToJson(expr).dump();
}

ScalarExpr scalarExprFromJson(std::string_view js) {
    auto j = json::parse(js);
    return exprFromJson(j);
}

} // namespace quantum::plan
