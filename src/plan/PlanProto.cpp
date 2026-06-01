#include "plan/PlanProto.hpp"

#include <fmt/format.h>

#include "common/Status.hpp"
#include "quantum_plan.pb.h"

namespace quantum::plan {

namespace {

// ============================================================================
// Type mapping
// ============================================================================

proto::TypeId encodeType(DataType t) {
    switch (t.id) {
        case TypeId::BOOL:    return proto::TYPE_BOOL;
        case TypeId::INT32:   return proto::TYPE_INT32;
        case TypeId::INT64:   return proto::TYPE_INT64;
        case TypeId::FLOAT64: return proto::TYPE_FLOAT64;
        case TypeId::STRING:  return proto::TYPE_STRING;
    }
    raise(ErrorCode::Internal, "PlanProto: unknown TypeId");
}

DataType decodeType(proto::TypeId t) {
    switch (t) {
        case proto::TYPE_BOOL:    return Bool;
        case proto::TYPE_INT32:   return Int32;
        case proto::TYPE_INT64:   return Int64;
        case proto::TYPE_FLOAT64: return Float64;
        case proto::TYPE_STRING:  return DataType{TypeId::STRING};
        default: raise(ErrorCode::InvalidArgument, "PlanProto: unknown proto TypeId");
    }
}

// ============================================================================
// ScalarExpr
// ============================================================================

void encodeLiteral(const LiteralExpr& lit, proto::Literal* out) {
    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>)         out->set_bool_value(v);
        else if constexpr (std::is_same_v<T, int64_t>) out->set_int64_value(v);
        else if constexpr (std::is_same_v<T, double>)  out->set_double_value(v);
        else if constexpr (std::is_same_v<T, std::string>) out->set_string_value(v);
    }, lit.value);
}

LiteralExpr decodeLiteral(const proto::Literal& msg) {
    LiteralExpr out;
    switch (msg.value_case()) {
        case proto::Literal::kBoolValue:   out.value = msg.bool_value(); break;
        case proto::Literal::kInt64Value:  out.value = msg.int64_value(); break;
        case proto::Literal::kDoubleValue: out.value = msg.double_value(); break;
        case proto::Literal::kStringValue: out.value = msg.string_value(); break;
        case proto::Literal::VALUE_NOT_SET:
            raise(ErrorCode::InvalidArgument, "PlanProto: Literal has no value set");
    }
    return out;
}

proto::BinaryOpType encodeBinOp(BinaryOpType op) {
    switch (op) {
        case BinaryOpType::Add: return proto::BINOP_ADD;
        case BinaryOpType::Sub: return proto::BINOP_SUB;
        case BinaryOpType::Mul: return proto::BINOP_MUL;
        case BinaryOpType::Eq:  return proto::BINOP_EQ;
        case BinaryOpType::Lt:  return proto::BINOP_LT;
        case BinaryOpType::Gt:  return proto::BINOP_GT;
        case BinaryOpType::Lte: return proto::BINOP_LTE;
        case BinaryOpType::Gte: return proto::BINOP_GTE;
        case BinaryOpType::And: return proto::BINOP_AND;
        case BinaryOpType::Or:  return proto::BINOP_OR;
    }
    raise(ErrorCode::Internal, "PlanProto: unknown BinaryOpType");
}

BinaryOpType decodeBinOp(proto::BinaryOpType op) {
    switch (op) {
        case proto::BINOP_ADD: return BinaryOpType::Add;
        case proto::BINOP_SUB: return BinaryOpType::Sub;
        case proto::BINOP_MUL: return BinaryOpType::Mul;
        case proto::BINOP_EQ:  return BinaryOpType::Eq;
        case proto::BINOP_LT:  return BinaryOpType::Lt;
        case proto::BINOP_GT:  return BinaryOpType::Gt;
        case proto::BINOP_LTE: return BinaryOpType::Lte;
        case proto::BINOP_GTE: return BinaryOpType::Gte;
        case proto::BINOP_AND: return BinaryOpType::And;
        case proto::BINOP_OR:  return BinaryOpType::Or;
        default: raise(ErrorCode::InvalidArgument, "PlanProto: unknown proto BinaryOpType");
    }
}

} // namespace

void encodeScalarExpr(const ScalarExpr& expr, proto::ScalarExpr* out) {
    std::visit([&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, LiteralExpr>) {
            encodeLiteral(x, out->mutable_literal());
        } else if constexpr (std::is_same_v<T, ColumnRefExpr>) {
            out->mutable_column_ref()->set_index(x.index);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<BinaryOpExpr>>) {
            auto* bm = out->mutable_binary();
            bm->set_op(encodeBinOp(x->op));
            encodeScalarExpr(x->lhs, bm->mutable_lhs());
            encodeScalarExpr(x->rhs, bm->mutable_rhs());
        } else if constexpr (std::is_same_v<T, std::unique_ptr<LikeOpExpr>>) {
            auto* lm = out->mutable_like();
            encodeScalarExpr(x->input, lm->mutable_input());
            lm->set_pattern(x->pattern);
        }
    }, expr);
}

ScalarExpr decodeScalarExpr(const proto::ScalarExpr& msg) {
    switch (msg.expr_case()) {
        case proto::ScalarExpr::kLiteral:
            return decodeLiteral(msg.literal());
        case proto::ScalarExpr::kColumnRef:
            return ColumnRefExpr{msg.column_ref().index()};
        case proto::ScalarExpr::kBinary: {
            auto e = std::make_unique<BinaryOpExpr>();
            e->op  = decodeBinOp(msg.binary().op());
            e->lhs = decodeScalarExpr(msg.binary().lhs());
            e->rhs = decodeScalarExpr(msg.binary().rhs());
            return ScalarExpr{std::move(e)};
        }
        case proto::ScalarExpr::kLike: {
            auto e = std::make_unique<LikeOpExpr>();
            e->input   = decodeScalarExpr(msg.like().input());
            e->pattern = msg.like().pattern();
            return ScalarExpr{std::move(e)};
        }
        case proto::ScalarExpr::EXPR_NOT_SET:
            raise(ErrorCode::InvalidArgument, "PlanProto: ScalarExpr is empty");
    }
    raise(ErrorCode::Internal, "PlanProto: unreachable ScalarExpr case");
}

namespace {

// ============================================================================
// Schema / Aggregate / JoinType
// ============================================================================

void encodeSchema(const Schema& s, proto::Schema* out) {
    for (const auto& f : s.fields()) {
        auto* fm = out->add_fields();
        fm->set_name(f.name);
        fm->set_type(encodeType(f.type));
    }
}

Schema decodeSchema(const proto::Schema& msg) {
    std::vector<Field> fields;
    fields.reserve(static_cast<size_t>(msg.fields_size()));
    for (const auto& fm : msg.fields()) {
        fields.push_back(Field{fm.name(), decodeType(fm.type())});
    }
    return Schema(std::move(fields));
}

proto::AggregateKind encodeAggKind(AggregateKind k) {
    switch (k) {
        case AggregateKind::Sum:       return proto::AGG_SUM;
        case AggregateKind::CountStar: return proto::AGG_COUNT_STAR;
        case AggregateKind::Min:       return proto::AGG_MIN;
        case AggregateKind::Max:       return proto::AGG_MAX;
        case AggregateKind::Avg:       return proto::AGG_AVG;
    }
    raise(ErrorCode::Internal, "PlanProto: unknown AggregateKind");
}

AggregateKind decodeAggKind(proto::AggregateKind k) {
    switch (k) {
        case proto::AGG_SUM:        return AggregateKind::Sum;
        case proto::AGG_COUNT_STAR: return AggregateKind::CountStar;
        case proto::AGG_MIN:        return AggregateKind::Min;
        case proto::AGG_MAX:        return AggregateKind::Max;
        case proto::AGG_AVG:        return AggregateKind::Avg;
        default: raise(ErrorCode::InvalidArgument, "PlanProto: unknown proto AggregateKind");
    }
}

void encodeAggregate(const AggregateSpec& spec, proto::AggregateSpec* out) {
    out->set_kind(encodeAggKind(spec.kind));
    if (spec.kind != AggregateKind::CountStar) {
        out->set_input_type(encodeType(spec.inputType));
    }
    if (spec.input.has_value()) {
        encodeScalarExpr(*spec.input, out->mutable_input());
    }
}

AggregateSpec decodeAggregate(const proto::AggregateSpec& msg) {
    AggregateSpec spec;
    spec.kind = decodeAggKind(msg.kind());
    if (spec.kind != AggregateKind::CountStar) {
        spec.inputType = decodeType(msg.input_type());
    } else {
        spec.inputType = Int64;  // unused
    }
    if (msg.has_input()) {
        spec.input = decodeScalarExpr(msg.input());
    }
    return spec;
}

proto::JoinType encodeJoinType(JoinTypeIR t) {
    switch (t) {
        case JoinTypeIR::Inner:      return proto::JOIN_INNER;
        case JoinTypeIR::LeftOuter:  return proto::JOIN_LEFT_OUTER;
        case JoinTypeIR::RightOuter: return proto::JOIN_RIGHT_OUTER;
        case JoinTypeIR::FullOuter:  return proto::JOIN_FULL_OUTER;
    }
    raise(ErrorCode::Internal, "PlanProto: unknown JoinTypeIR");
}

JoinTypeIR decodeJoinType(proto::JoinType t) {
    switch (t) {
        case proto::JOIN_INNER:       return JoinTypeIR::Inner;
        case proto::JOIN_LEFT_OUTER:  return JoinTypeIR::LeftOuter;
        case proto::JOIN_RIGHT_OUTER: return JoinTypeIR::RightOuter;
        case proto::JOIN_FULL_OUTER:  return JoinTypeIR::FullOuter;
        default: raise(ErrorCode::InvalidArgument, "PlanProto: unknown proto JoinType");
    }
}

} // namespace

void encodePlan(const PlanNode& node, proto::PlanNode* out) {
    std::visit([&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, ScanNode>) {
            auto* m = out->mutable_scan();
            m->set_source_id(x.source_id);
            encodeSchema(x.schema, m->mutable_schema());
        } else if constexpr (std::is_same_v<T, FileScanNode>) {
            auto* m = out->mutable_file_scan();
            for (const auto& p : x.paths) m->add_paths(p);
            m->set_format(x.format);
            if (x.schema.has_value()) encodeSchema(*x.schema, m->mutable_schema());
        } else if constexpr (std::is_same_v<T, std::unique_ptr<FilterNode>>) {
            auto* m = out->mutable_filter();
            encodePlan(x->child, m->mutable_child());
            encodeScalarExpr(x->predicate, m->mutable_predicate());
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ProjectNode>>) {
            auto* m = out->mutable_project();
            encodePlan(x->child, m->mutable_child());
            for (const auto& e : x->exprs) encodeScalarExpr(e, m->add_exprs());
            for (const auto& n : x->output_names) m->add_output_names(n);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashAggregateNode>>) {
            auto* m = out->mutable_hash_agg();
            encodePlan(x->child, m->mutable_child());
            for (const auto& k : x->group_keys) encodeScalarExpr(k, m->add_group_keys());
            for (const auto& a : x->aggregates) encodeAggregate(a, m->add_aggregates());
            for (const auto& n : x->output_names) m->add_output_names(n);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashJoinNode>>) {
            auto* m = out->mutable_hash_join();
            encodePlan(x->build, m->mutable_build());
            encodePlan(x->probe, m->mutable_probe());
            for (const auto& k : x->build_keys) encodeScalarExpr(k, m->add_build_keys());
            for (const auto& k : x->probe_keys) encodeScalarExpr(k, m->add_probe_keys());
            m->set_join_type(encodeJoinType(x->join_type));
            for (const auto& n : x->output_names) m->add_output_names(n);
        }
    }, node);
}

PlanNode decodePlan(const proto::PlanNode& msg) {
    switch (msg.node_case()) {
        case proto::PlanNode::kScan:
            return ScanNode{
                msg.scan().source_id(),
                decodeSchema(msg.scan().schema())};
        case proto::PlanNode::kFileScan: {
            FileScanNode n;
            for (const auto& p : msg.file_scan().paths()) n.paths.push_back(p);
            n.format = msg.file_scan().format();
            if (msg.file_scan().has_schema()) n.schema = decodeSchema(msg.file_scan().schema());
            return PlanNode{std::move(n)};
        }
        case proto::PlanNode::kFilter: {
            auto n = std::make_unique<FilterNode>();
            n->child     = decodePlan(msg.filter().child());
            n->predicate = decodeScalarExpr(msg.filter().predicate());
            return PlanNode{std::move(n)};
        }
        case proto::PlanNode::kProject: {
            auto n = std::make_unique<ProjectNode>();
            n->child = decodePlan(msg.project().child());
            for (const auto& e : msg.project().exprs()) n->exprs.push_back(decodeScalarExpr(e));
            for (const auto& s : msg.project().output_names()) n->output_names.push_back(s);
            return PlanNode{std::move(n)};
        }
        case proto::PlanNode::kHashAgg: {
            auto n = std::make_unique<HashAggregateNode>();
            n->child = decodePlan(msg.hash_agg().child());
            for (const auto& k : msg.hash_agg().group_keys()) n->group_keys.push_back(decodeScalarExpr(k));
            for (const auto& a : msg.hash_agg().aggregates()) n->aggregates.push_back(decodeAggregate(a));
            for (const auto& s : msg.hash_agg().output_names()) n->output_names.push_back(s);
            return PlanNode{std::move(n)};
        }
        case proto::PlanNode::kHashJoin: {
            auto n = std::make_unique<HashJoinNode>();
            n->build = decodePlan(msg.hash_join().build());
            n->probe = decodePlan(msg.hash_join().probe());
            for (const auto& k : msg.hash_join().build_keys()) n->build_keys.push_back(decodeScalarExpr(k));
            for (const auto& k : msg.hash_join().probe_keys()) n->probe_keys.push_back(decodeScalarExpr(k));
            n->join_type = decodeJoinType(msg.hash_join().join_type());
            for (const auto& s : msg.hash_join().output_names()) n->output_names.push_back(s);
            return PlanNode{std::move(n)};
        }
        case proto::PlanNode::NODE_NOT_SET:
            raise(ErrorCode::InvalidArgument, "PlanProto: PlanNode is empty");
    }
    raise(ErrorCode::Internal, "PlanProto: unreachable PlanNode case");
}

std::string planToProtoBytes(const PlanNode& node) {
    proto::PlanNode msg;
    encodePlan(node, &msg);
    return msg.SerializeAsString();
}

PlanNode planFromProtoBytes(std::string_view bytes) {
    proto::PlanNode msg;
    if (!msg.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
        raise(ErrorCode::InvalidArgument, "PlanProto: failed to parse PlanNode bytes");
    }
    return decodePlan(msg);
}

std::string scalarExprToProtoBytes(const ScalarExpr& expr) {
    proto::ScalarExpr msg;
    encodeScalarExpr(expr, &msg);
    return msg.SerializeAsString();
}

ScalarExpr scalarExprFromProtoBytes(std::string_view bytes) {
    proto::ScalarExpr msg;
    if (!msg.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
        raise(ErrorCode::InvalidArgument, "PlanProto: failed to parse ScalarExpr bytes");
    }
    return decodeScalarExpr(msg);
}

} // namespace quantum::plan
