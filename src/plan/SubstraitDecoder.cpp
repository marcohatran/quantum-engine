#include "plan/SubstraitDecoder.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <substrait/plan.pb.h>

#include "common/Status.hpp"

namespace quantum::plan {

namespace {

DataType typeFrom(const substrait::Type& t) {
    switch (t.kind_case()) {
        case substrait::Type::kBool:   return Bool;
        case substrait::Type::kI32:    return Int32;
        case substrait::Type::kI64:    return Int64;
        case substrait::Type::kFp64:   return Float64;
        case substrait::Type::kString: return DataType{TypeId::STRING};
        default:
            raise(ErrorCode::NotImplemented,
                  "SubstraitDecoder: unsupported type kind {}",
                  static_cast<int>(t.kind_case()));
    }
}

Schema schemaFrom(const substrait::NamedStruct& ns) {
    const auto& types = ns.struct_().types();
    if (ns.names_size() != types.size()) {
        raise(ErrorCode::NotImplemented,
              "SubstraitDecoder: nested base_schema not supported "
              "({} names, {} types)", ns.names_size(), types.size());
    }
    std::vector<Field> fields;
    fields.reserve(static_cast<size_t>(types.size()));
    for (int i = 0; i < types.size(); ++i) {
        fields.push_back(Field{ns.names(i), typeFrom(types[i])});
    }
    return Schema(std::move(fields));
}

// A decoded relation: the engine plan node plus its output schema (needed
// for expression typing further up the tree).
struct Decoded {
    PlanNode node;
    Schema   schema;
};

class Decoder {
public:
    explicit Decoder(const substrait::Plan& plan) {
        for (const auto& ext : plan.extensions()) {
            if (!ext.has_extension_function()) continue;
            std::string name = ext.extension_function().name();
            // Names may carry a signature suffix ("lt:any_any").
            const auto colon = name.find(':');
            if (colon != std::string::npos) name = name.substr(0, colon);
            functions_[ext.extension_function().function_anchor()] = std::move(name);
        }
    }

    Decoded decodeRel(const substrait::Rel& rel) {
        switch (rel.rel_type_case()) {
            case substrait::Rel::kRead:      return decodeRead(rel.read());
            case substrait::Rel::kFilter:    return decodeFilter(rel.filter());
            case substrait::Rel::kProject:   return decodeProject(rel.project());
            case substrait::Rel::kAggregate: return decodeAggregate(rel.aggregate());
            case substrait::Rel::kFetch:     return decodeFetch(rel.fetch());
            case substrait::Rel::kSort:      return decodeSort(rel.sort());
            default:
                raise(ErrorCode::NotImplemented,
                      "SubstraitDecoder: unsupported relation type {}",
                      static_cast<int>(rel.rel_type_case()));
        }
    }

private:
    const std::string& functionName(uint32_t anchor) const {
        auto it = functions_.find(anchor);
        if (it == functions_.end()) {
            raise(ErrorCode::InvalidArgument,
                  "SubstraitDecoder: function anchor {} has no extension "
                  "declaration", anchor);
        }
        return it->second;
    }

    static int32_t fieldIndexOf(const substrait::Expression& e) {
        if (!e.has_selection() ||
            !e.selection().has_direct_reference() ||
            !e.selection().direct_reference().has_struct_field()) {
            raise(ErrorCode::NotImplemented,
                  "SubstraitDecoder: expected a direct struct-field reference");
        }
        return e.selection().direct_reference().struct_field().field();
    }

    ScalarExpr decodeExpr(const substrait::Expression& e, const Schema& in) {
        switch (e.rex_type_case()) {
            case substrait::Expression::kLiteral: {
                const auto& l = e.literal();
                switch (l.literal_type_case()) {
                    case substrait::Expression::Literal::kBoolean:
                        return LiteralExpr{l.boolean()};
                    case substrait::Expression::Literal::kI32:
                        return LiteralExpr{static_cast<int64_t>(l.i32())};
                    case substrait::Expression::Literal::kI64:
                        return LiteralExpr{l.i64()};
                    case substrait::Expression::Literal::kFp64:
                        return LiteralExpr{l.fp64()};
                    case substrait::Expression::Literal::kString:
                        return LiteralExpr{l.string()};
                    default:
                        raise(ErrorCode::NotImplemented,
                              "SubstraitDecoder: unsupported literal kind {}",
                              static_cast<int>(l.literal_type_case()));
                }
            }
            case substrait::Expression::kSelection:
                return ColumnRefExpr{fieldIndexOf(e)};
            case substrait::Expression::kScalarFunction:
                return decodeScalarFunction(e.scalar_function(), in);
            default:
                raise(ErrorCode::NotImplemented,
                      "SubstraitDecoder: unsupported expression kind {}",
                      static_cast<int>(e.rex_type_case()));
        }
    }

    ScalarExpr decodeScalarFunction(
            const substrait::Expression::ScalarFunction& fn, const Schema& in) {
        const std::string& name = functionName(fn.function_reference());

        auto arg = [&](int i) -> const substrait::Expression& {
            if (i >= fn.arguments_size() || !fn.arguments(i).has_value()) {
                raise(ErrorCode::InvalidArgument,
                      "SubstraitDecoder: function '{}' missing argument {}",
                      name, i);
            }
            return fn.arguments(i).value();
        };

        static const std::unordered_map<std::string, BinaryOpType> kBinary = {
            {"lt", BinaryOpType::Lt},    {"gt", BinaryOpType::Gt},
            {"lte", BinaryOpType::Lte},  {"gte", BinaryOpType::Gte},
            {"equal", BinaryOpType::Eq}, {"and", BinaryOpType::And},
            {"or", BinaryOpType::Or},    {"add", BinaryOpType::Add},
            {"subtract", BinaryOpType::Sub},
            {"multiply", BinaryOpType::Mul},
        };
        if (auto it = kBinary.find(name); it != kBinary.end()) {
            if (fn.arguments_size() < 2) {
                raise(ErrorCode::InvalidArgument,
                      "SubstraitDecoder: '{}' needs 2+ arguments", name);
            }
            // Fold n-ary and/or left-to-right.
            ScalarExpr acc = decodeExpr(arg(0), in);
            for (int i = 1; i < fn.arguments_size(); ++i) {
                auto b = std::make_unique<BinaryOpExpr>();
                b->op  = it->second;
                b->lhs = std::move(acc);
                b->rhs = decodeExpr(arg(i), in);
                acc = ScalarExpr{std::move(b)};
            }
            return acc;
        }
        if (name == "not" || name == "is_null") {
            auto u = std::make_unique<UnaryOpExpr>();
            u->op = name == "not" ? UnaryOpType::Not : UnaryOpType::IsNull;
            u->input = decodeExpr(arg(0), in);
            return ScalarExpr{std::move(u)};
        }
        if (name == "like") {
            const auto& pat = arg(1);
            if (!pat.has_literal() ||
                pat.literal().literal_type_case() !=
                    substrait::Expression::Literal::kString) {
                raise(ErrorCode::NotImplemented,
                      "SubstraitDecoder: LIKE needs a string-literal pattern");
            }
            auto l = std::make_unique<LikeOpExpr>();
            l->input   = decodeExpr(arg(0), in);
            l->pattern = pat.literal().string();
            return ScalarExpr{std::move(l)};
        }
        raise(ErrorCode::NotImplemented,
              "SubstraitDecoder: unsupported scalar function '{}'", name);
    }

    // Result type of an expression — for Project output schemas.
    DataType inferType(const substrait::Expression& e, const Schema& in) {
        switch (e.rex_type_case()) {
            case substrait::Expression::kLiteral: {
                const auto& l = e.literal();
                switch (l.literal_type_case()) {
                    case substrait::Expression::Literal::kBoolean: return Bool;
                    case substrait::Expression::Literal::kI32:     return Int64;
                    case substrait::Expression::Literal::kI64:     return Int64;
                    case substrait::Expression::Literal::kFp64:    return Float64;
                    case substrait::Expression::Literal::kString:  return DataType{TypeId::STRING};
                    default: break;
                }
                raise(ErrorCode::NotImplemented,
                      "SubstraitDecoder: untyped literal");
            }
            case substrait::Expression::kSelection:
                return in.field(static_cast<size_t>(fieldIndexOf(e))).type;
            case substrait::Expression::kScalarFunction: {
                const std::string& name =
                    functionName(e.scalar_function().function_reference());
                if (name == "add" || name == "subtract" || name == "multiply") {
                    return inferType(e.scalar_function().arguments(0).value(), in);
                }
                return Bool;  // comparisons / logic / is_null / like
            }
            default:
                raise(ErrorCode::NotImplemented,
                      "SubstraitDecoder: cannot infer expression type");
        }
    }

    Decoded decodeRead(const substrait::ReadRel& read) {
        if (!read.has_named_table() || read.named_table().names_size() == 0) {
            raise(ErrorCode::NotImplemented,
                  "SubstraitDecoder: only named_table reads are supported");
        }
        std::string sourceId = read.named_table().names(0);
        for (int i = 1; i < read.named_table().names_size(); ++i) {
            sourceId += "." + read.named_table().names(i);
        }
        Schema schema = schemaFrom(read.base_schema());
        return Decoded{PlanNode{ScanNode{std::move(sourceId), schema}},
                       std::move(schema)};
    }

    Decoded decodeFilter(const substrait::FilterRel& f) {
        Decoded child = decodeRel(f.input());
        auto node = std::make_unique<FilterNode>();
        node->child     = std::move(child.node);
        node->predicate = decodeExpr(f.condition(), child.schema);
        return Decoded{PlanNode{std::move(node)}, std::move(child.schema)};
    }

    Decoded decodeProject(const substrait::ProjectRel& p) {
        Decoded child = decodeRel(p.input());
        const auto nChild = static_cast<int32_t>(child.schema.numFields());

        // Substrait Project output = input fields ++ expressions, then the
        // optional common.emit mapping picks/reorders. Translate each
        // emitted index to either a column ref or a decoded expression.
        std::vector<int32_t> emitted;
        if (p.has_common() && p.common().has_emit()) {
            for (const int32_t idx : p.common().emit().output_mapping()) {
                emitted.push_back(idx);
            }
        } else {
            for (int32_t i = 0; i < nChild + p.expressions_size(); ++i) {
                emitted.push_back(i);
            }
        }

        auto node = std::make_unique<ProjectNode>();
        std::vector<Field> outFields;
        for (const int32_t idx : emitted) {
            if (idx < nChild) {
                node->exprs.push_back(ColumnRefExpr{idx});
                outFields.push_back(child.schema.field(static_cast<size_t>(idx)));
            } else {
                const auto& e = p.expressions(idx - nChild);
                node->exprs.push_back(decodeExpr(e, child.schema));
                outFields.push_back(Field{
                    "expr" + std::to_string(idx - nChild),
                    inferType(e, child.schema)});
            }
            node->output_names.push_back(outFields.back().name);
        }
        node->child = std::move(child.node);
        Schema outSchema(std::move(outFields));
        return Decoded{PlanNode{std::move(node)}, std::move(outSchema)};
    }

    Decoded decodeAggregate(const substrait::AggregateRel& a) {
        Decoded child = decodeRel(a.input());
        if (a.groupings_size() != 1) {
            raise(ErrorCode::NotImplemented,
                  "SubstraitDecoder: exactly one grouping set supported "
                  "(got {})", a.groupings_size());
        }

        auto node = std::make_unique<HashAggregateNode>();
        std::vector<Field> outFields;
        for (const auto& g : a.groupings(0).grouping_expressions()) {
            const int32_t idx = fieldIndexOf(g);
            node->group_keys.push_back(ColumnRefExpr{idx});
            outFields.push_back(child.schema.field(static_cast<size_t>(idx)));
            node->output_names.push_back(outFields.back().name);
        }

        for (const auto& m : a.measures()) {
            const auto&        fn   = m.measure();
            const std::string& name = functionName(fn.function_reference());
            const bool distinct =
                fn.invocation() ==
                substrait::AggregateFunction::AGGREGATION_INVOCATION_DISTINCT;

            AggregateSpec spec;
            DataType      outType = Int64;
            if (name == "count" && fn.arguments_size() == 0) {
                spec.kind = AggregateKind::CountStar;
            } else {
                if (fn.arguments_size() != 1 || !fn.arguments(0).has_value()) {
                    raise(ErrorCode::NotImplemented,
                          "SubstraitDecoder: aggregate '{}' needs one "
                          "field-reference argument", name);
                }
                const int32_t idx = fieldIndexOf(fn.arguments(0).value());
                spec.input     = ColumnRefExpr{idx};
                spec.inputType = child.schema.field(static_cast<size_t>(idx)).type;
                if (name == "count") {
                    spec.kind = distinct ? AggregateKind::CountDistinct
                                         : AggregateKind::Count;
                } else if (name == "sum") {
                    spec.kind = AggregateKind::Sum;
                    outType   = spec.inputType;
                } else if (name == "min") {
                    spec.kind = AggregateKind::Min;
                    outType   = spec.inputType;
                } else if (name == "max") {
                    spec.kind = AggregateKind::Max;
                    outType   = spec.inputType;
                } else if (name == "avg") {
                    spec.kind = AggregateKind::Avg;
                    outType   = Float64;
                } else {
                    raise(ErrorCode::NotImplemented,
                          "SubstraitDecoder: unsupported aggregate '{}'", name);
                }
            }
            node->aggregates.push_back(std::move(spec));
            node->output_names.push_back(
                "agg" + std::to_string(node->aggregates.size() - 1));
            outFields.push_back(Field{node->output_names.back(), outType});
        }

        node->child = std::move(child.node);
        Schema outSchema(std::move(outFields));
        return Decoded{PlanNode{std::move(node)}, std::move(outSchema)};
    }

    Decoded decodeFetch(const substrait::FetchRel& f) {
        Decoded child = decodeRel(f.input());
        if (f.offset() != 0) {
            raise(ErrorCode::NotImplemented,
                  "SubstraitDecoder: FetchRel offset not supported");
        }
        auto node = std::make_unique<LimitNode>();
        node->child = std::move(child.node);
        node->limit = f.count();
        return Decoded{PlanNode{std::move(node)}, std::move(child.schema)};
    }

    Decoded decodeSort(const substrait::SortRel& s) {
        Decoded child = decodeRel(s.input());
        auto node = std::make_unique<SortNode>();
        for (const auto& sf : s.sorts()) {
            node->keys.push_back(decodeExpr(sf.expr(), child.schema));
            switch (sf.direction()) {
                case substrait::SortField::SORT_DIRECTION_ASC_NULLS_FIRST:
                    node->ascending.push_back(true);
                    node->nulls_first.push_back(true);
                    break;
                case substrait::SortField::SORT_DIRECTION_ASC_NULLS_LAST:
                    node->ascending.push_back(true);
                    node->nulls_first.push_back(false);
                    break;
                case substrait::SortField::SORT_DIRECTION_DESC_NULLS_FIRST:
                    node->ascending.push_back(false);
                    node->nulls_first.push_back(true);
                    break;
                case substrait::SortField::SORT_DIRECTION_DESC_NULLS_LAST:
                    node->ascending.push_back(false);
                    node->nulls_first.push_back(false);
                    break;
                default:
                    raise(ErrorCode::NotImplemented,
                          "SubstraitDecoder: unsupported sort direction {}",
                          static_cast<int>(sf.direction()));
            }
        }
        node->child = std::move(child.node);
        return Decoded{PlanNode{std::move(node)}, std::move(child.schema)};
    }

    std::unordered_map<uint32_t, std::string> functions_;
};

} // namespace

PlanNode planFromSubstraitBytes(std::string_view bytes) {
    substrait::Plan plan;
    if (!plan.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
        raise(ErrorCode::InvalidArgument,
              "SubstraitDecoder: failed to parse substrait.Plan bytes");
    }
    if (plan.relations_size() == 0) {
        raise(ErrorCode::InvalidArgument, "SubstraitDecoder: plan has no relations");
    }
    const auto& planRel = plan.relations(0);

    Decoder decoder(plan);
    if (planRel.has_root()) {
        return decoder.decodeRel(planRel.root().input()).node;
    }
    if (planRel.has_rel()) {
        return decoder.decodeRel(planRel.rel()).node;
    }
    raise(ErrorCode::InvalidArgument, "SubstraitDecoder: empty PlanRel");
}

} // namespace quantum::plan
