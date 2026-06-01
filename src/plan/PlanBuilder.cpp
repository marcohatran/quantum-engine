#include "plan/PlanBuilder.hpp"

#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "common/Status.hpp"
#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/LikeExpr.hpp"
#include "expression/Literal.hpp"
#include "io/ConcatFileReader.hpp"
#include "io/DeltaTableReader.hpp"
#include "io/ParquetReader.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/FileScanOperator.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/HashJoinOperator.hpp"
#include "operator/ProjectOperator.hpp"
#include "operator/ScanOperator.hpp"

namespace quantum::plan {

namespace {

BinaryOp toBinaryOp(BinaryOpType op) {
    switch (op) {
        case BinaryOpType::Add: return BinaryOp::Add;
        case BinaryOpType::Sub: return BinaryOp::Sub;
        case BinaryOpType::Mul: return BinaryOp::Mul;
        case BinaryOpType::Eq:  return BinaryOp::Eq;
        case BinaryOpType::Lt:  return BinaryOp::Lt;
        case BinaryOpType::Gt:  return BinaryOp::Gt;
        case BinaryOpType::Lte: return BinaryOp::Lte;
        case BinaryOpType::Gte: return BinaryOp::Gte;
        case BinaryOpType::And: return BinaryOp::And;
        case BinaryOpType::Or:  return BinaryOp::Or;
    }
    raise(ErrorCode::Internal, "PlanBuilder: unknown BinaryOpType");
}

JoinType toJoinType(JoinTypeIR t) {
    switch (t) {
        case JoinTypeIR::Inner:      return JoinType::Inner;
        case JoinTypeIR::LeftOuter:  return JoinType::LeftOuter;
        case JoinTypeIR::RightOuter: return JoinType::RightOuter;
        case JoinTypeIR::FullOuter:  return JoinType::FullOuter;
    }
    raise(ErrorCode::Internal, "PlanBuilder: unknown JoinTypeIR");
}

AggregateFunctionPtr buildAggregate(const AggregateSpec& spec, PlanBuilder& pb) {
    auto inputExpr = [&]() {
        if (!spec.input.has_value()) {
            raise(ErrorCode::InvalidArgument,
                  "PlanBuilder: aggregate requires an input expression");
        }
        return pb.buildExpr(*spec.input);
    };

    switch (spec.kind) {
        case AggregateKind::CountStar:
            return std::make_unique<CountStar>();

        case AggregateKind::Sum:
            if (spec.inputType.id != TypeId::INT64) {
                raise(ErrorCode::NotImplemented,
                      "PlanBuilder: SUM only supported on INT64 (got {})",
                      spec.inputType.toString());
            }
            return std::make_unique<SumI64>(inputExpr());

        case AggregateKind::Min:
            if (spec.inputType.id == TypeId::INT64) {
                return std::make_unique<MinI64>(inputExpr());
            }
            if (spec.inputType.id == TypeId::STRING) {
                return std::make_unique<MinString>(inputExpr());
            }
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: MIN not supported on {}", spec.inputType.toString());

        case AggregateKind::Max:
            if (spec.inputType.id == TypeId::INT64) {
                return std::make_unique<MaxI64>(inputExpr());
            }
            if (spec.inputType.id == TypeId::STRING) {
                return std::make_unique<MaxString>(inputExpr());
            }
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: MAX not supported on {}", spec.inputType.toString());

        case AggregateKind::Avg:
            if (spec.inputType.id != TypeId::INT64) {
                raise(ErrorCode::NotImplemented,
                      "PlanBuilder: AVG only supported on INT64 (got {})",
                      spec.inputType.toString());
            }
            return std::make_unique<AvgI64>(inputExpr());
    }
    raise(ErrorCode::Internal, "PlanBuilder: unknown AggregateKind");
}

} // namespace

ExpressionPtr PlanBuilder::buildExpr(const ScalarExpr& expr) {
    return std::visit([&](const auto& e) -> ExpressionPtr {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, LiteralExpr>) {
            return std::visit([](const auto& v) -> ExpressionPtr {
                return std::make_unique<Literal>(v);
            }, e.value);
        } else if constexpr (std::is_same_v<T, ColumnRefExpr>) {
            return std::make_unique<ColumnRef>(static_cast<size_t>(e.index));
        } else if constexpr (std::is_same_v<T, std::unique_ptr<BinaryOpExpr>>) {
            return std::make_unique<BinaryExpr>(
                toBinaryOp(e->op),
                buildExpr(e->lhs),
                buildExpr(e->rhs));
        } else if constexpr (std::is_same_v<T, std::unique_ptr<LikeOpExpr>>) {
            return std::make_unique<LikeExpr>(
                buildExpr(e->input), e->pattern);
        }
    }, expr);
}

OperatorPtr PlanBuilder::build(const PlanNode& node, SourceCatalog& catalog) {
    return std::visit([&](const auto& n) -> OperatorPtr {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, ScanNode>) {
            auto it = catalog.find(n.source_id);
            if (it == catalog.end()) {
                raise(ErrorCode::InvalidArgument,
                      "PlanBuilder: no source data registered for '{}'", n.source_id);
            }
            return std::make_unique<ScanOperator>(
                mm_, n.schema, std::move(it->second));
        } else if constexpr (std::is_same_v<T, FileScanNode>) {
            if (n.paths.empty()) {
                raise(ErrorCode::InvalidArgument,
                      "PlanBuilder: FileScanNode has no paths");
            }
            if (n.format == "parquet") {
                // Single-path: skip the ConcatFileReader wrapper to keep the
                // common case allocation-free.
                if (n.paths.size() == 1) {
                    auto reader = std::make_unique<ParquetReader>(n.paths.front(), mm_);
                    return std::make_unique<FileScanOperator>(mm_, std::move(reader));
                }
                std::vector<FileFormatReaderPtr> readers;
                readers.reserve(n.paths.size());
                for (const auto& p : n.paths) {
                    readers.push_back(std::make_unique<ParquetReader>(p, mm_));
                }
                auto concat = std::make_unique<ConcatFileReader>(std::move(readers));
                return std::make_unique<FileScanOperator>(mm_, std::move(concat));
            }
            if (n.format == "delta") {
                if (n.paths.size() != 1) {
                    raise(ErrorCode::InvalidArgument,
                          "PlanBuilder: delta FileScanNode expects 1 table-root path "
                          "(got {})", n.paths.size());
                }
                auto reader = std::make_unique<DeltaTableReader>(n.paths.front(), mm_);
                return std::make_unique<FileScanOperator>(mm_, std::move(reader));
            }
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: unknown FileScanNode format '{}'", n.format);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<FilterNode>>) {
            return std::make_unique<FilterOperator>(
                mm_, build(n->child, catalog), buildExpr(n->predicate));
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ProjectNode>>) {
            std::vector<ExpressionPtr> exprs;
            exprs.reserve(n->exprs.size());
            for (const auto& e : n->exprs) exprs.push_back(buildExpr(e));
            return std::make_unique<ProjectOperator>(
                mm_, build(n->child, catalog),
                std::move(exprs),
                n->output_names);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashAggregateNode>>) {
            std::vector<ExpressionPtr> keys;
            keys.reserve(n->group_keys.size());
            for (const auto& k : n->group_keys) keys.push_back(buildExpr(k));
            std::vector<AggregateFunctionPtr> aggs;
            aggs.reserve(n->aggregates.size());
            for (const auto& a : n->aggregates) aggs.push_back(buildAggregate(a, *this));
            return std::make_unique<HashAggregateOperator>(
                mm_, build(n->child, catalog),
                std::move(keys), std::move(aggs),
                n->output_names);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<HashJoinNode>>) {
            std::vector<ExpressionPtr> bk;
            bk.reserve(n->build_keys.size());
            for (const auto& k : n->build_keys) bk.push_back(buildExpr(k));
            std::vector<ExpressionPtr> pk;
            pk.reserve(n->probe_keys.size());
            for (const auto& k : n->probe_keys) pk.push_back(buildExpr(k));
            return std::make_unique<HashJoinOperator>(
                mm_, build(n->build, catalog), build(n->probe, catalog),
                std::move(bk), std::move(pk),
                n->output_names, toJoinType(n->join_type));
        }
    }, node);
}

} // namespace quantum::plan
