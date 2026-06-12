#include "plan/PlanBuilder.hpp"

#include "plan/OperatorRegistry.hpp"

#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include <cxxabi.h>

#include "common/Metrics.hpp"
#include "exec/TaskScheduler.hpp"
#include "common/Status.hpp"
#include "expression/BinaryExpr.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/LikeExpr.hpp"
#include "expression/Literal.hpp"
#include "expression/UnaryExpr.hpp"
#include "io/ConcatFileReader.hpp"
#include "io/DeltaTableReader.hpp"
#include "io/IcebergTableReader.hpp"
#include "io/ParquetReader.hpp"
#include "io/PrefetchReader.hpp"
#include "io/PartitionedFileReader.hpp"
#include "operator/AggregateFunction.hpp"
#include "operator/ExchangeOperator.hpp"
#include "operator/FileScanOperator.hpp"
#include "operator/FilterOperator.hpp"
#include "operator/HashAggregateOperator.hpp"
#include "operator/HashJoinOperator.hpp"
#include "operator/LimitOperator.hpp"
#include "operator/MetricsOperator.hpp"
#include "operator/ParallelHashAggregateOperator.hpp"
#include "operator/ProjectOperator.hpp"
#include "operator/ScanOperator.hpp"
#include "operator/SortOperator.hpp"
#include "operator/UnionAllOperator.hpp"

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

        case AggregateKind::Count:
            return std::make_unique<CountColumn>(inputExpr());

        case AggregateKind::CountDistinct:
            if (spec.inputType.id == TypeId::INT64)   return std::make_unique<CountDistinctI64>(inputExpr());
            if (spec.inputType.id == TypeId::STRING)  return std::make_unique<CountDistinctString>(inputExpr());
            if (spec.inputType.id == TypeId::FLOAT64) return std::make_unique<CountDistinctF64>(inputExpr());
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: COUNT(DISTINCT) not supported on {}",
                  spec.inputType.toString());

        case AggregateKind::Sum:
            if (spec.inputType.id == TypeId::INT64)   return std::make_unique<SumI64>(inputExpr());
            if (spec.inputType.id == TypeId::FLOAT64) return std::make_unique<SumF64>(inputExpr());
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: SUM not supported on {}", spec.inputType.toString());

        case AggregateKind::Min:
            if (spec.inputType.id == TypeId::INT64)   return std::make_unique<MinI64>(inputExpr());
            if (spec.inputType.id == TypeId::FLOAT64) return std::make_unique<MinF64>(inputExpr());
            if (spec.inputType.id == TypeId::STRING)  return std::make_unique<MinString>(inputExpr());
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: MIN not supported on {}", spec.inputType.toString());

        case AggregateKind::Max:
            if (spec.inputType.id == TypeId::INT64)   return std::make_unique<MaxI64>(inputExpr());
            if (spec.inputType.id == TypeId::FLOAT64) return std::make_unique<MaxF64>(inputExpr());
            if (spec.inputType.id == TypeId::STRING)  return std::make_unique<MaxString>(inputExpr());
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: MAX not supported on {}", spec.inputType.toString());

        case AggregateKind::Avg:
            if (spec.inputType.id == TypeId::INT64)   return std::make_unique<AvgI64>(inputExpr());
            if (spec.inputType.id == TypeId::FLOAT64) return std::make_unique<AvgF64>(inputExpr());
            raise(ErrorCode::NotImplemented,
                  "PlanBuilder: AVG not supported on {}", spec.inputType.toString());
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
        } else if constexpr (std::is_same_v<T, std::unique_ptr<UnaryOpExpr>>) {
            UnaryOp nativeOp;
            switch (e->op) {
                case UnaryOpType::Not:    nativeOp = UnaryOp::Not;    break;
                case UnaryOpType::IsNull: nativeOp = UnaryOp::IsNull; break;
            }
            return std::make_unique<UnaryExpr>(nativeOp, buildExpr(e->input));
        }
    }, expr);
}

namespace {

// Demangled class name of the built operator, without the namespace —
// e.g. "HashAggregateOperator". Used as the metrics label.
std::string operatorName(const Operator& op) {
    int status = 0;
    char* demangled =
        abi::__cxa_demangle(typeid(op).name(), nullptr, nullptr, &status);
    std::string name = (status == 0 && demangled) ? demangled : typeid(op).name();
    std::free(demangled);
    const auto pos = name.rfind("::");
    if (pos != std::string::npos) name = name.substr(pos + 2);
    return name;
}

} // namespace

// Default memory-fraction spill trigger for aggregates and sorts
// (QUANTUM_SPILL_FRACTION overrides; "0" disables). Bounds worst-case
// memory: when MemoryManager usage crosses the fraction, partial state
// flushes to disk and merges back before emit.
double defaultSpillFraction() {
    static const double v = [] {
        if (const char* env = std::getenv("QUANTUM_SPILL_FRACTION")) {
            const double f = std::strtod(env, nullptr);
            return (f >= 0.0 && f < 1.0) ? f : 0.0;
        }
        return 0.8;
    }();
    return v;
}

// Decode-ahead for file scans (QUANTUM_PREFETCH=0 disables): the reader
// runs on a scheduler task so decode overlaps downstream compute.
FileFormatReaderPtr wrapPrefetch(FileFormatReaderPtr reader) {
    static const bool enabled = [] {
        const char* env = std::getenv("QUANTUM_PREFETCH");
        return env == nullptr || std::string_view(env) != "0";
    }();
    if (!enabled) return reader;
    return std::make_unique<PrefetchReader>(std::move(reader));
}

OperatorPtr PlanBuilder::build(const PlanNode& node, SourceCatalog& catalog) {
    OperatorPtr op = buildImpl(node, catalog);
    if (metrics_ != nullptr) {
        OperatorMetrics* m = metrics_->add(operatorName(*op));
        op = std::make_unique<MetricsOperator>(std::move(op), m);
    }
    return op;
}

OperatorPtr PlanBuilder::buildImpl(const PlanNode& node, SourceCatalog& catalog) {
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
                const bool partitioned = n.partition_columns.numFields() > 0;
                if (partitioned && n.partition_values.size() != n.paths.size()) {
                    raise(ErrorCode::InvalidArgument,
                          "PlanBuilder: FileScanNode has {} paths but {} partition-value rows",
                          n.paths.size(), n.partition_values.size());
                }
                auto wrapPartitioned = [&](FileFormatReaderPtr inner, size_t pathIdx)
                    -> FileFormatReaderPtr {
                    if (!partitioned) return inner;
                    std::vector<PartitionValue> values;
                    values.reserve(n.partition_columns.numFields());
                    for (const auto& lit : n.partition_values[pathIdx]) {
                        std::visit([&](const auto& v) {
                            using T = std::decay_t<decltype(v)>;
                            if constexpr (std::is_same_v<T, bool>) {
                                values.emplace_back(static_cast<bool>(v));
                            } else if constexpr (std::is_same_v<T, int64_t>) {
                                values.emplace_back(static_cast<int64_t>(v));
                            } else if constexpr (std::is_same_v<T, double>) {
                                values.emplace_back(static_cast<double>(v));
                            } else if constexpr (std::is_same_v<T, std::string>) {
                                values.emplace_back(v);
                            }
                        }, lit.value);
                    }
                    return std::make_unique<PartitionedFileReader>(
                        std::move(inner),
                        std::vector<Field>(n.partition_columns.fields()),
                        std::move(values),
                        mm_);
                };

                if (n.paths.size() == 1) {
                    auto reader = std::make_unique<ParquetReader>(n.paths.front(), mm_);
                    auto wrapped = wrapPartitioned(std::move(reader), 0);
                    return std::make_unique<FileScanOperator>(mm_, wrapPrefetch(std::move(wrapped)));
                }
                // Per-file construction decodes the whole file (ParquetReader
                // is eager), so multi-file scans fan the decode out across
                // the scheduler — each file is one independent task touching
                // only the thread-safe MemoryManager.
                std::vector<FileFormatReaderPtr> readers(n.paths.size());
                TaskScheduler::global().parallelFor(
                    n.paths.size(), [&](size_t i) {
                        auto pr = std::make_unique<ParquetReader>(n.paths[i], mm_);
                        readers[i] = wrapPartitioned(std::move(pr), i);
                    });
                auto concat = std::make_unique<ConcatFileReader>(std::move(readers));
                return std::make_unique<FileScanOperator>(mm_, wrapPrefetch(std::move(concat)));
            }
            if (n.format == "delta") {
                if (n.paths.size() != 1) {
                    raise(ErrorCode::InvalidArgument,
                          "PlanBuilder: delta FileScanNode expects 1 table-root path "
                          "(got {})", n.paths.size());
                }
                auto reader = std::make_unique<DeltaTableReader>(n.paths.front(), mm_);
                return std::make_unique<FileScanOperator>(mm_, wrapPrefetch(std::move(reader)));
            }
            if (n.format == "iceberg") {
                if (n.paths.size() != 1) {
                    raise(ErrorCode::InvalidArgument,
                          "PlanBuilder: iceberg FileScanNode expects 1 table-root path "
                          "(got {})", n.paths.size());
                }
                auto reader = std::make_unique<IcebergTableReader>(n.paths.front(), mm_);
                return std::make_unique<FileScanOperator>(mm_, wrapPrefetch(std::move(reader)));
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
            if (n->parallelism > 1) {
                // Parallel partitioned aggregation: hash-exchange the child
                // by the group keys, then one HashAggregate per partition
                // on the TaskScheduler. Each partition needs its own key
                // expressions + aggregate instances (single-threaded
                // operators; only MemoryManager is shared).
                const auto P = static_cast<size_t>(n->parallelism);
                std::vector<ExpressionPtr> exchangeKeys;
                exchangeKeys.reserve(n->group_keys.size());
                for (const auto& k : n->group_keys) exchangeKeys.push_back(buildExpr(k));
                auto exchange = std::make_unique<ExchangeOperator>(
                    mm_, build(n->child, catalog), std::move(exchangeKeys), P);

                std::vector<ParallelHashAggregateOperator::PartitionAggSpec> specs(P);
                for (size_t p = 0; p < P; ++p) {
                    for (const auto& k : n->group_keys) specs[p].keys.push_back(buildExpr(k));
                    for (const auto& a : n->aggregates) specs[p].aggs.push_back(buildAggregate(a, *this));
                    specs[p].outputNames = n->output_names;
                }
                return std::make_unique<ParallelHashAggregateOperator>(
                    mm_, std::move(exchange), std::move(specs));
            }
            std::vector<ExpressionPtr> keys;
            keys.reserve(n->group_keys.size());
            for (const auto& k : n->group_keys) keys.push_back(buildExpr(k));
            std::vector<AggregateFunctionPtr> aggs;
            aggs.reserve(n->aggregates.size());
            for (const auto& a : n->aggregates) aggs.push_back(buildAggregate(a, *this));
            return std::make_unique<HashAggregateOperator>(
                mm_, build(n->child, catalog),
                std::move(keys), std::move(aggs),
                n->output_names,
                /*spillTriggerGroups=*/0, defaultSpillFraction());
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ExchangeNode>>) {
            std::vector<ExpressionPtr> keys;
            keys.reserve(n->keys.size());
            for (const auto& k : n->keys) keys.push_back(buildExpr(k));
            return std::make_unique<ExchangeOperator>(
                mm_, build(n->child, catalog), std::move(keys),
                static_cast<size_t>(n->num_partitions));
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
        } else if constexpr (std::is_same_v<T, std::unique_ptr<LimitNode>>) {
            return std::make_unique<LimitOperator>(
                mm_, build(n->child, catalog), n->limit);
        } else if constexpr (std::is_same_v<T, std::unique_ptr<SortNode>>) {
            std::vector<ExpressionPtr> keys;
            keys.reserve(n->keys.size());
            for (const auto& k : n->keys) keys.push_back(buildExpr(k));
            return std::make_unique<SortOperator>(
                mm_, build(n->child, catalog),
                std::move(keys), n->ascending, n->nulls_first,
                /*spillTriggerRows=*/0, defaultSpillFraction());
        } else if constexpr (std::is_same_v<T, std::unique_ptr<ExtensionNode>>) {
            ExtensionContext ctx;
            ctx.mm      = mm_;
            ctx.payload = n->payload;
            for (const auto& c : n->children) {
                ctx.children.push_back(build(c, catalog));
            }
            return OperatorRegistry::instance().build(n->extension_type,
                                                      std::move(ctx));
        } else if constexpr (std::is_same_v<T, std::unique_ptr<UnionAllNode>>) {
            std::vector<OperatorPtr> ops;
            ops.reserve(n->children.size());
            for (const auto& c : n->children) ops.push_back(build(c, catalog));
            return std::make_unique<UnionAllOperator>(mm_, std::move(ops));
        }
    }, node);
}

} // namespace quantum::plan
