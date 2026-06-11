#include "operator/FilterOperator.hpp"

#include "common/Status.hpp"
#include "expression/ColumnRef.hpp"
#include "expression/Literal.hpp"

namespace quantum {

FilterOperator::FilterOperator(MemoryManager* mm, OperatorPtr child,
                               ExpressionPtr predicate)
    : Operator(mm),
      child_(std::move(child)),
      predicate_(std::move(predicate)) {
    // Detect `ColumnRef <cmp> Literal(int64|int32)` over an INT64 column.
    const auto* bin = dynamic_cast<const BinaryExpr*>(predicate_.get());
    if (bin == nullptr || !isComparisonOp(bin->op())) return;
    const auto* ref = dynamic_cast<const ColumnRef*>(&bin->lhs());
    const auto* lit = dynamic_cast<const Literal*>(&bin->rhs());
    if (ref == nullptr || lit == nullptr) return;
    if (ref->index() >= child_->outputSchema().numFields()) return;
    if (child_->outputSchema().field(ref->index()).type.id != TypeId::INT64) return;

    int64_t value = 0;
    if (const auto* v64 = std::get_if<int64_t>(&lit->value())) value = *v64;
    else if (const auto* v32 = std::get_if<int32_t>(&lit->value())) value = *v32;
    else return;

    fused_ = FusedI64Compare{ref->index(), bin->op(), value};
}

void FilterOperator::open()  { child_->open(); }
void FilterOperator::close() { child_->close(); }

std::optional<ColumnBatch> FilterOperator::getNext() {
    auto in = child_->getNext();
    if (!in) return std::nullopt;

    if (fused_.has_value()) {
        const auto&    col   = in->column(fused_->col);
        const int64_t* v     = col.typed<int64_t>();
        const auto&    valid = col.validity();
        const int64_t  lit   = fused_->lit;
        const PositionList& inPos = in->positions();

        std::vector<int32_t> next;
        next.reserve(inPos.size());
        auto scan = [&](auto cmp) {
            if (inPos.isIdentity()) {
                const size_t n = inPos.size();
                for (size_t i = 0; i < n; ++i) {
                    if (valid.isValid(i) && cmp(v[i], lit)) {
                        next.push_back(static_cast<int32_t>(i));
                    }
                }
            } else {
                for (size_t k = 0; k < inPos.size(); ++k) {
                    const auto i = static_cast<size_t>(inPos.at(k));
                    if (valid.isValid(i) && cmp(v[i], lit)) {
                        next.push_back(static_cast<int32_t>(i));
                    }
                }
            }
        };
        switch (fused_->op) {
            case BinaryOp::Gt:  scan([](int64_t a, int64_t b) { return a >  b; }); break;
            case BinaryOp::Lt:  scan([](int64_t a, int64_t b) { return a <  b; }); break;
            case BinaryOp::Gte: scan([](int64_t a, int64_t b) { return a >= b; }); break;
            case BinaryOp::Lte: scan([](int64_t a, int64_t b) { return a <= b; }); break;
            case BinaryOp::Eq:  scan([](int64_t a, int64_t b) { return a == b; }); break;
            default:
                raise(ErrorCode::Internal, "FilterOperator: bad fused op");
        }
        in->setPositions(PositionList::fromIndices(std::move(next)));
        return in;
    }

    ColumnVector mask = predicate_->eval(*in, mm_);
    if (mask.type().id != TypeId::BOOL) {
        raise(ErrorCode::TypeMismatch,
              "FilterOperator: predicate result is {}, expected BOOL",
              mask.type().toString());
    }

    const uint8_t*      bits     = mask.typed<uint8_t>();
    const ValidityBitmask& validity = mask.validity();
    const PositionList& inPos    = in->positions();

    std::vector<int32_t> next;
    next.reserve(inPos.size());

    if (inPos.isIdentity()) {
        const size_t n = inPos.size();
        for (size_t i = 0; i < n; ++i) {
            if (validity.isValid(i) && bits[i]) {
                next.push_back(static_cast<int32_t>(i));
            }
        }
    } else {
        for (size_t k = 0; k < inPos.size(); ++k) {
            const auto i = static_cast<size_t>(inPos.at(k));
            if (validity.isValid(i) && bits[i]) {
                next.push_back(static_cast<int32_t>(i));
            }
        }
    }

    in->setPositions(PositionList::fromIndices(std::move(next)));
    return in;
}

} // namespace quantum
