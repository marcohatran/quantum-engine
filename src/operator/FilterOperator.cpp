#include "operator/FilterOperator.hpp"

#include "common/Status.hpp"

namespace quantum {

void FilterOperator::open()  { child_->open(); }
void FilterOperator::close() { child_->close(); }

std::optional<ColumnBatch> FilterOperator::getNext() {
    auto in = child_->getNext();
    if (!in) return std::nullopt;

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
