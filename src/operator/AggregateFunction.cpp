#include "operator/AggregateFunction.hpp"

#include <cstring>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

// ---------- SumI64 ----------

SumI64::SumI64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "SumI64: input expression required");
}

void SumI64::init(uint8_t* state) const {
    int64_t z = 0;
    std::memcpy(state, &z, sizeof(z));
}

void SumI64::update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "SumI64: input column is required");
    }
    if (input->type().id != TypeId::INT64) {
        raise(ErrorCode::TypeMismatch,
              "SumI64: input type {} != INT64", input->type().toString());
    }
    const int64_t*         vals     = input->typed<int64_t>();
    const ValidityBitmask& validity = input->validity();

    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        int64_t cur;
        std::memcpy(&cur, groupSlots[k], sizeof(cur));
        cur += vals[i];
        std::memcpy(groupSlots[k], &cur, sizeof(cur));
    }
}

ColumnVector SumI64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Int64, states.size(), mm);
    out.validity().setAllValid();
    int64_t* o = out.typed<int64_t>();
    for (size_t i = 0; i < states.size(); ++i) {
        std::memcpy(&o[i], states[i], sizeof(int64_t));
    }
    return out;
}

// ---------- CountStar ----------

void CountStar::init(uint8_t* state) const {
    int64_t z = 0;
    std::memcpy(state, &z, sizeof(z));
}

void CountStar::update(uint8_t* const* groupSlots,
                       const ColumnVector* /*input*/,
                       const PositionList& pos) const {
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        int64_t cur;
        std::memcpy(&cur, groupSlots[k], sizeof(cur));
        ++cur;
        std::memcpy(groupSlots[k], &cur, sizeof(cur));
    }
}

ColumnVector CountStar::finalize(const std::vector<const uint8_t*>& states,
                                 MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Int64, states.size(), mm);
    out.validity().setAllValid();
    int64_t* o = out.typed<int64_t>();
    for (size_t i = 0; i < states.size(); ++i) {
        std::memcpy(&o[i], states[i], sizeof(int64_t));
    }
    return out;
}

} // namespace quantum
