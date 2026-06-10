#include "operator/AggregateFunction.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

namespace {

// POD append/read helpers for serializeState / deserializeState.
template <typename T>
void appendPod(std::vector<uint8_t>& out, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
T readPod(const uint8_t*& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

} // namespace

void AggregateFunction::serializeState(const uint8_t* state,
                                       std::vector<uint8_t>& out) const {
    out.insert(out.end(), state, state + stateSize());
}

size_t AggregateFunction::deserializeState(uint8_t* state,
                                           const uint8_t* data) const {
    std::memcpy(state, data, stateSize());
    return stateSize();
}

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

void SumI64::merge(uint8_t* dst, const uint8_t* src) const {
    int64_t a, b;
    std::memcpy(&a, dst, sizeof(a));
    std::memcpy(&b, src, sizeof(b));
    a += b;
    std::memcpy(dst, &a, sizeof(a));
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

void CountStar::merge(uint8_t* dst, const uint8_t* src) const {
    int64_t a, b;
    std::memcpy(&a, dst, sizeof(a));
    std::memcpy(&b, src, sizeof(b));
    a += b;
    std::memcpy(dst, &a, sizeof(a));
}

// ---------- CountColumn ----------

CountColumn::CountColumn(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "CountColumn: input expression required");
}

void CountColumn::init(uint8_t* state) const {
    int64_t z = 0;
    std::memcpy(state, &z, sizeof(z));
}

void CountColumn::update(uint8_t* const* groupSlots,
                         const ColumnVector* input,
                         const PositionList& pos) const {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "CountColumn: input column is required");
    }
    const ValidityBitmask& validity = input->validity();
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        int64_t cur;
        std::memcpy(&cur, groupSlots[k], sizeof(cur));
        ++cur;
        std::memcpy(groupSlots[k], &cur, sizeof(cur));
    }
}

ColumnVector CountColumn::finalize(const std::vector<const uint8_t*>& states,
                                   MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Int64, states.size(), mm);
    out.validity().setAllValid();
    int64_t* o = out.typed<int64_t>();
    for (size_t i = 0; i < states.size(); ++i) {
        std::memcpy(&o[i], states[i], sizeof(int64_t));
    }
    return out;
}

void CountColumn::merge(uint8_t* dst, const uint8_t* src) const {
    int64_t a, b;
    std::memcpy(&a, dst, sizeof(a));
    std::memcpy(&b, src, sizeof(b));
    a += b;
    std::memcpy(dst, &a, sizeof(a));
}

// ---------- helpers for MIN/MAX state ----------

namespace {

// State layout for MIN/MAX: bytes [0..7] = int64 value, byte [8] = hasValue.
void initMinMaxState(uint8_t* state) {
    std::memset(state, 0, 16);  // value=0, hasValue=0
}

inline bool minMaxHasValue(const uint8_t* state) noexcept { return state[8] != 0; }
inline int64_t minMaxValue(const uint8_t* state) noexcept {
    int64_t v;
    std::memcpy(&v, state, sizeof(v));
    return v;
}
inline void minMaxSet(uint8_t* state, int64_t v) noexcept {
    std::memcpy(state, &v, sizeof(v));
    state[8] = 1;
}

template <typename Cmp>
void updateMinMax(uint8_t* const*       groupSlots,
                  const ColumnVector*   input,
                  const PositionList&   pos,
                  const char*           opName) {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "{}: input column is required", opName);
    }
    if (input->type().id != TypeId::INT64) {
        raise(ErrorCode::TypeMismatch, "{}: input type {} != INT64", opName,
              input->type().toString());
    }
    const int64_t*         vals     = input->typed<int64_t>();
    const ValidityBitmask& validity = input->validity();
    Cmp                    cmp;

    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        uint8_t* st = groupSlots[k];
        if (!minMaxHasValue(st) || cmp(vals[i], minMaxValue(st))) {
            minMaxSet(st, vals[i]);
        }
    }
}

ColumnVector finalizeMinMax(const std::vector<const uint8_t*>& states,
                            MemoryManager* mm) {
    ColumnVector out = ColumnVector::make(Int64, states.size(), mm);
    out.validity().setAllValid();
    int64_t* o = out.typed<int64_t>();
    for (size_t i = 0; i < states.size(); ++i) {
        if (minMaxHasValue(states[i])) {
            o[i] = minMaxValue(states[i]);
        } else {
            o[i] = 0;
            out.validity().setValid(i, false);
        }
    }
    return out;
}

template <typename Cmp>
void mergeMinMax(uint8_t* dst, const uint8_t* src) {
    if (!minMaxHasValue(src)) return;
    if (!minMaxHasValue(dst)) {
        std::memcpy(dst, src, 16);
        return;
    }
    Cmp cmp;
    if (cmp(minMaxValue(src), minMaxValue(dst))) {
        minMaxSet(dst, minMaxValue(src));
    }
}

} // namespace

// ---------- MinI64 ----------

MinI64::MinI64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "MinI64: input expression required");
}

void MinI64::init(uint8_t* state) const { initMinMaxState(state); }

void MinI64::update(uint8_t* const* groupSlots, const ColumnVector* input,
                    const PositionList& pos) const {
    updateMinMax<std::less<int64_t>>(groupSlots, input, pos, "MinI64");
}

ColumnVector MinI64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    return finalizeMinMax(states, mm);
}

void MinI64::merge(uint8_t* dst, const uint8_t* src) const {
    mergeMinMax<std::less<int64_t>>(dst, src);
}

// ---------- MaxI64 ----------

MaxI64::MaxI64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "MaxI64: input expression required");
}

void MaxI64::init(uint8_t* state) const { initMinMaxState(state); }

void MaxI64::update(uint8_t* const* groupSlots, const ColumnVector* input,
                    const PositionList& pos) const {
    updateMinMax<std::greater<int64_t>>(groupSlots, input, pos, "MaxI64");
}

ColumnVector MaxI64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    return finalizeMinMax(states, mm);
}

void MaxI64::merge(uint8_t* dst, const uint8_t* src) const {
    mergeMinMax<std::greater<int64_t>>(dst, src);
}

// ---------- MinString / MaxString ----------

namespace {

constexpr int32_t kStringIdxNone = -1;

template <bool IsMax>
void updateStringAgg(uint8_t* const*       groupSlots,
                     const ColumnVector*   input,
                     const PositionList&   pos,
                     std::vector<std::string>& values,
                     const char*           opName) {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "{}: input column is required", opName);
    }
    if (input->type().id != TypeId::STRING) {
        raise(ErrorCode::TypeMismatch, "{}: input type {} != STRING", opName,
              input->type().toString());
    }
    const ValidityBitmask& validity = input->validity();
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        const std::string_view s = input->stringAt(i);

        int32_t idx;
        std::memcpy(&idx, groupSlots[k], sizeof(idx));
        if (idx == kStringIdxNone) {
            idx = static_cast<int32_t>(values.size());
            values.emplace_back(s);
            std::memcpy(groupSlots[k], &idx, sizeof(idx));
            continue;
        }
        const std::string& cur = values[static_cast<size_t>(idx)];
        const bool replace = IsMax ? (s > cur) : (s < cur);
        if (replace) values[static_cast<size_t>(idx)].assign(s);
    }
}

ColumnVector finalizeStringAgg(const std::vector<const uint8_t*>& states,
                               const std::vector<std::string>&    values,
                               MemoryManager*                     mm) {
    StringBuilder sb(mm);
    for (const auto* s : states) {
        int32_t idx;
        std::memcpy(&idx, s, sizeof(idx));
        if (idx == kStringIdxNone) sb.appendNull();
        else                       sb.append(values[static_cast<size_t>(idx)]);
    }
    return sb.build();
}

// merge for MIN/MAX(string): both handles index the same `values` vector.
// src is a scratch state about to be discarded, so adopting its index is
// safe — no other slab will ever reference it.
template <bool IsMax>
void mergeStringAgg(uint8_t* dst, const uint8_t* src,
                    const std::vector<std::string>& values) {
    int32_t s;
    std::memcpy(&s, src, sizeof(s));
    if (s == kStringIdxNone) return;
    int32_t d;
    std::memcpy(&d, dst, sizeof(d));
    const bool adopt =
        d == kStringIdxNone ||
        (IsMax ? values[static_cast<size_t>(s)] > values[static_cast<size_t>(d)]
               : values[static_cast<size_t>(s)] < values[static_cast<size_t>(d)]);
    if (adopt) std::memcpy(dst, &s, sizeof(s));
}

// Serialized MIN/MAX(string) state: [uint8 hasValue][int32 len][len bytes].
void serializeStringAgg(const uint8_t* state,
                        const std::vector<std::string>& values,
                        std::vector<uint8_t>& out) {
    int32_t idx;
    std::memcpy(&idx, state, sizeof(idx));
    const uint8_t has = idx != kStringIdxNone ? 1 : 0;
    appendPod(out, has);
    if (has) {
        const std::string& v = values[static_cast<size_t>(idx)];
        appendPod(out, static_cast<int32_t>(v.size()));
        out.insert(out.end(), v.begin(), v.end());
    }
}

size_t deserializeStringAgg(uint8_t* state,
                            const uint8_t* data,
                            std::vector<std::string>& values) {
    const uint8_t* p = data;
    const auto has = readPod<uint8_t>(p);
    if (has) {
        const auto len = readPod<int32_t>(p);
        const int32_t idx = static_cast<int32_t>(values.size());
        values.emplace_back(reinterpret_cast<const char*>(p),
                            static_cast<size_t>(len));
        p += len;
        std::memcpy(state, &idx, sizeof(idx));
    }
    return static_cast<size_t>(p - data);
}

} // namespace

MinString::MinString(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "MinString: input expression required");
}

void MinString::init(uint8_t* state) const {
    int32_t v = kStringIdxNone;
    std::memcpy(state, &v, sizeof(v));
}

void MinString::update(uint8_t* const* groupSlots,
                       const ColumnVector* input,
                       const PositionList& pos) const {
    updateStringAgg</*IsMax=*/false>(groupSlots, input, pos, values_, "MinString");
}

ColumnVector MinString::finalize(const std::vector<const uint8_t*>& states,
                                 MemoryManager* mm) const {
    return finalizeStringAgg(states, values_, mm);
}

void MinString::merge(uint8_t* dst, const uint8_t* src) const {
    mergeStringAgg</*IsMax=*/false>(dst, src, values_);
}

void MinString::serializeState(const uint8_t* state,
                               std::vector<uint8_t>& out) const {
    serializeStringAgg(state, values_, out);
}

size_t MinString::deserializeState(uint8_t* state, const uint8_t* data) const {
    return deserializeStringAgg(state, data, values_);
}

MaxString::MaxString(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "MaxString: input expression required");
}

void MaxString::init(uint8_t* state) const {
    int32_t v = kStringIdxNone;
    std::memcpy(state, &v, sizeof(v));
}

void MaxString::update(uint8_t* const* groupSlots,
                       const ColumnVector* input,
                       const PositionList& pos) const {
    updateStringAgg</*IsMax=*/true>(groupSlots, input, pos, values_, "MaxString");
}

ColumnVector MaxString::finalize(const std::vector<const uint8_t*>& states,
                                 MemoryManager* mm) const {
    return finalizeStringAgg(states, values_, mm);
}

void MaxString::merge(uint8_t* dst, const uint8_t* src) const {
    mergeStringAgg</*IsMax=*/true>(dst, src, values_);
}

void MaxString::serializeState(const uint8_t* state,
                               std::vector<uint8_t>& out) const {
    serializeStringAgg(state, values_, out);
}

size_t MaxString::deserializeState(uint8_t* state, const uint8_t* data) const {
    return deserializeStringAgg(state, data, values_);
}

// ---------- AvgI64 ----------

AvgI64::AvgI64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "AvgI64: input expression required");
}

void AvgI64::init(uint8_t* state) const {
    std::memset(state, 0, 16);  // sum=0.0, count=0
}

void AvgI64::update(uint8_t* const* groupSlots, const ColumnVector* input,
                    const PositionList& pos) const {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "AvgI64: input column is required");
    }
    if (input->type().id != TypeId::INT64) {
        raise(ErrorCode::TypeMismatch, "AvgI64: input type {} != INT64",
              input->type().toString());
    }
    const int64_t*         vals     = input->typed<int64_t>();
    const ValidityBitmask& validity = input->validity();

    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        uint8_t* st = groupSlots[k];
        double   sum;
        int64_t  count;
        std::memcpy(&sum,   st,     sizeof(sum));
        std::memcpy(&count, st + 8, sizeof(count));
        sum   += static_cast<double>(vals[i]);
        count += 1;
        std::memcpy(st,     &sum,   sizeof(sum));
        std::memcpy(st + 8, &count, sizeof(count));
    }
}

ColumnVector AvgI64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Float64, states.size(), mm);
    out.validity().setAllValid();
    double* o = out.typed<double>();
    for (size_t i = 0; i < states.size(); ++i) {
        double  sum;
        int64_t count;
        std::memcpy(&sum,   states[i],     sizeof(sum));
        std::memcpy(&count, states[i] + 8, sizeof(count));
        if (count == 0) {
            o[i] = 0.0;
            out.validity().setValid(i, false);
        } else {
            o[i] = sum / static_cast<double>(count);
        }
    }
    return out;
}

void AvgI64::merge(uint8_t* dst, const uint8_t* src) const {
    double  sumA, sumB;
    int64_t cntA, cntB;
    std::memcpy(&sumA, dst,     sizeof(sumA));
    std::memcpy(&cntA, dst + 8, sizeof(cntA));
    std::memcpy(&sumB, src,     sizeof(sumB));
    std::memcpy(&cntB, src + 8, sizeof(cntB));
    sumA += sumB;
    cntA += cntB;
    std::memcpy(dst,     &sumA, sizeof(sumA));
    std::memcpy(dst + 8, &cntA, sizeof(cntA));
}

// ---------- CountDistinctI64 ----------

CountDistinctI64::CountDistinctI64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "CountDistinctI64: input expression required");
}

void CountDistinctI64::init(uint8_t* state) const {
    int32_t idx = static_cast<int32_t>(sets_.size());
    sets_.emplace_back();
    std::memcpy(state, &idx, sizeof(idx));
}

void CountDistinctI64::update(uint8_t* const* groupSlots,
                              const ColumnVector* input,
                              const PositionList& pos) const {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "CountDistinctI64: input column required");
    }
    if (input->type().id != TypeId::INT64) {
        raise(ErrorCode::TypeMismatch, "CountDistinctI64: input type {} != INT64",
              input->type().toString());
    }
    const int64_t*         vals     = input->typed<int64_t>();
    const ValidityBitmask& validity = input->validity();
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        int32_t idx;
        std::memcpy(&idx, groupSlots[k], sizeof(idx));
        sets_[static_cast<size_t>(idx)].insert(vals[i]);
    }
}

ColumnVector CountDistinctI64::finalize(const std::vector<const uint8_t*>& states,
                                        MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Int64, states.size(), mm);
    out.validity().setAllValid();
    int64_t* o = out.typed<int64_t>();
    for (size_t i = 0; i < states.size(); ++i) {
        int32_t idx;
        std::memcpy(&idx, states[i], sizeof(idx));
        o[i] = static_cast<int64_t>(sets_[static_cast<size_t>(idx)].size());
    }
    return out;
}

void CountDistinctI64::merge(uint8_t* dst, const uint8_t* src) const {
    int32_t s, d;
    std::memcpy(&s, src, sizeof(s));
    std::memcpy(&d, dst, sizeof(d));
    auto& srcSet = sets_[static_cast<size_t>(s)];
    sets_[static_cast<size_t>(d)].insert(srcSet.begin(), srcSet.end());
    srcSet.clear();  // src is scratch; release its memory now
}

// Serialized state: [int64 n][n × int64].
void CountDistinctI64::serializeState(const uint8_t* state,
                                      std::vector<uint8_t>& out) const {
    int32_t idx;
    std::memcpy(&idx, state, sizeof(idx));
    const auto& set = sets_[static_cast<size_t>(idx)];
    appendPod(out, static_cast<int64_t>(set.size()));
    for (int64_t v : set) appendPod(out, v);
}

size_t CountDistinctI64::deserializeState(uint8_t* state,
                                          const uint8_t* data) const {
    int32_t idx;
    std::memcpy(&idx, state, sizeof(idx));  // init() allocated a fresh set
    auto& set = sets_[static_cast<size_t>(idx)];
    const uint8_t* p = data;
    const auto n = readPod<int64_t>(p);
    set.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) set.insert(readPod<int64_t>(p));
    return static_cast<size_t>(p - data);
}

// ---------- CountDistinctString ----------

CountDistinctString::CountDistinctString(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "CountDistinctString: input expression required");
}

void CountDistinctString::init(uint8_t* state) const {
    int32_t idx = static_cast<int32_t>(sets_.size());
    sets_.emplace_back();
    std::memcpy(state, &idx, sizeof(idx));
}

void CountDistinctString::update(uint8_t* const* groupSlots,
                                 const ColumnVector* input,
                                 const PositionList& pos) const {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "CountDistinctString: input column required");
    }
    if (input->type().id != TypeId::STRING) {
        raise(ErrorCode::TypeMismatch, "CountDistinctString: input type {} != STRING",
              input->type().toString());
    }
    const ValidityBitmask& validity = input->validity();
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        int32_t idx;
        std::memcpy(&idx, groupSlots[k], sizeof(idx));
        const std::string_view sv = input->stringAt(i);
        sets_[static_cast<size_t>(idx)].emplace(sv);
    }
}

ColumnVector CountDistinctString::finalize(const std::vector<const uint8_t*>& states,
                                           MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Int64, states.size(), mm);
    out.validity().setAllValid();
    int64_t* o = out.typed<int64_t>();
    for (size_t i = 0; i < states.size(); ++i) {
        int32_t idx;
        std::memcpy(&idx, states[i], sizeof(idx));
        o[i] = static_cast<int64_t>(sets_[static_cast<size_t>(idx)].size());
    }
    return out;
}

void CountDistinctString::merge(uint8_t* dst, const uint8_t* src) const {
    int32_t s, d;
    std::memcpy(&s, src, sizeof(s));
    std::memcpy(&d, dst, sizeof(d));
    auto& srcSet = sets_[static_cast<size_t>(s)];
    auto& dstSet = sets_[static_cast<size_t>(d)];
    while (!srcSet.empty()) dstSet.insert(srcSet.extract(srcSet.begin()));
}

// Serialized state: [int64 n][n × (int32 len + len bytes)].
void CountDistinctString::serializeState(const uint8_t* state,
                                         std::vector<uint8_t>& out) const {
    int32_t idx;
    std::memcpy(&idx, state, sizeof(idx));
    const auto& set = sets_[static_cast<size_t>(idx)];
    appendPod(out, static_cast<int64_t>(set.size()));
    for (const auto& v : set) {
        appendPod(out, static_cast<int32_t>(v.size()));
        out.insert(out.end(), v.begin(), v.end());
    }
}

size_t CountDistinctString::deserializeState(uint8_t* state,
                                             const uint8_t* data) const {
    int32_t idx;
    std::memcpy(&idx, state, sizeof(idx));
    auto& set = sets_[static_cast<size_t>(idx)];
    const uint8_t* p = data;
    const auto n = readPod<int64_t>(p);
    set.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const auto len = readPod<int32_t>(p);
        set.emplace(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
        p += len;
    }
    return static_cast<size_t>(p - data);
}

// ---------- CountDistinctF64 ----------

CountDistinctF64::CountDistinctF64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "CountDistinctF64: input expression required");
}

void CountDistinctF64::init(uint8_t* state) const {
    int32_t idx = static_cast<int32_t>(sets_.size());
    sets_.emplace_back();
    std::memcpy(state, &idx, sizeof(idx));
}

void CountDistinctF64::update(uint8_t* const* groupSlots,
                              const ColumnVector* input,
                              const PositionList& pos) const {
    if (input == nullptr) {
        raise(ErrorCode::InvalidArgument, "CountDistinctF64: input column required");
    }
    if (input->type().id != TypeId::FLOAT64) {
        raise(ErrorCode::TypeMismatch, "CountDistinctF64: input type {} != FLOAT64",
              input->type().toString());
    }
    const double*          vals     = input->typed<double>();
    const ValidityBitmask& validity = input->validity();
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        int32_t idx;
        std::memcpy(&idx, groupSlots[k], sizeof(idx));
        sets_[static_cast<size_t>(idx)].insert(vals[i]);
    }
}

ColumnVector CountDistinctF64::finalize(const std::vector<const uint8_t*>& states,
                                        MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Int64, states.size(), mm);
    out.validity().setAllValid();
    int64_t* o = out.typed<int64_t>();
    for (size_t i = 0; i < states.size(); ++i) {
        int32_t idx;
        std::memcpy(&idx, states[i], sizeof(idx));
        o[i] = static_cast<int64_t>(sets_[static_cast<size_t>(idx)].size());
    }
    return out;
}

void CountDistinctF64::merge(uint8_t* dst, const uint8_t* src) const {
    int32_t s, d;
    std::memcpy(&s, src, sizeof(s));
    std::memcpy(&d, dst, sizeof(d));
    auto& srcSet = sets_[static_cast<size_t>(s)];
    sets_[static_cast<size_t>(d)].insert(srcSet.begin(), srcSet.end());
    srcSet.clear();
}

// Serialized state: [int64 n][n × double] (bit patterns, NaN-safe).
void CountDistinctF64::serializeState(const uint8_t* state,
                                      std::vector<uint8_t>& out) const {
    int32_t idx;
    std::memcpy(&idx, state, sizeof(idx));
    const auto& set = sets_[static_cast<size_t>(idx)];
    appendPod(out, static_cast<int64_t>(set.size()));
    for (double v : set) appendPod(out, v);
}

size_t CountDistinctF64::deserializeState(uint8_t* state,
                                          const uint8_t* data) const {
    int32_t idx;
    std::memcpy(&idx, state, sizeof(idx));
    auto& set = sets_[static_cast<size_t>(idx)];
    const uint8_t* p = data;
    const auto n = readPod<int64_t>(p);
    set.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) set.insert(readPod<double>(p));
    return static_cast<size_t>(p - data);
}

// ---------- SumF64 ----------

SumF64::SumF64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "SumF64: input expression required");
}

void SumF64::init(uint8_t* state) const {
    double z = 0.0;
    std::memcpy(state, &z, sizeof(z));
}

void SumF64::update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const {
    if (input == nullptr) raise(ErrorCode::InvalidArgument, "SumF64: input column required");
    if (input->type().id != TypeId::FLOAT64) {
        raise(ErrorCode::TypeMismatch, "SumF64: input type {} != FLOAT64",
              input->type().toString());
    }
    const double*          vals     = input->typed<double>();
    const ValidityBitmask& validity = input->validity();
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        double cur;
        std::memcpy(&cur, groupSlots[k], sizeof(cur));
        cur += vals[i];
        std::memcpy(groupSlots[k], &cur, sizeof(cur));
    }
}

ColumnVector SumF64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Float64, states.size(), mm);
    out.validity().setAllValid();
    double* o = out.typed<double>();
    for (size_t i = 0; i < states.size(); ++i) {
        std::memcpy(&o[i], states[i], sizeof(double));
    }
    return out;
}

void SumF64::merge(uint8_t* dst, const uint8_t* src) const {
    double a, b;
    std::memcpy(&a, dst, sizeof(a));
    std::memcpy(&b, src, sizeof(b));
    a += b;
    std::memcpy(dst, &a, sizeof(a));
}

// ---------- MIN/MAX F64 helpers ----------

namespace {

void initMinMaxF64State(uint8_t* state) { std::memset(state, 0, 16); }

inline bool   minMaxF64HasValue(const uint8_t* state) noexcept { return state[8] != 0; }
inline double minMaxF64Value   (const uint8_t* state) noexcept {
    double v; std::memcpy(&v, state, sizeof(v)); return v;
}
inline void   minMaxF64Set     (uint8_t* state, double v) noexcept {
    std::memcpy(state, &v, sizeof(v));
    state[8] = 1;
}

template <typename Cmp>
void updateMinMaxF64(uint8_t* const* groupSlots,
                     const ColumnVector* input,
                     const PositionList& pos,
                     const char* opName) {
    if (input == nullptr) raise(ErrorCode::InvalidArgument, "{}: input column required", opName);
    if (input->type().id != TypeId::FLOAT64) {
        raise(ErrorCode::TypeMismatch, "{}: input type {} != FLOAT64", opName,
              input->type().toString());
    }
    const double*          vals     = input->typed<double>();
    const ValidityBitmask& validity = input->validity();
    Cmp                    cmp;
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        uint8_t* st = groupSlots[k];
        if (!minMaxF64HasValue(st) || cmp(vals[i], minMaxF64Value(st))) {
            minMaxF64Set(st, vals[i]);
        }
    }
}

ColumnVector finalizeMinMaxF64(const std::vector<const uint8_t*>& states,
                               MemoryManager* mm) {
    ColumnVector out = ColumnVector::make(Float64, states.size(), mm);
    out.validity().setAllValid();
    double* o = out.typed<double>();
    for (size_t i = 0; i < states.size(); ++i) {
        if (minMaxF64HasValue(states[i])) {
            o[i] = minMaxF64Value(states[i]);
        } else {
            o[i] = 0.0;
            out.validity().setValid(i, false);
        }
    }
    return out;
}

template <typename Cmp>
void mergeMinMaxF64(uint8_t* dst, const uint8_t* src) {
    if (!minMaxF64HasValue(src)) return;
    if (!minMaxF64HasValue(dst)) { std::memcpy(dst, src, 16); return; }
    Cmp cmp;
    if (cmp(minMaxF64Value(src), minMaxF64Value(dst))) {
        minMaxF64Set(dst, minMaxF64Value(src));
    }
}

} // namespace

// ---------- MinF64 ----------

MinF64::MinF64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "MinF64: input expression required");
}
void MinF64::init(uint8_t* state) const { initMinMaxF64State(state); }
void MinF64::update(uint8_t* const* groupSlots, const ColumnVector* input,
                    const PositionList& pos) const {
    updateMinMaxF64<std::less<double>>(groupSlots, input, pos, "MinF64");
}
ColumnVector MinF64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    return finalizeMinMaxF64(states, mm);
}
void MinF64::merge(uint8_t* dst, const uint8_t* src) const {
    mergeMinMaxF64<std::less<double>>(dst, src);
}

// ---------- MaxF64 ----------

MaxF64::MaxF64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "MaxF64: input expression required");
}
void MaxF64::init(uint8_t* state) const { initMinMaxF64State(state); }
void MaxF64::update(uint8_t* const* groupSlots, const ColumnVector* input,
                    const PositionList& pos) const {
    updateMinMaxF64<std::greater<double>>(groupSlots, input, pos, "MaxF64");
}
ColumnVector MaxF64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    return finalizeMinMaxF64(states, mm);
}
void MaxF64::merge(uint8_t* dst, const uint8_t* src) const {
    mergeMinMaxF64<std::greater<double>>(dst, src);
}

// ---------- AvgF64 ----------

AvgF64::AvgF64(ExpressionPtr input) : input_(std::move(input)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "AvgF64: input expression required");
}

void AvgF64::init(uint8_t* state) const {
    double  zSum   = 0.0;
    int64_t zCount = 0;
    std::memcpy(state,     &zSum,   sizeof(zSum));
    std::memcpy(state + 8, &zCount, sizeof(zCount));
}

void AvgF64::update(uint8_t* const* groupSlots,
                    const ColumnVector* input,
                    const PositionList& pos) const {
    if (input == nullptr) raise(ErrorCode::InvalidArgument, "AvgF64: input column required");
    if (input->type().id != TypeId::FLOAT64) {
        raise(ErrorCode::TypeMismatch, "AvgF64: input type {} != FLOAT64",
              input->type().toString());
    }
    const double*          vals     = input->typed<double>();
    const ValidityBitmask& validity = input->validity();
    const size_t n = pos.size();
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!validity.isValid(i)) continue;
        uint8_t* st = groupSlots[k];
        double  sum;
        int64_t cnt;
        std::memcpy(&sum, st,     sizeof(sum));
        std::memcpy(&cnt, st + 8, sizeof(cnt));
        sum += vals[i];
        cnt += 1;
        std::memcpy(st,     &sum, sizeof(sum));
        std::memcpy(st + 8, &cnt, sizeof(cnt));
    }
}

ColumnVector AvgF64::finalize(const std::vector<const uint8_t*>& states,
                              MemoryManager* mm) const {
    ColumnVector out = ColumnVector::make(Float64, states.size(), mm);
    out.validity().setAllValid();
    double* o = out.typed<double>();
    for (size_t i = 0; i < states.size(); ++i) {
        double  sum;
        int64_t cnt;
        std::memcpy(&sum, states[i],     sizeof(sum));
        std::memcpy(&cnt, states[i] + 8, sizeof(cnt));
        if (cnt == 0) {
            o[i] = 0.0;
            out.validity().setValid(i, false);
        } else {
            o[i] = sum / static_cast<double>(cnt);
        }
    }
    return out;
}

void AvgF64::merge(uint8_t* dst, const uint8_t* src) const {
    double  sumA, sumB;
    int64_t cntA, cntB;
    std::memcpy(&sumA, dst,     sizeof(sumA));
    std::memcpy(&cntA, dst + 8, sizeof(cntA));
    std::memcpy(&sumB, src,     sizeof(sumB));
    std::memcpy(&cntB, src + 8, sizeof(cntB));
    sumA += sumB;
    cntA += cntB;
    std::memcpy(dst,     &sumA, sizeof(sumA));
    std::memcpy(dst + 8, &cntA, sizeof(cntA));
}

} // namespace quantum
