#include "expression/LikeExpr.hpp"

#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

namespace {

// SQL LIKE with % and _. Anchored, non-backtracking iterative matcher.
bool likeMatch(std::string_view s, std::string_view p) noexcept {
    size_t si = 0;
    size_t pi = 0;
    size_t starIdx   = std::string_view::npos;  // last `%` position in p
    size_t starMatch = 0;                       // position in s when `%` was seen

    while (si < s.size()) {
        if (pi < p.size() && (p[pi] == '_' || p[pi] == s[si])) {
            ++si;
            ++pi;
        } else if (pi < p.size() && p[pi] == '%') {
            starIdx   = pi;
            starMatch = si;
            ++pi;
        } else if (starIdx != std::string_view::npos) {
            // Backtrack: let % consume one more char of s.
            pi = starIdx + 1;
            ++starMatch;
            si = starMatch;
        } else {
            return false;
        }
    }
    while (pi < p.size() && p[pi] == '%') ++pi;
    return pi == p.size();
}

} // namespace

LikeExpr::LikeExpr(ExpressionPtr input, std::string pattern)
    : input_(std::move(input)), pattern_(std::move(pattern)) {
    if (!input_) raise(ErrorCode::InvalidArgument, "LikeExpr: input expression required");
}

ColumnVector LikeExpr::eval(const ColumnBatch& batch, MemoryManager* mm) const {
    ColumnVector in = input_->eval(batch, mm);
    if (in.type().id != TypeId::STRING) {
        raise(ErrorCode::TypeMismatch,
              "LikeExpr: input type {} != STRING", in.type().toString());
    }

    ColumnVector out = ColumnVector::make(Bool, batch.numRows(), mm);
    out.validity().setAllNull();
    uint8_t* o = out.typed<uint8_t>();

    const ValidityBitmask& inValidity = in.validity();
    const PositionList&    pos        = batch.positions();
    const std::string_view pat        = pattern_;

    // Dict-aware fast path: evaluate the pattern against each dictionary entry
    // once, then for each row just do a code-indexed lookup. For a column with
    // dictSize <<= numRows (the common low-cardinality case) this drops the
    // pattern-match cost by ~ numRows / dictSize.
    if (in.isDictEncoded()) {
        const size_t   dictN = in.dictSize();
        const int32_t* doff  = in.dictOffsets().as<int32_t>();
        const char*    dval  = in.dictValues().as<char>();

        std::vector<uint8_t> dictResult(dictN);
        for (size_t d = 0; d < dictN; ++d) {
            std::string_view ds(dval + doff[d], static_cast<size_t>(doff[d + 1] - doff[d]));
            dictResult[d] = likeMatch(ds, pat) ? 1U : 0U;
        }

        const int32_t* codes = in.values().as<int32_t>();
        auto runOneDict = [&](size_t i) {
            const bool isV = inValidity.isValid(i);
            out.validity().setValid(i, isV);
            o[i] = isV ? dictResult[static_cast<size_t>(codes[i])] : 0U;
        };
        if (pos.isIdentity()) {
            for (size_t i = 0; i < pos.size(); ++i) runOneDict(i);
        } else {
            const auto idx = pos.indices();
            for (size_t k = 0; k < idx.size(); ++k) runOneDict(static_cast<size_t>(idx[k]));
        }
        return out;
    }

    auto runOne = [&](size_t i) {
        const bool isV = inValidity.isValid(i);
        out.validity().setValid(i, isV);
        if (!isV) {
            o[i] = 0;
        } else {
            o[i] = likeMatch(in.stringAt(i), pat) ? 1U : 0U;
        }
    };

    if (pos.isIdentity()) {
        for (size_t i = 0; i < pos.size(); ++i) runOne(i);
    } else {
        const auto idx = pos.indices();
        for (size_t k = 0; k < idx.size(); ++k) runOne(static_cast<size_t>(idx[k]));
    }
    return out;
}

std::string LikeExpr::toString() const {
    return fmt::format("({} LIKE \"{}\")", input_->toString(), pattern_);
}

} // namespace quantum
