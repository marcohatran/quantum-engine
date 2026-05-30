#include "operator/HashJoinOperator.hpp"

#include <cstring>

#include "common/Bits.hpp"
#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

namespace {

constexpr int32_t kEndOfChain = -1;

void initEndOfChain(uint8_t* state) {
    int32_t v = kEndOfChain;
    std::memcpy(state, &v, sizeof(v));
}

ColumnVector emptyColumn(DataType type, MemoryManager* mm) {
    if (type.isFixedWidth()) {
        return ColumnVector::make(type, 0, mm);
    }
    StringBuilder b(mm);
    return b.build();
}

ColumnVector allNullColumn(DataType type, size_t n, MemoryManager* mm) {
    if (type.isFixedWidth()) {
        ColumnVector c = ColumnVector::make(type, n, mm);
        c.validity().setAllNull();
        return c;
    }
    StringBuilder b(mm);
    for (size_t i = 0; i < n; ++i) b.appendNull();
    return b.build();
}

ColumnVector concatFixedWidthColumn(
        const std::vector<ColumnBatch>& batches, size_t col,
        size_t total, DataType type, MemoryManager* mm) {
    ColumnVector out = ColumnVector::make(type, total, mm);
    out.validity().setAllValid();
    const size_t width = static_cast<size_t>(type.fixedWidth());
    size_t       row   = 0;
    for (const auto& b : batches) {
        const auto& src = b.column(col);
        std::memcpy(out.values().data() + row * width,
                    src.values().data(),
                    src.numRows() * width);
        for (size_t i = 0; i < src.numRows(); ++i) {
            out.validity().setValid(row + i, src.validity().isValid(i));
        }
        row += src.numRows();
    }
    return out;
}

ColumnVector concatVariableWidthColumn(
        const std::vector<ColumnBatch>& batches, size_t col,
        size_t total, DataType type, MemoryManager* mm) {
    // If any input batch carries this column dict-encoded, fall back to
    // decoding everything into a regular STRING via StringBuilder. Merging
    // dictionaries across batches would preserve encoding but is a follow-up.
    bool anyDict = false;
    for (const auto& b : batches) {
        if (b.column(col).isDictEncoded()) { anyDict = true; break; }
    }
    if (anyDict) {
        StringBuilder sb(mm);
        for (const auto& b : batches) {
            const auto& src = b.column(col);
            for (size_t i = 0; i < src.numRows(); ++i) {
                if (src.validity().isValid(i)) sb.append(src.stringAt(i));
                else                            sb.appendNull();
            }
        }
        (void)total;
        return sb.build();
    }

    size_t totalBytes = 0;
    for (const auto& b : batches) {
        const auto& src = b.column(col);
        const int32_t* off = src.offsets().as<int32_t>();
        totalBytes += static_cast<size_t>(off[src.numRows()]);
    }

    Buffer  offsetsBuf = mm->allocate(roundUp((total + 1) * sizeof(int32_t), kCacheLineBytes));
    Buffer  valuesBuf  = mm->allocate(roundUp(totalBytes, kCacheLineBytes));
    int32_t* outOff    = offsetsBuf.as<int32_t>();
    char*    outVals   = valuesBuf.as<char>();

    size_t  rowCursor  = 0;
    int64_t byteCursor = 0;
    outOff[0] = 0;
    for (const auto& b : batches) {
        const auto&    src    = b.column(col);
        const int32_t* srcOff = src.offsets().as<int32_t>();
        const size_t   n      = src.numRows();
        const size_t   blen   = static_cast<size_t>(srcOff[n]);
        if (blen > 0) {
            std::memcpy(outVals + byteCursor, src.values().as<char>(), blen);
        }
        for (size_t i = 0; i < n; ++i) {
            outOff[rowCursor + i + 1] =
                static_cast<int32_t>(byteCursor + srcOff[i + 1]);
        }
        rowCursor  += n;
        byteCursor += static_cast<int64_t>(blen);
    }

    ColumnVector out = ColumnVector::makeVariableWidth(
        type, total, std::move(offsetsBuf), std::move(valuesBuf), mm);
    out.validity().setAllValid();
    size_t row = 0;
    for (const auto& b : batches) {
        const auto& src = b.column(col);
        for (size_t i = 0; i < src.numRows(); ++i) {
            out.validity().setValid(row + i, src.validity().isValid(i));
        }
        row += src.numRows();
    }
    return out;
}

ColumnBatch concatDense(std::vector<ColumnBatch> batches, MemoryManager* mm) {
    if (batches.empty()) return ColumnBatch({}, PositionList::identity(0));

    const size_t numCols = batches[0].numColumns();
    size_t       total   = 0;
    for (const auto& b : batches) total += b.numRows();

    std::vector<ColumnVector> outCols;
    outCols.reserve(numCols);

    for (size_t c = 0; c < numCols; ++c) {
        const DataType type = batches[0].column(c).type();
        outCols.push_back(type.isFixedWidth()
            ? concatFixedWidthColumn(batches, c, total, type, mm)
            : concatVariableWidthColumn(batches, c, total, type, mm));
    }
    return ColumnBatch(std::move(outCols), PositionList::identity(total));
}

ColumnVector gatherDictEncoded(const ColumnVector&         src,
                               const std::vector<int32_t>& idx,
                               MemoryManager*              mm) {
    const size_t   nOut     = idx.size();
    Buffer  codes = mm->allocate(roundUp(nOut * sizeof(int32_t), kCacheLineBytes));
    const int32_t* srcCodes = src.values().as<int32_t>();
    int32_t*       outCodes = codes.as<int32_t>();
    for (size_t k = 0; k < nOut; ++k) outCodes[k] = srcCodes[idx[k]];

    Buffer newDictOff = mm->allocate(roundUp(src.dictOffsets().size(), kCacheLineBytes));
    Buffer newDictVal = mm->allocate(roundUp(src.dictValues().size(),  kCacheLineBytes));
    std::memcpy(newDictOff.data(), src.dictOffsets().data(), src.dictOffsets().size());
    if (src.dictValues().size() > 0) {
        std::memcpy(newDictVal.data(), src.dictValues().data(), src.dictValues().size());
    }
    ColumnVector out = ColumnVector::makeDictEncoded(
        src.type(), nOut, src.dictSize(), std::move(codes),
        std::move(newDictOff), std::move(newDictVal), mm);
    out.validity().setAllValid();
    for (size_t k = 0; k < nOut; ++k) {
        out.validity().setValid(k, src.validity().isValid(static_cast<size_t>(idx[k])));
    }
    return out;
}

ColumnVector gatherVariableWidth(const ColumnVector&         src,
                                 const std::vector<int32_t>& idx,
                                 MemoryManager*              mm) {
    if (src.isDictEncoded()) return gatherDictEncoded(src, idx, mm);

    const size_t   nOut   = idx.size();
    const int32_t* srcOff = src.offsets().as<int32_t>();

    Buffer  offsetsBuf = mm->allocate(roundUp((nOut + 1) * sizeof(int32_t), kCacheLineBytes));
    int32_t* outOff    = offsetsBuf.as<int32_t>();
    int64_t running = 0;
    for (size_t k = 0; k < nOut; ++k) {
        outOff[k] = static_cast<int32_t>(running);
        const auto i = static_cast<size_t>(idx[k]);
        running += srcOff[i + 1] - srcOff[i];
    }
    outOff[nOut] = static_cast<int32_t>(running);

    Buffer values = mm->allocate(roundUp(static_cast<size_t>(running), kCacheLineBytes));
    char*       outVals = values.as<char>();
    const char* srcVals = src.values().as<char>();
    for (size_t k = 0; k < nOut; ++k) {
        const auto i   = static_cast<size_t>(idx[k]);
        const auto beg = static_cast<size_t>(srcOff[i]);
        const auto len = static_cast<size_t>(srcOff[i + 1] - srcOff[i]);
        if (len > 0) std::memcpy(outVals + outOff[k], srcVals + beg, len);
    }

    ColumnVector out = ColumnVector::makeVariableWidth(
        src.type(), nOut, std::move(offsetsBuf), std::move(values), mm);
    out.validity().setAllValid();
    for (size_t k = 0; k < nOut; ++k) {
        out.validity().setValid(k, src.validity().isValid(static_cast<size_t>(idx[k])));
    }
    return out;
}

ColumnVector gather(const ColumnVector&         src,
                    const std::vector<int32_t>& idx,
                    MemoryManager*              mm) {
    if (src.isVariableWidth()) return gatherVariableWidth(src, idx, mm);

    ColumnVector out = ColumnVector::make(src.type(), idx.size(), mm);
    out.validity().setAllValid();

    switch (src.type().id) {
        case TypeId::INT32: {
            const auto* sp = src.typed<int32_t>();
            auto*       op = out.typed<int32_t>();
            for (size_t k = 0; k < idx.size(); ++k) op[k] = sp[idx[k]];
            break;
        }
        case TypeId::INT64: {
            const auto* sp = src.typed<int64_t>();
            auto*       op = out.typed<int64_t>();
            for (size_t k = 0; k < idx.size(); ++k) op[k] = sp[idx[k]];
            break;
        }
        case TypeId::FLOAT64: {
            const auto* sp = src.typed<double>();
            auto*       op = out.typed<double>();
            for (size_t k = 0; k < idx.size(); ++k) op[k] = sp[idx[k]];
            break;
        }
        case TypeId::BOOL: {
            const auto* sp = src.typed<uint8_t>();
            auto*       op = out.typed<uint8_t>();
            for (size_t k = 0; k < idx.size(); ++k) op[k] = sp[idx[k]];
            break;
        }
        default:
            raise(ErrorCode::NotImplemented,
                  "HashJoin gather: type {} not supported", src.type().toString());
    }

    for (size_t k = 0; k < idx.size(); ++k) {
        out.validity().setValid(k, src.validity().isValid(static_cast<size_t>(idx[k])));
    }
    return out;
}

std::vector<DataType> keyTypesFromExprs(const std::vector<ExpressionPtr>& exprs,
                                        const Schema&                     schema) {
    std::vector<DataType> out;
    out.reserve(exprs.size());
    for (const auto& e : exprs) out.push_back(e->resultType(schema));
    return out;
}

} // namespace

HashJoinOperator::HashJoinOperator(MemoryManager*               mm,
                                   OperatorPtr                  buildSide,
                                   OperatorPtr                  probeSide,
                                   std::vector<ExpressionPtr>   buildKeys,
                                   std::vector<ExpressionPtr>   probeKeys,
                                   std::vector<std::string>     outputNames,
                                   JoinType                     joinType)
    : Operator(mm),
      build_(std::move(buildSide)),
      probe_(std::move(probeSide)),
      buildKeys_(std::move(buildKeys)),
      probeKeys_(std::move(probeKeys)),
      joinType_(joinType) {

    if (buildKeys_.empty() || buildKeys_.size() != probeKeys_.size()) {
        raise(ErrorCode::InvalidArgument,
              "HashJoin: build/probe key counts must match and be non-zero (build={}, probe={})",
              buildKeys_.size(), probeKeys_.size());
    }

    const Schema& bs = build_->outputSchema();
    const Schema& ps = probe_->outputSchema();

    if (outputNames.size() != bs.numFields() + ps.numFields()) {
        raise(ErrorCode::InvalidArgument,
              "HashJoin: {} output names, expected {} (build:{} + probe:{})",
              outputNames.size(), bs.numFields() + ps.numFields(),
              bs.numFields(), ps.numFields());
    }

    auto buildTypes = keyTypesFromExprs(buildKeys_, bs);
    auto probeTypes = keyTypesFromExprs(probeKeys_, ps);
    for (size_t i = 0; i < buildTypes.size(); ++i) {
        if (buildTypes[i] != probeTypes[i]) {
            raise(ErrorCode::TypeMismatch,
                  "HashJoin: key {} types differ (build={}, probe={})",
                  i, buildTypes[i].toString(), probeTypes[i].toString());
        }
    }
    codec_ = std::make_unique<KeyCodec>(buildTypes);

    std::vector<Field> fields;
    fields.reserve(outputNames.size());
    for (size_t i = 0; i < bs.numFields(); ++i) {
        fields.push_back(Field{std::move(outputNames[i]), bs.field(i).type});
    }
    for (size_t i = 0; i < ps.numFields(); ++i) {
        fields.push_back(Field{std::move(outputNames[bs.numFields() + i]),
                               ps.field(i).type});
    }
    outSchema_ = Schema(std::move(fields));
}

void HashJoinOperator::open() {
    build_->open();
    probe_->open();
}

void HashJoinOperator::close() {
    build_->close();
    probe_->close();
}

void HashJoinOperator::build() {
    std::vector<ColumnBatch> dense;
    while (auto b = build_->getNext()) {
        if (b->activeRows() == 0) continue;
        dense.push_back(b->selectActive(mm_));
    }

    if (dense.empty()) {
        const Schema& bs = build_->outputSchema();
        std::vector<ColumnVector> emptyCols;
        emptyCols.reserve(bs.numFields());
        for (size_t c = 0; c < bs.numFields(); ++c) {
            emptyCols.push_back(emptyColumn(bs.field(c).type, mm_));
        }
        rowStore_ = ColumnBatch(std::move(emptyCols), PositionList::identity(0));
    } else {
        rowStore_ = concatDense(std::move(dense), mm_);
    }

    if (codec_->isFixedWidth()) {
        indexFixed_ = std::make_unique<BytesHashTable>(
            codec_->keyBytes(), sizeof(int32_t), mm_);
    } else {
        indexVar_ = std::make_unique<VarBytesHashTable>(sizeof(int32_t), mm_);
    }
    next_.assign(rowStore_->numRows(), kEndOfChain);
    buildMatched_.assign(rowStore_->numRows(), false);

    if (rowStore_->numRows() == 0) {
        built_ = true;
        return;
    }

    std::vector<ColumnVector> keyEvals;
    keyEvals.reserve(buildKeys_.size());
    for (const auto& e : buildKeys_) keyEvals.push_back(e->eval(*rowStore_, mm_));
    std::vector<const ColumnVector*> keyPtrs;
    keyPtrs.reserve(keyEvals.size());
    for (const auto& c : keyEvals) keyPtrs.push_back(&c);

    std::vector<uint8_t> keyBufFixed(codec_->isFixedWidth() ? codec_->keyBytes() : 0);
    std::vector<uint8_t> keyBufVar;

    for (size_t i = 0; i < rowStore_->numRows(); ++i) {
        uint8_t* slot = nullptr;
        if (codec_->isFixedWidth()) {
            codec_->encode(keyPtrs, i, keyBufFixed.data());
            if (codec_->hasAnyNull(keyBufFixed.data())) continue;
            slot = indexFixed_->findOrInsert(keyBufFixed.data(), initEndOfChain);
        } else {
            codec_->encodeVar(keyPtrs, i, keyBufVar);
            if (codec_->hasAnyNullVar(keyBufVar.data(), keyBufVar.size())) continue;
            slot = indexVar_->findOrInsert(
                keyBufVar.data(), keyBufVar.size(), initEndOfChain);
        }
        int32_t prevHead;
        std::memcpy(&prevHead, slot, sizeof(prevHead));
        next_[i] = prevHead;
        int32_t newHead = static_cast<int32_t>(i);
        std::memcpy(slot, &newHead, sizeof(newHead));
    }

    built_ = true;
}

std::optional<ColumnBatch> HashJoinOperator::probeOne() {
    auto in = probe_->getNext();
    if (!in) {
        probeExhausted_ = true;
        return std::nullopt;
    }
    if (in->activeRows() == 0) {
        return ColumnBatch(std::vector<ColumnVector>{}, PositionList::identity(0));
    }

    std::vector<ColumnVector> keyEvals;
    keyEvals.reserve(probeKeys_.size());
    for (const auto& e : probeKeys_) keyEvals.push_back(e->eval(*in, mm_));
    std::vector<const ColumnVector*> keyPtrs;
    keyPtrs.reserve(keyEvals.size());
    for (const auto& c : keyEvals) keyPtrs.push_back(&c);

    const PositionList& pos = in->positions();
    const bool emitLeftMisses = (joinType_ == JoinType::LeftOuter ||
                                 joinType_ == JoinType::FullOuter);
    const bool buildEmpty  = (rowStore_->numRows() == 0);

    std::vector<uint8_t> keyBufFixed(codec_->isFixedWidth() ? codec_->keyBytes() : 0);
    std::vector<uint8_t> keyBufVar;
    std::vector<int32_t> probeIdx;
    std::vector<int32_t> buildIdx;
    std::vector<bool>    buildIsValid;

    auto emitUnmatched = [&](size_t probeRow) {
        probeIdx.push_back(static_cast<int32_t>(probeRow));
        buildIdx.push_back(0);
        buildIsValid.push_back(false);
    };

    // Dict-aware fast path: when the join key is a single dict-encoded STRING
    // column on the probe side, precompute per-batch probe-code -> build-side
    // chain head ONCE per unique dict entry. Per-row work then collapses to a
    // single array load + chain walk.
    const bool dictFastPath = !codec_->isFixedWidth() &&
                              probeKeys_.size() == 1 &&
                              keyEvals[0].type().id == TypeId::STRING &&
                              keyEvals[0].isDictEncoded() &&
                              !buildEmpty;
    std::vector<const uint8_t*> dictCodeToSlot;
    if (dictFastPath) {
        const ColumnVector& keyCol = keyEvals[0];
        const size_t        dictN  = keyCol.dictSize();
        dictCodeToSlot.assign(dictN, nullptr);
        for (size_t b = 0; b < dictN; ++b) {
            const int32_t* doff = keyCol.dictOffsets().as<int32_t>();
            const char*    dval = keyCol.dictValues().as<char>();
            std::string_view ds(dval + doff[b],
                                static_cast<size_t>(doff[b + 1] - doff[b]));
            // Encode this dictionary entry into the var-key format the build
            // index uses (validity bits + length-prefix + bytes).
            // We synthesize a one-row ColumnVector with this string via a
            // tiny StringBuilder.
            StringBuilder sb(mm_);
            sb.append(ds);
            auto tmpCol = sb.build();
            std::vector<const ColumnVector*> tmpPtrs{&tmpCol};
            codec_->encodeVar(tmpPtrs, 0, keyBufVar);
            if (codec_->hasAnyNullVar(keyBufVar.data(), keyBufVar.size())) continue;
            dictCodeToSlot[b] = indexVar_->find(keyBufVar.data(), keyBufVar.size());
        }
    }

    for (size_t k = 0; k < pos.size(); ++k) {
        const auto i = static_cast<size_t>(pos.at(k));

        const uint8_t* slot = nullptr;
        bool anyNull = false;
        if (dictFastPath) {
            const ColumnVector& keyCol = keyEvals[0];
            if (!keyCol.validity().isValid(i)) {
                anyNull = true;
            } else {
                const int32_t code = keyCol.values().as<int32_t>()[i];
                slot = dictCodeToSlot[static_cast<size_t>(code)];
            }
        } else if (codec_->isFixedWidth()) {
            codec_->encode(keyPtrs, i, keyBufFixed.data());
            anyNull = codec_->hasAnyNull(keyBufFixed.data());
            if (!anyNull && !buildEmpty) slot = indexFixed_->find(keyBufFixed.data());
        } else {
            codec_->encodeVar(keyPtrs, i, keyBufVar);
            anyNull = codec_->hasAnyNullVar(keyBufVar.data(), keyBufVar.size());
            if (!anyNull && !buildEmpty) {
                slot = indexVar_->find(keyBufVar.data(), keyBufVar.size());
            }
        }

        if (anyNull || buildEmpty || slot == nullptr) {
            if (emitLeftMisses) emitUnmatched(i);
            continue;
        }

        int32_t cur;
        std::memcpy(&cur, slot, sizeof(cur));
        while (cur != kEndOfChain) {
            probeIdx.push_back(static_cast<int32_t>(i));
            buildIdx.push_back(cur);
            buildIsValid.push_back(true);
            buildMatched_[static_cast<size_t>(cur)] = true;
            cur = next_[static_cast<size_t>(cur)];
        }
    }

    std::vector<ColumnVector> outCols;
    outCols.reserve(rowStore_->numColumns() + in->numColumns());

    for (size_t c = 0; c < rowStore_->numColumns(); ++c) {
        const auto& srcCol  = rowStore_->column(c);
        const auto  type    = srcCol.type();
        ColumnVector col = buildEmpty
            ? allNullColumn(type, buildIdx.size(), mm_)
            : gather(srcCol, buildIdx, mm_);
        for (size_t k = 0; k < buildIsValid.size(); ++k) {
            if (!buildIsValid[k]) col.validity().setValid(k, false);
        }
        outCols.push_back(std::move(col));
    }
    for (size_t c = 0; c < in->numColumns(); ++c) {
        outCols.push_back(gather(in->column(c), probeIdx, mm_));
    }

    return ColumnBatch(std::move(outCols), PositionList::identity(probeIdx.size()));
}

// Emit one batch holding every build row never matched during probe, with
// NULL probe columns. Returns nullopt when nothing's left to emit.
static std::optional<ColumnBatch> emitRightTailImpl(
        const std::optional<ColumnBatch>& rowStore,
        const std::vector<bool>&          buildMatched,
        const Schema&                     probeSchema,
        MemoryManager*                    mm) {
    std::vector<int32_t> unmatched;
    for (size_t i = 0; i < buildMatched.size(); ++i) {
        if (!buildMatched[i]) unmatched.push_back(static_cast<int32_t>(i));
    }
    if (unmatched.empty()) return std::nullopt;

    std::vector<ColumnVector> outCols;
    outCols.reserve(rowStore->numColumns() + probeSchema.numFields());
    for (size_t c = 0; c < rowStore->numColumns(); ++c) {
        outCols.push_back(gather(rowStore->column(c), unmatched, mm));
    }
    for (size_t c = 0; c < probeSchema.numFields(); ++c) {
        outCols.push_back(allNullColumn(probeSchema.field(c).type, unmatched.size(), mm));
    }
    return ColumnBatch(std::move(outCols), PositionList::identity(unmatched.size()));
}

std::optional<ColumnBatch> HashJoinOperator::getNext() {
    if (!built_) build();

    if (!probeExhausted_) {
        auto out = probeOne();
        // probeOne sets probeExhausted_ when the child returns nullopt; in
        // that case we fall through to the right-tail check below.
        if (out.has_value() || !probeExhausted_) return out;
    }

    if (!rightTailEmitted_ &&
        (joinType_ == JoinType::RightOuter || joinType_ == JoinType::FullOuter)) {
        rightTailEmitted_ = true;
        return emitRightTailImpl(rowStore_, buildMatched_,
                                 probe_->outputSchema(), mm_);
    }
    return std::nullopt;
}

} // namespace quantum
