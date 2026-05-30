#include "operator/HashJoinOperator.hpp"

#include <cstring>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

namespace {

constexpr int32_t kEndOfChain = -1;

void initEndOfChain(uint8_t* state) {
    int32_t v = kEndOfChain;
    std::memcpy(state, &v, sizeof(v));
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
        ColumnVector   out  = ColumnVector::make(type, total, mm);
        out.validity().setAllValid();

        const size_t width = static_cast<size_t>(type.fixedWidth());
        size_t       row   = 0;
        for (const auto& b : batches) {
            const auto& src = b.column(c);
            std::memcpy(out.values().data() + row * width,
                        src.values().data(),
                        src.numRows() * width);
            for (size_t i = 0; i < src.numRows(); ++i) {
                out.validity().setValid(row + i, src.validity().isValid(i));
            }
            row += src.numRows();
        }
        outCols.push_back(std::move(out));
    }
    return ColumnBatch(std::move(outCols), PositionList::identity(total));
}

ColumnVector gather(const ColumnVector&         src,
                    const std::vector<int32_t>& idx,
                    MemoryManager*              mm) {
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
        if (!buildTypes[i].isFixedWidth()) {
            raise(ErrorCode::NotImplemented,
                  "HashJoin: variable-width key type {} not supported",
                  buildTypes[i].toString());
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
        // Empty build — still materialize an empty row store with the build
        // schema's column placeholders so LEFT OUTER probes can construct
        // the output schema correctly.
        const Schema& bs = build_->outputSchema();
        std::vector<ColumnVector> emptyCols;
        emptyCols.reserve(bs.numFields());
        for (size_t c = 0; c < bs.numFields(); ++c) {
            emptyCols.push_back(ColumnVector::make(bs.field(c).type, 0, mm_));
        }
        rowStore_ = ColumnBatch(std::move(emptyCols), PositionList::identity(0));
    } else {
        rowStore_ = concatDense(std::move(dense), mm_);
    }

    index_ = std::make_unique<BytesHashTable>(codec_->keyBytes(), sizeof(int32_t), mm_);
    next_.assign(rowStore_->numRows(), kEndOfChain);

    if (rowStore_->numRows() == 0) {
        built_ = true;
        return;
    }

    // Evaluate every build key over the dense row store and encode per-row.
    std::vector<ColumnVector> keyEvals;
    keyEvals.reserve(buildKeys_.size());
    for (const auto& e : buildKeys_) keyEvals.push_back(e->eval(*rowStore_, mm_));
    std::vector<const ColumnVector*> keyPtrs;
    keyPtrs.reserve(keyEvals.size());
    for (const auto& c : keyEvals) keyPtrs.push_back(&c);

    std::vector<uint8_t> keyBuf(codec_->keyBytes());

    for (size_t i = 0; i < rowStore_->numRows(); ++i) {
        codec_->encode(keyPtrs, i, keyBuf.data());
        if (codec_->hasAnyNull(keyBuf.data())) continue;  // null keys never match
        uint8_t* slot = index_->findOrInsert(keyBuf.data(), initEndOfChain);
        int32_t  prevHead;
        std::memcpy(&prevHead, slot, sizeof(prevHead));
        next_[i] = prevHead;
        int32_t newHead = static_cast<int32_t>(i);
        std::memcpy(slot, &newHead, sizeof(newHead));
    }

    built_ = true;
}

std::optional<ColumnBatch> HashJoinOperator::probeOne() {
    auto in = probe_->getNext();
    if (!in) return std::nullopt;
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
    const bool leftOuter   = (joinType_ == JoinType::LeftOuter);
    const bool buildEmpty  = (rowStore_->numRows() == 0);

    std::vector<uint8_t> keyBuf(codec_->keyBytes());
    std::vector<int32_t> probeIdx;
    std::vector<int32_t> buildIdx;
    std::vector<bool>    buildIsValid;

    auto emitUnmatched = [&](size_t probeRow) {
        probeIdx.push_back(static_cast<int32_t>(probeRow));
        buildIdx.push_back(0);
        buildIsValid.push_back(false);
    };

    for (size_t k = 0; k < pos.size(); ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        codec_->encode(keyPtrs, i, keyBuf.data());

        if (codec_->hasAnyNull(keyBuf.data())) {
            if (leftOuter) emitUnmatched(i);
            continue;
        }
        if (buildEmpty) {
            if (leftOuter) emitUnmatched(i);
            continue;
        }

        const uint8_t* slot = index_->find(keyBuf.data());
        if (slot == nullptr) {
            if (leftOuter) emitUnmatched(i);
            continue;
        }
        int32_t cur;
        std::memcpy(&cur, slot, sizeof(cur));
        while (cur != kEndOfChain) {
            probeIdx.push_back(static_cast<int32_t>(i));
            buildIdx.push_back(cur);
            buildIsValid.push_back(true);
            cur = next_[static_cast<size_t>(cur)];
        }
    }

    std::vector<ColumnVector> outCols;
    outCols.reserve(rowStore_->numColumns() + in->numColumns());

    for (size_t c = 0; c < rowStore_->numColumns(); ++c) {
        ColumnVector col = buildEmpty
            ? ColumnVector::make(rowStore_->column(c).type(), buildIdx.size(), mm_)
            : gather(rowStore_->column(c), buildIdx, mm_);
        if (buildEmpty) col.validity().setAllNull();
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

std::optional<ColumnBatch> HashJoinOperator::getNext() {
    if (!built_) build();
    return probeOne();
}

} // namespace quantum
