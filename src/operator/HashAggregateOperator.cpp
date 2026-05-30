#include "operator/HashAggregateOperator.hpp"

#include <cstring>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"
#include "vector/StringBuilder.hpp"

namespace quantum {

namespace {

struct InitCtx {
    const std::vector<AggregateFunctionPtr>* aggs;
    const std::vector<size_t>*               offsets;
};

thread_local InitCtx* g_initCtx = nullptr;

void initSlab(uint8_t* slab) {
    const auto& aggs    = *g_initCtx->aggs;
    const auto& offsets = *g_initCtx->offsets;
    for (size_t i = 0; i < aggs.size(); ++i) {
        aggs[i]->init(slab + offsets[i]);
    }
}

} // namespace

HashAggregateOperator::HashAggregateOperator(
        MemoryManager*                    mm,
        OperatorPtr                       child,
        std::vector<ExpressionPtr>        groupKeys,
        std::vector<AggregateFunctionPtr> aggs,
        std::vector<std::string>          outputNames)
    : Operator(mm),
      child_(std::move(child)),
      groupKeys_(std::move(groupKeys)),
      aggs_(std::move(aggs)) {

    if (groupKeys_.empty()) {
        raise(ErrorCode::InvalidArgument,
              "HashAggregate: at least one group key required");
    }
    if (outputNames.size() != groupKeys_.size() + aggs_.size()) {
        raise(ErrorCode::InvalidArgument,
              "HashAggregate: {} output names, expected {} ({} keys + {} aggs)",
              outputNames.size(), groupKeys_.size() + aggs_.size(),
              groupKeys_.size(), aggs_.size());
    }

    aggOffsets_.reserve(aggs_.size());
    for (const auto& a : aggs_) {
        aggOffsets_.push_back(stateBytes_);
        stateBytes_ += a->stateSize();
    }
    if (stateBytes_ == 0) stateBytes_ = 1;

    const Schema& inSchema = child_->outputSchema();
    keyTypes_.reserve(groupKeys_.size());
    for (auto& e : groupKeys_) keyTypes_.push_back(e->resultType(inSchema));
    codec_ = std::make_unique<KeyCodec>(keyTypes_);

    std::vector<Field> fields;
    fields.reserve(outputNames.size());
    for (size_t i = 0; i < groupKeys_.size(); ++i) {
        fields.push_back(Field{std::move(outputNames[i]), keyTypes_[i]});
    }
    for (size_t i = 0; i < aggs_.size(); ++i) {
        fields.push_back(Field{
            std::move(outputNames[groupKeys_.size() + i]),
            aggs_[i]->outputType()});
    }
    outSchema_ = Schema(std::move(fields));
}

void HashAggregateOperator::open()  { child_->open(); }
void HashAggregateOperator::close() { child_->close(); }

void HashAggregateOperator::build() {
    if (codec_->isFixedWidth()) {
        tableFixed_ = std::make_unique<BytesHashTable>(
            codec_->keyBytes(), stateBytes_, mm_);
    } else {
        tableVar_ = std::make_unique<VarBytesHashTable>(stateBytes_, mm_);
    }

    InitCtx ctx{&aggs_, &aggOffsets_};
    g_initCtx = &ctx;

    std::vector<uint8_t>  keyBufFixed(codec_->isFixedWidth() ? codec_->keyBytes() : 0);
    std::vector<uint8_t>  keyBufVar;
    std::vector<uint8_t*> groupSlots;

    while (auto in = child_->getNext()) {
        const PositionList& pos = in->positions();
        if (pos.size() == 0) continue;

        std::vector<ColumnVector> keyEvals;
        keyEvals.reserve(groupKeys_.size());
        for (const auto& e : groupKeys_) keyEvals.push_back(e->eval(*in, mm_));
        std::vector<const ColumnVector*> keyPtrs;
        keyPtrs.reserve(keyEvals.size());
        for (const auto& c : keyEvals) keyPtrs.push_back(&c);

        groupSlots.clear();
        groupSlots.reserve(pos.size());
        for (size_t k = 0; k < pos.size(); ++k) {
            const auto i = static_cast<size_t>(pos.at(k));
            if (codec_->isFixedWidth()) {
                codec_->encode(keyPtrs, i, keyBufFixed.data());
                groupSlots.push_back(tableFixed_->findOrInsert(keyBufFixed.data(), initSlab));
            } else {
                codec_->encodeVar(keyPtrs, i, keyBufVar);
                groupSlots.push_back(tableVar_->findOrInsert(
                    keyBufVar.data(), keyBufVar.size(), initSlab));
            }
        }

        for (size_t a = 0; a < aggs_.size(); ++a) {
            const auto& agg = aggs_[a];
            std::optional<ColumnVector> inputCol;
            if (const Expression* e = agg->inputExpr()) inputCol = e->eval(*in, mm_);

            std::vector<uint8_t*> slots;
            slots.reserve(groupSlots.size());
            for (auto* s : groupSlots) slots.push_back(s + aggOffsets_[a]);

            agg->update(slots.data(),
                        inputCol.has_value() ? &*inputCol : nullptr,
                        pos);
        }
    }

    g_initCtx = nullptr;
    built_ = true;
}

ColumnBatch HashAggregateOperator::emitFixed() {
    const size_t total = tableFixed_->numGroups();
    std::vector<ColumnVector> cols;
    cols.reserve(groupKeys_.size() + aggs_.size());
    for (const auto& t : keyTypes_) {
        ColumnVector c = ColumnVector::make(t, total, mm_);
        c.validity().setAllValid();
        cols.push_back(std::move(c));
    }
    std::vector<const uint8_t*> states;
    states.reserve(total);

    size_t row = 0;
    for (auto cur = tableFixed_->cursor(); cur.valid(); cur.next()) {
        for (size_t c = 0; c < groupKeys_.size(); ++c) {
            codec_->decode(cur.key(), c, cols[c], row);
        }
        states.push_back(cur.state());
        ++row;
    }

    for (size_t a = 0; a < aggs_.size(); ++a) {
        std::vector<const uint8_t*> slices;
        slices.reserve(states.size());
        for (const auto* s : states) slices.push_back(s + aggOffsets_[a]);
        cols.push_back(aggs_[a]->finalize(slices, mm_));
    }
    return ColumnBatch(std::move(cols), PositionList::identity(total));
}

ColumnBatch HashAggregateOperator::emitVar() {
    const size_t total = tableVar_->numGroups();

    // Pre-allocate fixed-width key columns; lazy-init StringBuilders for
    // variable-width key columns. Both kinds get populated inside the cursor
    // walk via codec_->decodeVar().
    std::vector<ColumnVector>                  fixedCols(groupKeys_.size());
    std::vector<std::unique_ptr<StringBuilder>> stringBuilders(groupKeys_.size());
    for (size_t c = 0; c < groupKeys_.size(); ++c) {
        if (keyTypes_[c].isFixedWidth()) {
            fixedCols[c] = ColumnVector::make(keyTypes_[c], total, mm_);
            fixedCols[c].validity().setAllValid();
        } else {
            stringBuilders[c] = std::make_unique<StringBuilder>(mm_);
        }
    }

    std::vector<const uint8_t*> states;
    states.reserve(total);

    size_t row = 0;
    for (auto cur = tableVar_->cursor(); cur.valid(); cur.next()) {
        const uint8_t* key    = cur.key();
        const size_t   keyLen = cur.keyLen();
        codec_->decodeVar(key, keyLen,
            [&](size_t colIdx, bool isValid, const uint8_t* bytes, size_t bytesLen) {
                if (keyTypes_[colIdx].isFixedWidth()) {
                    auto& col = fixedCols[colIdx];
                    col.validity().setValid(row, isValid);
                    if (isValid) {
                        const size_t w = static_cast<size_t>(keyTypes_[colIdx].fixedWidth());
                        std::memcpy(col.values().data() + row * w, bytes, w);
                    }
                } else {
                    if (isValid) {
                        stringBuilders[colIdx]->append(
                            std::string_view(reinterpret_cast<const char*>(bytes), bytesLen));
                    } else {
                        stringBuilders[colIdx]->appendNull();
                    }
                }
            });
        states.push_back(cur.state());
        ++row;
    }

    std::vector<ColumnVector> cols;
    cols.reserve(groupKeys_.size() + aggs_.size());
    for (size_t c = 0; c < groupKeys_.size(); ++c) {
        if (keyTypes_[c].isFixedWidth()) {
            cols.push_back(std::move(fixedCols[c]));
        } else {
            cols.push_back(stringBuilders[c]->build());
        }
    }
    for (size_t a = 0; a < aggs_.size(); ++a) {
        std::vector<const uint8_t*> slices;
        slices.reserve(states.size());
        for (const auto* s : states) slices.push_back(s + aggOffsets_[a]);
        cols.push_back(aggs_[a]->finalize(slices, mm_));
    }
    return ColumnBatch(std::move(cols), PositionList::identity(total));
}

ColumnBatch HashAggregateOperator::emit() {
    return codec_->isFixedWidth() ? emitFixed() : emitVar();
}

std::optional<ColumnBatch> HashAggregateOperator::getNext() {
    if (!built_) build();
    if (emitted_) return std::nullopt;
    emitted_ = true;
    return emit();
}

} // namespace quantum
