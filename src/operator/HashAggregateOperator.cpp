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
        std::vector<std::string>          outputNames,
        size_t                            spillTriggerGroups,
        double                            spillFraction)
    : Operator(mm),
      child_(std::move(child)),
      groupKeys_(std::move(groupKeys)),
      aggs_(std::move(aggs)),
      path_(Path::Var),
      spillTriggerGroups_(spillTriggerGroups),
      spillFraction_(spillFraction) {

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

    // Path selection.
    if (groupKeys_.size() == 1 && keyTypes_[0].id == TypeId::STRING) {
        path_ = Path::StringFastPath;
        nullGroupState_.resize(stateBytes_);
    } else {
        codec_ = std::make_unique<KeyCodec>(keyTypes_);
        path_  = codec_->isFixedWidth() ? Path::Fixed : Path::Var;
    }

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

int32_t HashAggregateOperator::internGlobalString(std::string_view s) {
    auto it = globalDict_.find(std::string(s));
    if (it != globalDict_.end()) return it->second;
    const int32_t code = static_cast<int32_t>(globalDictReverse_.size());
    globalDictReverse_.emplace_back(s);
    globalDict_.emplace(globalDictReverse_.back(), code);
    return code;
}

uint8_t* HashAggregateOperator::fastPathStateForCode(int32_t globalCode) {
    const size_t needed = (static_cast<size_t>(globalCode) + 1) * stateBytes_;
    if (codeStates_.size() < needed) {
        const size_t oldSlots = codeStates_.size() / stateBytes_;
        codeStates_.resize(needed);
        // Initialize newly-added slots.
        for (size_t s = oldSlots; s <= static_cast<size_t>(globalCode); ++s) {
            uint8_t* slab = codeStates_.data() + s * stateBytes_;
            for (size_t a = 0; a < aggs_.size(); ++a) {
                aggs_[a]->init(slab + aggOffsets_[a]);
            }
        }
    }
    return codeStates_.data() + static_cast<size_t>(globalCode) * stateBytes_;
}

// Serialize one group's state slab — including out-of-line data (strings,
// distinct sets) — into `rec` via each aggregate's serializeState.
void HashAggregateOperator::serializeSlab(const uint8_t*        slab,
                                          std::vector<uint8_t>& rec) const {
    rec.clear();
    for (size_t a = 0; a < aggs_.size(); ++a) {
        aggs_[a]->serializeState(slab + aggOffsets_[a], rec);
    }
}

// Rebuild a state slab from a spilled record into `scratch` (init'd per
// aggregate first), then fold it into `dst`.
void HashAggregateOperator::mergeSerializedInto(uint8_t*       dst,
                                                const uint8_t* data) {
    if (mergeScratch_.size() < stateBytes_) mergeScratch_.resize(stateBytes_);
    uint8_t* scratch = mergeScratch_.data();
    const uint8_t* p = data;
    for (size_t a = 0; a < aggs_.size(); ++a) {
        aggs_[a]->init(scratch + aggOffsets_[a]);
        p += aggs_[a]->deserializeState(scratch + aggOffsets_[a], p);
    }
    for (size_t a = 0; a < aggs_.size(); ++a) {
        aggs_[a]->merge(dst + aggOffsets_[a], scratch + aggOffsets_[a]);
    }
}

void HashAggregateOperator::spillStringFastPath() {
    if (globalDictReverse_.empty() && !nullGroupInit_) return;

    SpillFile file(stateBytes_);
    std::vector<uint8_t> rec;
    for (size_t c = 0; c < globalDictReverse_.size(); ++c) {
        const auto& s = globalDictReverse_[c];
        serializeSlab(codeStates_.data() + c * stateBytes_, rec);
        file.write(reinterpret_cast<const uint8_t*>(s.data()),
                   static_cast<int32_t>(s.size()),
                   rec.data(), rec.size());
    }
    if (nullGroupInit_) {
        serializeSlab(nullGroupState_.data(), rec);
        file.writeNullGroup(rec.data(), rec.size());
    }
    file.finishWriting();
    spillFiles_.push_back(std::move(file));

    // Reset in-memory state. Every group was spilled, so out-of-line
    // aggregate state (strings, distinct sets) can be released too.
    globalDict_.clear();
    globalDictReverse_.clear();
    codeStates_.clear();
    nullGroupInit_ = false;
    for (auto& a : aggs_) a->clearOutOfLineState();
}

void HashAggregateOperator::mergeSpillFiles() {
    if (spillFiles_.empty()) return;

    for (auto& file : spillFiles_) {
        auto reader = file.reader();
        while (reader.next()) {
            const uint8_t* srcState = reader.state();
            if (reader.isNullGroup()) {
                if (!nullGroupInit_) {
                    InitCtx ctx{&aggs_, &aggOffsets_};
                    g_initCtx = &ctx;
                    initSlab(nullGroupState_.data());
                    nullGroupInit_ = true;
                }
                mergeSerializedInto(nullGroupState_.data(), srcState);
            } else {
                std::string_view s(reinterpret_cast<const char*>(reader.key()),
                                   reader.keyLen());
                const int32_t g = internGlobalString(s);
                uint8_t* dstSlab = fastPathStateForCode(g);
                mergeSerializedInto(dstSlab, srcState);
            }
        }
    }
    spillFiles_.clear();
}

void HashAggregateOperator::spillFixed() {
    if (tableFixed_->numGroups() == 0) return;
    SpillFile file(stateBytes_);
    std::vector<uint8_t> rec;
    for (auto cur = tableFixed_->cursor(); cur.valid(); cur.next()) {
        serializeSlab(cur.state(), rec);
        file.write(cur.key(), static_cast<int32_t>(codec_->keyBytes()),
                   rec.data(), rec.size());
    }
    file.finishWriting();
    spillFiles_.push_back(std::move(file));
    tableFixed_ = std::make_unique<BytesHashTable>(codec_->keyBytes(), stateBytes_, mm_);
    for (auto& a : aggs_) a->clearOutOfLineState();
}

void HashAggregateOperator::spillVar() {
    if (tableVar_->numGroups() == 0) return;
    SpillFile file(stateBytes_);
    std::vector<uint8_t> rec;
    for (auto cur = tableVar_->cursor(); cur.valid(); cur.next()) {
        serializeSlab(cur.state(), rec);
        file.write(cur.key(), static_cast<int32_t>(cur.keyLen()),
                   rec.data(), rec.size());
    }
    file.finishWriting();
    spillFiles_.push_back(std::move(file));
    tableVar_ = std::make_unique<VarBytesHashTable>(stateBytes_, mm_);
    for (auto& a : aggs_) a->clearOutOfLineState();
}

void HashAggregateOperator::mergeSpillFilesFixed() {
    if (spillFiles_.empty()) return;
    InitCtx ctx{&aggs_, &aggOffsets_};
    g_initCtx = &ctx;
    for (auto& file : spillFiles_) {
        auto reader = file.reader();
        while (reader.next()) {
            uint8_t* dst = tableFixed_->findOrInsert(reader.key(), initSlab);
            mergeSerializedInto(dst, reader.state());
        }
    }
    g_initCtx = nullptr;
    spillFiles_.clear();
}

void HashAggregateOperator::mergeSpillFilesVar() {
    if (spillFiles_.empty()) return;
    InitCtx ctx{&aggs_, &aggOffsets_};
    g_initCtx = &ctx;
    for (auto& file : spillFiles_) {
        auto reader = file.reader();
        while (reader.next()) {
            uint8_t* dst = tableVar_->findOrInsert(reader.key(), reader.keyLen(), initSlab);
            mergeSerializedInto(dst, reader.state());
        }
    }
    g_initCtx = nullptr;
    spillFiles_.clear();
}

void HashAggregateOperator::build() {
    InitCtx ctx{&aggs_, &aggOffsets_};
    g_initCtx = &ctx;

    if (path_ == Path::Fixed) {
        tableFixed_ = std::make_unique<BytesHashTable>(
            codec_->keyBytes(), stateBytes_, mm_);
    } else if (path_ == Path::Var) {
        tableVar_ = std::make_unique<VarBytesHashTable>(stateBytes_, mm_);
    }

    std::vector<uint8_t>  keyBufFixed(path_ == Path::Fixed ? codec_->keyBytes() : 0);
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

        if (path_ == Path::StringFastPath) {
            const ColumnVector&    keyCol   = keyEvals[0];
            const ValidityBitmask& keyValid = keyCol.validity();

            // Two-pass: (1) intern all strings to global codes (which may
            // grow globalDict_ / codeStates_ and invalidate any pointers),
            // (2) resize codeStates_ once, (3) materialize per-row slot
            // pointers from the now-stable buffer.
            constexpr int32_t kNullCodeMarker = -1;
            std::vector<int32_t> rowCodes(pos.size(), kNullCodeMarker);

            if (keyCol.isDictEncoded()) {
                std::vector<int32_t> batchToGlobal(keyCol.dictSize());
                for (size_t b = 0; b < keyCol.dictSize(); ++b) {
                    const int32_t* doff = keyCol.dictOffsets().as<int32_t>();
                    const char*    dval = keyCol.dictValues().as<char>();
                    std::string_view s(dval + doff[b],
                                       static_cast<size_t>(doff[b + 1] - doff[b]));
                    batchToGlobal[b] = internGlobalString(s);
                }
                const int32_t* codes = keyCol.values().as<int32_t>();
                for (size_t k = 0; k < pos.size(); ++k) {
                    const size_t i = static_cast<size_t>(pos.at(k));
                    if (keyValid.isValid(i)) {
                        rowCodes[k] = batchToGlobal[static_cast<size_t>(codes[i])];
                    }
                }
            } else {
                for (size_t k = 0; k < pos.size(); ++k) {
                    const size_t i = static_cast<size_t>(pos.at(k));
                    if (keyValid.isValid(i)) {
                        rowCodes[k] = internGlobalString(keyCol.stringAt(i));
                    }
                }
            }

            // Now codeStates_ is stable: every new global code created above is
            // covered by a single resize-and-init pass.
            if (!globalDictReverse_.empty()) {
                fastPathStateForCode(static_cast<int32_t>(globalDictReverse_.size()) - 1);
            }

            for (size_t k = 0; k < pos.size(); ++k) {
                if (rowCodes[k] == kNullCodeMarker) {
                    if (!nullGroupInit_) {
                        initSlab(nullGroupState_.data());
                        nullGroupInit_ = true;
                    }
                    groupSlots.push_back(nullGroupState_.data());
                } else {
                    groupSlots.push_back(
                        codeStates_.data() +
                        static_cast<size_t>(rowCodes[k]) * stateBytes_);
                }
            }
        } else {
            // Existing fixed / var paths via KeyCodec + Bytes/VarBytesHashTable.
            for (size_t k = 0; k < pos.size(); ++k) {
                const auto i = static_cast<size_t>(pos.at(k));
                if (path_ == Path::Fixed) {
                    codec_->encode(keyPtrs, i, keyBufFixed.data());
                    groupSlots.push_back(tableFixed_->findOrInsert(keyBufFixed.data(), initSlab));
                } else {
                    codec_->encodeVar(keyPtrs, i, keyBufVar);
                    groupSlots.push_back(tableVar_->findOrInsert(
                        keyBufVar.data(), keyBufVar.size(), initSlab));
                }
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

        // Spill v1 trigger: group-count or memory-fraction.
        if (spillTriggerGroups_ > 0 || spillFraction_ > 0.0) {
            size_t currentGroups = 0;
            switch (path_) {
                case Path::StringFastPath: currentGroups = globalDictReverse_.size(); break;
                case Path::Fixed:          currentGroups = tableFixed_->numGroups();  break;
                case Path::Var:            currentGroups = tableVar_->numGroups();    break;
            }
            const bool overGroups = spillTriggerGroups_ > 0 &&
                                    currentGroups >= spillTriggerGroups_;
            const bool overMemory = spillFraction_ > 0.0 &&
                                    mm_->usedFraction() >= spillFraction_;
            if (overGroups || overMemory) {
                switch (path_) {
                    case Path::StringFastPath: spillStringFastPath(); break;
                    case Path::Fixed:          spillFixed();          break;
                    case Path::Var:            spillVar();            break;
                }
            }
        }
    }

    g_initCtx = nullptr;

    switch (path_) {
        case Path::StringFastPath: mergeSpillFiles();     break;
        case Path::Fixed:          mergeSpillFilesFixed(); break;
        case Path::Var:            mergeSpillFilesVar();   break;
    }

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

ColumnBatch HashAggregateOperator::emitStringFastPath() {
    const size_t numCodes  = globalDictReverse_.size();
    const size_t total     = numCodes + (nullGroupInit_ ? 1 : 0);

    StringBuilder keyBuilder(mm_);
    std::vector<const uint8_t*> states;
    states.reserve(total);

    if (nullGroupInit_) {
        keyBuilder.appendNull();
        states.push_back(nullGroupState_.data());
    }
    for (size_t c = 0; c < numCodes; ++c) {
        keyBuilder.append(globalDictReverse_[c]);
        states.push_back(codeStates_.data() + c * stateBytes_);
    }

    std::vector<ColumnVector> cols;
    cols.reserve(1 + aggs_.size());
    cols.push_back(keyBuilder.build());
    for (size_t a = 0; a < aggs_.size(); ++a) {
        std::vector<const uint8_t*> slices;
        slices.reserve(states.size());
        for (const auto* s : states) slices.push_back(s + aggOffsets_[a]);
        cols.push_back(aggs_[a]->finalize(slices, mm_));
    }
    return ColumnBatch(std::move(cols), PositionList::identity(total));
}

ColumnBatch HashAggregateOperator::emit() {
    switch (path_) {
        case Path::StringFastPath: return emitStringFastPath();
        case Path::Fixed:          return emitFixed();
        case Path::Var:            return emitVar();
    }
    raise(ErrorCode::Internal, "HashAggregate: unknown emit path");
}

std::optional<ColumnBatch> HashAggregateOperator::getNext() {
    if (!built_) build();
    if (emitted_) return std::nullopt;
    emitted_ = true;
    return emit();
}

} // namespace quantum
