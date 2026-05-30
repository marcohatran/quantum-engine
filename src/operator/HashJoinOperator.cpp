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

// Concatenates a vector of dense ColumnBatches (identity position lists,
// matching schemas) into a single dense ColumnBatch. Allocates fresh buffers
// owned by `mm`. Returns an empty batch if `batches` is empty.
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

// Gather rows from `src` at `idx` into a fresh ColumnVector owned by `mm`.
ColumnVector gather(const ColumnVector&         src,
                    const std::vector<int32_t>& idx,
                    MemoryManager*              mm) {
    ColumnVector out = ColumnVector::make(src.type(), idx.size(), mm);
    out.validity().setAllValid();

    const size_t width = static_cast<size_t>(src.type().fixedWidth());

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
    (void)width;

    for (size_t k = 0; k < idx.size(); ++k) {
        out.validity().setValid(k, src.validity().isValid(static_cast<size_t>(idx[k])));
    }
    return out;
}

} // namespace

HashJoinOperator::HashJoinOperator(MemoryManager*           mm,
                                   OperatorPtr              buildSide,
                                   OperatorPtr              probeSide,
                                   ExpressionPtr            buildKey,
                                   ExpressionPtr            probeKey,
                                   std::vector<std::string> outputNames)
    : Operator(mm),
      build_(std::move(buildSide)),
      probe_(std::move(probeSide)),
      buildKey_(std::move(buildKey)),
      probeKey_(std::move(probeKey)) {
    const Schema& bs = build_->outputSchema();
    const Schema& ps = probe_->outputSchema();

    if (outputNames.size() != bs.numFields() + ps.numFields()) {
        raise(ErrorCode::InvalidArgument,
              "HashJoin: {} output names, expected {} (build:{} + probe:{})",
              outputNames.size(), bs.numFields() + ps.numFields(),
              bs.numFields(), ps.numFields());
    }

    if (buildKey_->resultType(bs).id != TypeId::INT64 ||
        probeKey_->resultType(ps).id != TypeId::INT64) {
        raise(ErrorCode::NotImplemented,
              "HashJoin: only INT64 keys supported in Phase 2 slice");
    }

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
    // 1. Drain the build side, densifying each batch.
    std::vector<ColumnBatch> dense;
    while (auto b = build_->getNext()) {
        if (b->activeRows() == 0) continue;
        dense.push_back(b->selectActive(mm_));
    }
    rowStore_ = concatDense(std::move(dense), mm_);

    // 2. Index the row store by build key.
    index_ = std::make_unique<HashTable>(sizeof(int32_t), mm_);
    next_.assign(rowStore_->numRows(), kEndOfChain);

    if (rowStore_->numRows() == 0) {
        built_ = true;
        return;
    }

    // Evaluate the build key over the (already dense) row store.
    ColumnVector           keyCol  = buildKey_->eval(*rowStore_, mm_);
    const int64_t*         keys    = keyCol.typed<int64_t>();
    const ValidityBitmask& valid   = keyCol.validity();

    for (size_t i = 0; i < rowStore_->numRows(); ++i) {
        if (!valid.isValid(i)) continue;  // null keys never match in inner join
        uint8_t* slot = index_->findOrInsert(keys[i], initEndOfChain);
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
        // Forward an empty result; caller may keep draining for the next batch.
        return ColumnBatch(std::vector<ColumnVector>{}, PositionList::identity(0));
    }
    if (rowStore_->numRows() == 0) {
        // Empty build → no matches; emit an empty batch but keep draining.
        return ColumnBatch(std::vector<ColumnVector>{}, PositionList::identity(0));
    }

    ColumnVector           keyCol = probeKey_->eval(*in, mm_);
    const int64_t*         keys   = keyCol.typed<int64_t>();
    const ValidityBitmask& valid  = keyCol.validity();
    const PositionList&    pos    = in->positions();

    std::vector<int32_t> probeIdx;
    std::vector<int32_t> buildIdx;

    for (size_t k = 0; k < pos.size(); ++k) {
        const auto i = static_cast<size_t>(pos.at(k));
        if (!valid.isValid(i)) continue;  // null keys never match

        const uint8_t* slot = index_->find(keys[i]);
        if (slot == nullptr) continue;  // no build row had this key
        int32_t cur;
        std::memcpy(&cur, slot, sizeof(cur));
        while (cur != kEndOfChain) {
            probeIdx.push_back(static_cast<int32_t>(i));
            buildIdx.push_back(cur);
            cur = next_[static_cast<size_t>(cur)];
        }
    }

    // Materialize output: build columns first, then probe columns.
    std::vector<ColumnVector> outCols;
    outCols.reserve(rowStore_->numColumns() + in->numColumns());
    for (size_t c = 0; c < rowStore_->numColumns(); ++c) {
        outCols.push_back(gather(rowStore_->column(c), buildIdx, mm_));
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
