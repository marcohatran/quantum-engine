#include "operator/HashAggregateOperator.hpp"

#include <cstring>

#include "common/Status.hpp"
#include "memory/MemoryManager.hpp"

namespace quantum {

namespace {

// Init the full per-group state slab by initializing each aggregate's slice.
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
        ExpressionPtr                     groupKey,
        std::vector<AggregateFunctionPtr> aggs,
        std::vector<std::string>          outputNames)
    : Operator(mm),
      child_(std::move(child)),
      groupKey_(std::move(groupKey)),
      aggs_(std::move(aggs)) {
    if (outputNames.size() != aggs_.size() + 1) {
        raise(ErrorCode::InvalidArgument,
              "HashAggregate: {} output names, expected {} (key + {} aggs)",
              outputNames.size(), aggs_.size() + 1, aggs_.size());
    }

    // State layout: pack aggregate slices back-to-back.
    aggOffsets_.reserve(aggs_.size());
    for (const auto& a : aggs_) {
        aggOffsets_.push_back(stateBytes_);
        stateBytes_ += a->stateSize();
    }
    if (stateBytes_ == 0) stateBytes_ = 1;  // HashTable disallows zero-byte slots

    // Output schema: group key column + one column per aggregate.
    const Schema& inSchema = child_->outputSchema();
    DataType keyType = groupKey_->resultType(inSchema);
    if (keyType.id != TypeId::INT64) {
        raise(ErrorCode::NotImplemented,
              "HashAggregate: only INT64 grouping keys supported in Phase 2 slice (got {})",
              keyType.toString());
    }
    std::vector<Field> fields;
    fields.reserve(outputNames.size());
    fields.push_back(Field{std::move(outputNames[0]), keyType});
    for (size_t i = 0; i < aggs_.size(); ++i) {
        fields.push_back(Field{std::move(outputNames[i + 1]), aggs_[i]->outputType()});
    }
    outSchema_ = Schema(std::move(fields));
}

void HashAggregateOperator::open()  { child_->open(); }
void HashAggregateOperator::close() { child_->close(); }

void HashAggregateOperator::build() {
    table_ = std::make_unique<HashTable>(stateBytes_, mm_);

    InitCtx ctx{&aggs_, &aggOffsets_};
    g_initCtx = &ctx;

    std::vector<uint8_t*> groupSlots;  // one entry per active row in the batch

    while (auto in = child_->getNext()) {
        const PositionList& pos = in->positions();
        if (pos.size() == 0) continue;

        // 1. Evaluate the group key column.
        ColumnVector keyCol = groupKey_->eval(*in, mm_);
        const int64_t*         keys     = keyCol.typed<int64_t>();
        const ValidityBitmask& keyValid = keyCol.validity();

        // 2. Insert / look up the per-row group slot.
        groupSlots.clear();
        groupSlots.reserve(pos.size());
        for (size_t k = 0; k < pos.size(); ++k) {
            const auto i = static_cast<size_t>(pos.at(k));
            if (!keyValid.isValid(i)) {
                groupSlots.push_back(table_->findOrInsertNull(initSlab));
            } else {
                groupSlots.push_back(table_->findOrInsert(keys[i], initSlab));
            }
        }

        // 3. For each aggregate, evaluate its (optional) input and update.
        // groupSlots holds pointers to the slab; offset each by the agg's slice.
        for (size_t a = 0; a < aggs_.size(); ++a) {
            const auto& agg = aggs_[a];

            std::optional<ColumnVector> inputCol;
            if (const Expression* e = agg->inputExpr()) {
                inputCol = e->eval(*in, mm_);
            }

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

ColumnBatch HashAggregateOperator::emit() {
    const size_t numNonNull = table_->numGroups();
    const size_t total      = numNonNull + (table_->hasNullGroup() ? 1 : 0);

    // 1. Build the group-key column.
    ColumnVector keyCol = ColumnVector::make(Int64, total, mm_);
    keyCol.validity().setAllValid();
    int64_t* keyData = keyCol.typed<int64_t>();

    // 2. Collect state pointers in the same order we'll write the keys.
    std::vector<const uint8_t*> states;
    states.reserve(total);

    size_t row = 0;
    if (table_->hasNullGroup()) {
        keyData[row] = 0;
        keyCol.validity().setValid(row, false);
        states.push_back(table_->nullGroupState());
        ++row;
    }
    for (auto cur = table_->cursor(); cur.valid(); cur.next()) {
        keyData[row] = cur.key();
        states.push_back(cur.state());
        ++row;
    }

    // 3. Materialize each aggregate's output column from its slice.
    std::vector<ColumnVector> cols;
    cols.reserve(1 + aggs_.size());
    cols.push_back(std::move(keyCol));

    for (size_t a = 0; a < aggs_.size(); ++a) {
        std::vector<const uint8_t*> slices;
        slices.reserve(states.size());
        for (const auto* s : states) slices.push_back(s + aggOffsets_[a]);
        cols.push_back(aggs_[a]->finalize(slices, mm_));
    }

    return ColumnBatch(std::move(cols), PositionList::identity(total));
}

std::optional<ColumnBatch> HashAggregateOperator::getNext() {
    if (!built_) build();
    if (emitted_) return std::nullopt;
    emitted_ = true;
    return emit();
}

} // namespace quantum
