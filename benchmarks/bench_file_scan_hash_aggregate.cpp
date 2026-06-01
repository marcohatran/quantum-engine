#include <benchmark/benchmark.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include "memory/MemoryManager.hpp"
#include "plan/Plan.hpp"
#include "plan/PlanBuilder.hpp"

using namespace quantum;
using namespace quantum::plan;

namespace {

// Write a Parquet file with `nRows` rows of (k:int64, v:int64) where k cycles
// 0..cardinality-1 (so the HashAgg ends up with `cardinality` groups). Each
// row group is sized at 16 KiB to keep the in-memory Arrow Table footprint
// modest while still touching the disk/page-cache codepaths.
std::string writeKvParquet(size_t nRows, size_t cardinality, const std::string& hint) {
    auto dir   = std::filesystem::temp_directory_path();
    auto path  = (dir / (hint + std::to_string(std::random_device{}()) + ".parquet")).string();

    arrow::Int64Builder kb, vb;
    for (size_t i = 0; i < nRows; ++i) {
        (void)kb.Append(static_cast<int64_t>(i % cardinality));
        (void)vb.Append(static_cast<int64_t>(i));
    }
    std::shared_ptr<arrow::Array> ka, va;
    (void)kb.Finish(&ka);
    (void)vb.Finish(&va);
    auto schema = arrow::schema({
        arrow::field("k", arrow::int64()),
        arrow::field("v", arrow::int64())});
    auto table = arrow::Table::Make(schema, {ka, va});

    auto out_r = arrow::io::FileOutputStream::Open(path);
    auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                              out_r.ValueOrDie());
    (void)status;
    (void)out_r.ValueOrDie()->Close();
    return path;
}

} // namespace

// End-to-end Phase 5 path: FileScan(parquet) -> HashAggregate(sum(v) by k).
// Includes the cost of opening the file each iteration (matches the
// per-query reality), excludes the fixture write.
static void BM_FileScan_HashAggregate(benchmark::State& state) {
    const size_t totalRows   = 1U << 20;             // 1M rows
    const size_t cardinality = static_cast<size_t>(state.range(0));

    auto path = writeKvParquet(totalRows, cardinality, "bench_filescan_");

    MemoryManager mm(256ULL * 1024 * 1024);

    for (auto _ : state) {
        // Build the plan inside the timed region — fresh plan/operator
        // construction is part of a real per-query path.
        FileScanNode fs;
        fs.paths  = {path};
        fs.format = "parquet";

        auto agg = std::make_unique<HashAggregateNode>();
        agg->child = PlanNode{std::move(fs)};
        agg->group_keys.push_back(ColumnRefExpr{0});

        AggregateSpec sumSpec;
        sumSpec.kind      = AggregateKind::Sum;
        sumSpec.inputType = Int64;
        sumSpec.input     = ColumnRefExpr{1};
        agg->aggregates.push_back(std::move(sumSpec));
        agg->output_names = {"k", "sum_v"};

        PlanNode plan{std::move(agg)};

        PlanBuilder pb(&mm);
        SourceCatalog catalog;
        auto op = pb.build(plan, catalog);

        op->open();
        while (auto b = op->getNext()) benchmark::DoNotOptimize(b);
        op->close();
    }
    const int64_t bytesPerIter =
        static_cast<int64_t>(totalRows) * 2 * static_cast<int64_t>(sizeof(int64_t));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(totalRows));
    state.SetBytesProcessed(state.iterations() * bytesPerIter);

    std::filesystem::remove(path);
}
BENCHMARK(BM_FileScan_HashAggregate)
    ->Arg(16)        // low cardinality
    ->Arg(1024)      // mid
    ->Arg(65536);    // high (likely each group has ~16 rows)
