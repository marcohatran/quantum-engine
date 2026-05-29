# Benchmark

Build and run performance benchmarks using Google Benchmark.

## Usage

```
/benchmark [filter] [--compare baseline.json]
```

- `filter`: run only benchmarks matching this substring (e.g., `HashJoin`, `Aggregation`)
- `--compare`: compare results against a previously saved baseline JSON

## Steps

1. Build a release benchmark binary:
   ```bash
   cmake -B build-release -S . -GNinja \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_BENCHMARKS=ON
   cmake --build build-release --parallel $(nproc) --target benchmarks
   ```
2. Run benchmarks and save results:
   ```bash
   ./build-release/benchmarks/benchmark_runner \
     --benchmark_filter="<filter>" \
     --benchmark_out=benchmark_results.json \
     --benchmark_out_format=json
   ```
3. If `--compare` is given, diff the JSON outputs and highlight regressions > 5%.
4. Print a summary table: benchmark name, mean latency, throughput (rows/sec or GB/sec).

## Notes

- Always run on a release build with CPU frequency scaling disabled for stable results:
  `sudo cpupower frequency-set --governor performance` (Linux)
- Isolate benchmark runs from other workloads — close background processes.
- Store baseline JSONs as `benchmarks/baselines/<name>.json` and commit them.
