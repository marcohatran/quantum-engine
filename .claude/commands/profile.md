# Profile

Profile CPU hotspots using Linux perf and generate a flame graph.

## Usage

```
/profile <binary> [args] [--duration 30]
```

- `binary`: path to the executable to profile (e.g., `./build-release/benchmarks/benchmark_runner`)
- `args`: arguments passed to the binary
- `--duration`: recording duration in seconds (default: 30)

## Steps

1. Ensure the binary is built with `RelWithDebInfo` (debug symbols, optimizations on):
   ```bash
   cmake -B build-reldbg -S . -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build-reldbg --parallel $(nproc)
   ```
2. Record with perf:
   ```bash
   perf record -g -F 999 --call-graph dwarf -o perf.data -- <binary> <args>
   ```
3. Generate a text report:
   ```bash
   perf report -i perf.data --stdio | head -60
   ```
4. Generate a flame graph (if `flamegraph.pl` is available):
   ```bash
   perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
   ```
5. Open `flamegraph.svg` and report the top 5 hotspot functions.

## Notes

- Requires `linux-tools-$(uname -r)` or equivalent perf package.
- Flame graph scripts: https://github.com/brendangregg/FlameGraph
- On macOS, use Instruments (`xctrace`) instead of perf.
- Increase `perf_event_paranoid` if permission denied: `echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid`
