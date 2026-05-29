# Sanitize

Build and run the test suite under memory/thread/UB sanitizers to catch correctness issues.

## Usage

```
/sanitize [asan|tsan|ubsan]
```

Default: `asan` (AddressSanitizer + LeakSanitizer)

## Steps

1. Configure a sanitizer build:
   ```bash
   # AddressSanitizer
   cmake -B build-asan -S . -GNinja \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_CXX_FLAGS="-fsanitize=address,leak -fno-omit-frame-pointer" \
     -DCMAKE_C_FLAGS="-fsanitize=address,leak -fno-omit-frame-pointer"

   # ThreadSanitizer
   cmake -B build-tsan -S . -GNinja \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer"

   # UndefinedBehaviorSanitizer
   cmake -B build-ubsan -S . -GNinja \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer"
   ```
2. Build: `cmake --build build-<sanitizer> --parallel $(nproc)`
3. Run tests with sanitizer options:
   ```bash
   ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
   ctest --test-dir build-asan --output-on-failure -j$(nproc)
   ```
4. Report any sanitizer violations with full stack traces.

## Notes

- ASan slows execution ~2x; TSan ~5-15x — don't use for benchmarks.
- ASan and TSan are mutually exclusive; run separately.
- UBSan can be combined with ASan: `-fsanitize=address,undefined`.
