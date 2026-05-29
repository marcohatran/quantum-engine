# Test

Run the test suite using CTest.

## Usage

```
/test [filter]
```

- No filter: runs all tests in parallel
- With filter: runs only tests matching the pattern (substring or regex)

## Steps

1. Ensure the project has been built (run `/build debug` if `build-debug/` doesn't exist).
2. Run tests:
   ```bash
   # All tests
   ctest --test-dir build-debug --output-on-failure -j$(nproc)

   # Single test or filter
   ctest --test-dir build-debug --output-on-failure -R "<filter>"
   ```
3. Report pass/fail counts and print full output for any failures.

## Notes

- Always test against a `debug` build to get meaningful stack traces on failure.
- To run a specific test binary directly: `./build-debug/<path/to/test_binary>`.
- Use `ctest --test-dir build-debug -N` to list all available test names without running them.
