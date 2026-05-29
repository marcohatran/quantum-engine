# Build

Configure and build the project using CMake and vcpkg.

## Usage

```
/build [preset]
```

Preset options: `debug` (default), `release`, `asan`, `tsan`, `ubsan`

## Steps

1. Detect vcpkg toolchain file at `vcpkg/scripts/buildsystems/vcpkg.cmake` (or `$VCPKG_ROOT`)
2. Run CMake configure:
   ```bash
   cmake -B build-$PRESET -S . \
     -DCMAKE_BUILD_TYPE=<Debug|Release> \
     -DCMAKE_TOOLCHAIN_FILE=<vcpkg_toolchain> \
     -GNinja
   ```
3. Build with all available cores:
   ```bash
   cmake --build build-$PRESET --parallel $(nproc)
   ```
4. Report build output directory and any warnings.

## Notes

- Use `ccache` automatically if available in PATH (speeds up incremental builds).
- Release builds use `-O3 -DNDEBUG`; Debug builds include full debug symbols.
- Build directories follow the `build-<preset>/` naming convention (all gitignored).
