# Format

Auto-format all C++ source files using clang-format.

## Usage

```
/format [path]
```

- No path: formats all staged files (pre-commit style)
- With path: formats the specified file or directory recursively

## Steps

1. Find files to format:
   ```bash
   # Staged files only
   git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|cc|cxx|h|hpp)$'

   # Or all files under a path
   find <path> -name '*.cpp' -o -name '*.h' -o -name '*.hpp' | sort
   ```
2. Apply clang-format in-place:
   ```bash
   clang-format -i --style=file <files>
   ```
3. Report which files were modified.

## Notes

- Style is defined by `.clang-format` at the repo root. If absent, default to Google style (`--style=Google`) until one is created.
- Minimum recommended clang-format version: 17.
- To check without modifying (CI mode): `clang-format --dry-run --Werror <files>`
