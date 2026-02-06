# Release

Release the current changes to PyPI and GitHub.

## Steps

1. **Check current version** on PyPI:
   ```bash
   uv run pip index versions minsearch
   ```

2. **Bump version** (e.g., 0.0.7 -> 0.0.8):
   ```bash
   echo "__version__ = '0.0.8'" > minsearch/__version__.py
   ```

3. **Commit changes** (if not already committed):
   ```bash
   git add -A
   git commit -m "Describe changes"
   ```

4. **Commit version bump**:
   ```bash
   git add minsearch/__version__.py
   git commit -m "Bump version to 0.0.8"
   ```

5. **Build package**:
   ```bash
   uv run python -m build
   ```

6. **Check distribution**:
   ```bash
   uv run twine check dist/*
   ```

7. **Publish to PyPI**:
   ```bash
   uv run twine upload dist/*
   ```

8. **Push to GitHub**:
   ```bash
   git push origin main
   ```

9. **Create GitHub release** with binaries and release notes:
   ```bash
   gh release create 0.0.8 dist/* --title "0.0.8" --notes "Release notes here..."
   ```

## Notes

- Make sure `twine` is installed: `uv pip install twine`
- The version should be bumped based on [semantic versioning](https://semver.org/):
  - **Patch** (0.0.x): Bug fixes, small improvements
  - **Minor** (0.x.0): New features, backward compatible
  - **Major** (x.0.0): Breaking changes
