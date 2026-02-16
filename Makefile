.PHONY: setup test clean publish-build publish-test publish publish-clean clean-build install install-dev shell list notebook build check

# Development setup
setup:
	uv sync --extra dev

# Run tests
test:
	uv run pytest

# Build package
build:
	uv run hatch build

# Check built packages
check:
	uv run hatch check

# Build for publishing (clean + build + check)
publish-build: clean build check

# Publish to test PyPI
publish-test:
	uv run hatch publish --repo test

# Publish to PyPI
publish:
	uv run hatch publish

# Clean build artifacts after publishing
publish-clean:
	rm -rf dist/

# Clean build artifacts
clean:
	rm -rf build/ dist/ minsearch.egg-info/ .pytest_cache/ __pycache__/ .coverage htmlcov/

# Clean and rebuild
clean-build: clean build

# Install dependencies (production only)
install:
	uv sync

# Install dependencies (with dev extras)
install-dev:
	uv sync --extra dev

# Run shell with uv environment
shell:
	uv shell

# List installed packages
list:
	uv pip list 