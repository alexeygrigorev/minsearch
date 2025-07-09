.PHONY: setup test clean publish-test publish clean-build install install-dev shell list notebook build check

# Development setup
setup:
	uv sync --extra dev

# Run tests
test:
	uv run pytest

# Build package
build:
	uv run python -m build

# Check built packages
check:
	uv run twine check dist/*

# Publish to test PyPI
publish-test:
	uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Publish to PyPI
publish:
	uv run twine upload dist/*

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