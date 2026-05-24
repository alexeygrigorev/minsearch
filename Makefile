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

# Release: tag the current version and push to trigger CI publish.
# CI workflow: .github/workflows/publish.yml (on tag push v*)
release:
	@VERSION=$$(grep -E "^__version__" minsearch/__version__.py | sed -E "s/.*['\"]([^'\"]+)['\"].*/\1/"); \
	echo "Releasing v$$VERSION"; \
	git tag "v$$VERSION"; \
	git push origin "v$$VERSION"

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