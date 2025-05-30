.PHONY: setup test clean publish-test publish clean-build

# Development setup
setup:
	pipenv install --dev

# Run tests
test:
	pipenv run pytest

# Start Jupyter notebook
notebook:
	pipenv run jupyter notebook

# Build package
build:
	pipenv run python -m build

# Check built packages
check:
	pipenv run twine check dist/*

# Publish to test PyPI
publish-test:
	pipenv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Publish to PyPI
publish:
	pipenv run twine upload dist/*

# Clean build artifacts
clean:
	rm -rf build/ dist/ minsearch.egg-info/ .pytest_cache/ __pycache__/ .coverage htmlcov/

# Clean and rebuild
clean-build: clean build 