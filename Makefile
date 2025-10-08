.PHONY: install install-dev test clean lint format

# Install package in editable mode
install:
	pip install -e .

# Install package with development dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

# Lint code
lint:
	flake8 src/ tests/

# Format code
format:
	black src/ tests/ examples/

# Build distribution
build:
	python -m build

# Run example
example:
	python examples/kpmp.py

# Help
help:
	@echo "Available commands:"
	@echo "  make install       - Install package in editable mode"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make lint          - Lint code with flake8"
	@echo "  make format        - Format code with black"
	@echo "  make build         - Build distribution packages"
	@echo "  make example       - Run example script"
