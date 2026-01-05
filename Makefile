.PHONY: format lint test clean install dev-install help

help:
	@echo "Available targets:"
	@echo "  make format      - Format code with black and ruff"
	@echo "  make lint        - Lint code with ruff"
	@echo "  make test        - Run pytest tests"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make install     - Install package"
	@echo "  make dev-install - Install package in editable mode with dev dependencies"

format:
	@echo "Running black..."
	black biorsp/ tests/ examples/ scripts/ benchmarks/
	@echo "Running ruff --fix..."
	ruff check --fix biorsp/ tests/ examples/ scripts/ benchmarks/
	@echo "Running isort..."
	isort biorsp/ tests/ examples/ scripts/ benchmarks/

lint:
	@echo "Running ruff..."
	ruff check biorsp/ tests/ examples/ scripts/ benchmarks/
	@echo "Running black --check..."
	black --check biorsp/ tests/ examples/ scripts/ benchmarks/
	@echo "Running isort --check..."
	isort --check-only biorsp/ tests/ examples/ scripts/ benchmarks/

test:
	@echo "Running pytest..."
	pytest tests/ -v

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

install:
	pip install .

dev-install:
	pip install -e ".[dev]"
