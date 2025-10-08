# BioRSP Installation & Testing Guide

## Quick Setup

### 1. Install the package in development mode

```bash
# Make sure you're in the biorsp_v3 directory
cd /Users/cytronicoder/Documents/GitHub/biorsp_v3

# Install in editable mode (recommended for development)
pip install -e .
```

This installs the package so that changes to the source code are immediately reflected without reinstalling.

### 2. Install with development dependencies

```bash
# Install with testing and development tools
pip install -e ".[dev]"
```

This includes pytest, coverage tools, black (formatter), and flake8 (linter).

### 3. Run tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Or use the Makefile
make test
make test-cov
```

## Using Make Commands

The Makefile provides convenient shortcuts:

```bash
# Install package
make install

# Install with dev dependencies
make install-dev

# Run tests
make test

# Run tests with HTML coverage report
make test-cov

# Format code with black
make format

# Lint code with flake8
make lint

# Clean build artifacts
make clean

# Run the example
make example

# See all available commands
make help
```

## Project Structure

```
biorsp_v3/
├── src/                    # Main package code
│   ├── __init__.py
│   ├── radar_scan.py
│   ├── preprocessing.py
│   ├── stats.py
│   ├── utils.py
│   └── null_models.py
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_radar_scan.py
│   ├── test_preprocessing.py
│   └── test_utils.py
├── examples/               # Example scripts
│   └── kpmp.py
├── figures/                # Figure generation scripts
├── pyproject.toml         # Package configuration
├── setup.py               # Setup script
├── Makefile               # Convenient commands
└── README.md              # Documentation
```

## Verifying Installation

After installation, verify it works:

```python
# In Python
import src
from src.radar_scan import ScanParams, RadarScanner

# Create params
params = ScanParams(B=180)
print(f"Scanner configured with {params.B} angles")
```

Or run the example:

```bash
python examples/kpmp.py
```

## Development Workflow

1. **Make changes** to code in `src/`
2. **Run tests** to ensure nothing broke: `make test`
3. **Format code**: `make format`
4. **Check linting**: `make lint`
5. **Commit changes**

## Troubleshooting

### Import errors

If you get import errors, make sure you installed in editable mode:

```bash
pip install -e .
```

### Missing dependencies

Install all dependencies:

```bash
pip install -e ".[dev]"
```

### Test failures

Some tests may fail if the actual function names differ. Update tests to match your implementation.

## Next Steps

1. Update the test files to match your actual function signatures
2. Add more comprehensive tests
3. Update author info in `pyproject.toml`
4. Add a LICENSE file if needed
5. Set up CI/CD (GitHub Actions) for automated testing
