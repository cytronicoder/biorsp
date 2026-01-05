# Installation Guide

## Prerequisites
- Python 3.9 or higher
- NumPy, SciPy, Pandas, Matplotlib

## Standard Installation
```bash
pip install biorsp
```

## From Source
```bash
git clone https://github.com/cytronicoder/biorsp.git
cd biorsp
pip install .
```

## Development Installation
To install with testing and linting tools:
```bash
pip install -e ".[dev]"
```

## Common Issues
- **Missing Dependencies**: Ensure `numpy` and `scipy` are installed before installing `biorsp` if you encounter build errors.
- **Conda Users**: We recommend creating a clean environment:
  ```bash
  conda create -n biorsp python=3.10
  conda activate biorsp
  pip install biorsp
  ```
