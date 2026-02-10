# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q
```

## Lint and format

```bash
ruff check .
ruff format .
```

## Canonical smoke commands

```bash
biorsp-smoke-rsp --h5ad /path/to/input.h5ad --outdir .
biorsp-smoke-moran --h5ad /path/to/input.h5ad --outdir .
biorsp-smoke-perm --h5ad /path/to/input.h5ad --outdir . --n-perm 100
```

## Pipeline entrypoints

```bash
python scripts/prereg_pipeline.py --config configs/biorsp_prereg.json
python scripts/genomewide_pipeline.py --config configs/biorsp_genomewide.json
```
