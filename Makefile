PYTHON ?= python
H5AD ?= adata_embed_graph.h5ad

.PHONY: install test lint format smoke

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

smoke:
	biorsp-smoke-rsp --h5ad $(H5AD) --outdir .
	biorsp-smoke-moran --h5ad $(H5AD) --outdir .
	biorsp-smoke-perm --h5ad $(H5AD) --outdir . --n-perm 100
