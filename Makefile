PYTHON ?= python
H5AD ?= data/processed/HT_pca_umap.h5ad
OUT ?= outputs/heart_case_study/local_run

.PHONY: install test lint format case-study

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

case-study:
	PYTHONPATH=. $(PYTHON) analysis/heart_case_study/run.py --h5ad $(H5AD) --out $(OUT) --donor_key hubmap_id --cluster_key azimuth_id --celltype_key azimuth_label --do_hierarchy true
