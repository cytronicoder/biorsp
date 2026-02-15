# Developer Guide

## Scope

This repository is intentionally scoped to a heart case study workflow.

## Key Modules

- `biorsp/core/*`: compute, geometry, feature resolution, types
- `biorsp/stats/*`: permutation, Moran's I, scoring utilities
- `biorsp/plotting/*`: QC plots, RSP plots, pair panels, style config
- `biorsp/pipeline/io.py`: JSON/file helpers
- `biorsp/pipeline/hierarchy.py`: `run_case_study` and scope orchestration
- `analysis/heart_case_study/run.py`: canonical entrypoint
- `experiments/heart_smoketest/run_heart_smoketest.py`: CI sanity wrapper

## Public API

Top-level `biorsp` exports:

- `compute_rsp`
- `plot_rsp`
- `plot_umap_rsp_pair`
- `compute_vantage_point`
- `resolve_feature_index`
- `run_case_study`

## Adding Another Case Study

1. Add a new analysis entrypoint under `analysis/<case_name>/run.py`.
2. Keep scientific framing explicit in emitted metadata and runlog.
3. Define fixed marker/controls panel and fallback policy.
4. Keep deterministic seeds and stable output schemas.
5. Add focused tests for marker resolution, stratified inference fallback, and API contract.

## Tests

Run:

```bash
pytest -q
```

Core tests include:

- `tests/test_theta_convention.py`
- `tests/test_feature_resolution.py`
- `tests/test_compute_plot_contract.py`
- case-study behavior tests (`tests/test_case_study_pipeline.py`)
