# Testing and CI quickstart

Use the markers and make targets below to keep local runs aligned with CI.

## Quick paths
- **Default (CI/local fast path):** `make test` → runs pytest with `-m "not slow and not scientific"` (unit + integration + plotting smoke)
- **Focused subsets:** `make test-quick` (unit+integration), `make test-plot` (plotting-only)

## Full / slow suites
- **Slow / scheduled:** `make test-slow` or `pytest -q -m "slow"` (benchmarks and heavy runners). These are excluded from CI by default.
- **All tests including slow:** `pytest -q -m "unit or integration or scientific or plotting or slow"`.

## Determinism and reproducibility
- RNG is seeded via `--seed` (default from `$PYTEST_SEED` or 1337). Override with `pytest --seed 20240101`.
- Matplotlib backend is forced to `Agg` to avoid GUI dependencies.
- CI sets `BIORSP_TEST_MODE=quick` automatically through the `fast_mode` fixture.

## Reproducing CI locally
- Mirror CI selection: `pytest -q -m "not slow"` from repo root.
- Run a single category: e.g., `pytest -q tests/scientific -m scientific`.
- For CLI/runner tests, ensure dependencies are installed (`pip install -e ".[dev]"`).
