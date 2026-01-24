# Testing

Use the Makefile targets to align local runs with CI defaults.

## Quick paths

- `make test`: runs `pytest -q -m "not slow and not scientific"`.
- `make test-quick`: unit + integration subset.
- `make test-plot`: plotting-focused subset.

## Slow suites

- `make test-slow` or `pytest -q -m "slow"` runs slow-marked tests only.

## Reproducing CI locally

```bash
pytest -q -m "not slow" tests/
```

## Determinism

- Provide explicit seeds in tests when available.
- Use `MPLBACKEND=Agg` to avoid GUI dependencies when running plotting tests.
