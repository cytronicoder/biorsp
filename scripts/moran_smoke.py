"""Legacy wrapper for BioRSP Moran smoke CLI."""

from __future__ import annotations

import warnings

from biorsp.cli import smoke_moran_main

if __name__ == "__main__":
    warnings.warn(
        "scripts/moran_smoke.py is deprecated; use the canonical 'biorsp-smoke-moran' entrypoint.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(smoke_moran_main())
