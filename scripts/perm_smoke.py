"""Legacy wrapper for BioRSP permutation smoke CLI."""

from __future__ import annotations

import warnings

from biorsp.cli import smoke_perm_main

if __name__ == "__main__":
    warnings.warn(
        "scripts/perm_smoke.py is deprecated; use the canonical 'biorsp-smoke-perm' entrypoint.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(smoke_perm_main())
