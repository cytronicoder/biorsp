"""Legacy wrapper for BioRSP RSP smoke CLI."""

from __future__ import annotations

import warnings

from biorsp.cli import smoke_rsp_main

if __name__ == "__main__":
    warnings.warn(
        "scripts/rsp_smoke.py is deprecated; use the canonical 'biorsp-smoke-rsp' entrypoint.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(smoke_rsp_main())
