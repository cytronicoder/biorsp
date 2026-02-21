"""Shared tiny deterministic defaults for simulation test-mode runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TestModeConfig:
    N: int
    G: int
    n_perm: int
    bins_grid: list[int]
    w_grid: list[int]
    D_grid: list[int]
    sigma_eta_grid: list[float]
    pi_grid: list[float]
    beta_grid: list[float]
    dropout_grid: list[float]
    geometries: list[str]
    genes_per_class: int
    seeds: list[int]


def get_testmode_config(master_seed: int) -> TestModeConfig:
    return TestModeConfig(
        N=3000,
        G=120,
        n_perm=120,
        bins_grid=[36],
        w_grid=[1, 3],
        D_grid=[5],
        sigma_eta_grid=[0.4],
        pi_grid=[0.05, 0.2],
        beta_grid=[0.0, 1.0],
        dropout_grid=[0.0, 0.2],
        geometries=["disk_gaussian", "density_gradient_disk"],
        genes_per_class=40,
        seeds=[int(master_seed)],
    )


def resolve_outdir(outdir: str | Path, test_mode: bool) -> Path:
    forced = os.environ.get("BIORSP_SIM_RUN_DIR")
    if forced:
        return Path(forced)
    p = Path(outdir)
    if p.name.startswith("run_") and p.parent.name == "runs":
        return p
    return p / "test_mode" if bool(test_mode) else p


def banner(test_mode: bool, cfg: TestModeConfig | None) -> None:
    if not bool(test_mode):
        print("test_mode=False (full run)")
        return
    if cfg is None:
        print("test_mode=True")
        return
    print(
        (
            "test_mode=True "
            f"(N={cfg.N}, G={cfg.G}, n_perm={cfg.n_perm}, "
            f"D={cfg.D_grid}, sigma={cfg.sigma_eta_grid}, "
            f"bins={cfg.bins_grid}, w={cfg.w_grid}, "
            f"pi={cfg.pi_grid}, geometries={cfg.geometries}, seeds={cfg.seeds})"
        )
    )


def apply_testmode_overrides(args: Any, exp_name: str | None = None) -> Any:
    """Apply generic deterministic down-scaling to arbitrary argparse namespaces."""
    cfg = get_testmode_config(int(getattr(args, "master_seed", 123)))

    def _set_if(name: str, value: Any) -> None:
        if hasattr(args, name):
            setattr(args, name, value)

    # Common scalar sizes.
    _set_if("N", int(cfg.N))
    _set_if("N_total", int(cfg.N))
    _set_if("G", int(cfg.G))
    _set_if("G_total", int(cfg.G))
    _set_if(
        "genes_per_condition", min(int(getattr(args, "genes_per_condition", cfg.G)), 40)
    )
    _set_if("n_perm", int(cfg.n_perm))
    _set_if("n_perm_pool", max(80, int(cfg.n_perm)))
    _set_if("n_perm_donor", max(60, int(cfg.n_perm // 2)))
    _set_if("runs", 1)
    _set_if("n_master_seeds", 1)
    _set_if("n_seeds", 1)

    # Representative grids.
    _set_if("D", int(cfg.D_grid[0]))
    _set_if("D_grid", [int(cfg.D_grid[0])])
    _set_if("sigma_eta_grid", [float(cfg.sigma_eta_grid[0])])
    _set_if("pi_grid", [float(x) for x in cfg.pi_grid[:2]])
    _set_if("beta_grid", [float(x) for x in cfg.beta_grid[:2]])
    _set_if("dropout_grid", [float(x) for x in cfg.dropout_grid[:2]])
    _set_if("g_qc_grid", [0.0, 1.0])
    _set_if("gamma_grid", [0.5, 1.0])
    _set_if("N_grid", [int(cfg.N)])
    _set_if("bins_grid", [int(cfg.bins_grid[0])])
    _set_if("w_grid", [int(cfg.w_grid[0]), int(cfg.w_grid[1])])

    if hasattr(args, "geometries"):
        geos = getattr(args, "geometries")
        if isinstance(geos, list):
            keep = [g for g in geos if g in cfg.geometries]
            setattr(args, "geometries", keep or cfg.geometries[:1])
    if hasattr(args, "geometry"):
        g = getattr(args, "geometry")
        if isinstance(g, list):
            keep = [x for x in g if x in cfg.geometries]
            setattr(args, "geometry", keep or cfg.geometries[:1])

    # Embedding-robustness stress subset in fast mode.
    _set_if("variants", ["V0", "V1", "V2", "V5", "V8"])
    _set_if("origins", ["O1", "O2", "O3"])
    _set_if("fast_mode", True)

    _set_if("progress_every", min(int(getattr(args, "progress_every", 100)), 50))
    return args
