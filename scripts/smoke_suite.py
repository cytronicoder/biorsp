#!/usr/bin/env python3
"""
Orchestrate all BioRSP smoke tests with robust logging, timing, and validation.

Runs three experiments in order:
1. RSP smoke test
2. Moran's I baseline
3. Donor-stratified permutation test (n_perm=100)

Validates outputs, enforces quality gates, and prints a summary table.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any

import pandas as pd


# Expected output files
EXPECTED_FILES = {
    "rsp": "outputs/figures_v1/debug_rsp_marker.png",
    "moran": "outputs/tables/debug_moran.csv",
    "perm_fig": "outputs/figures_v1/null_marker.png",
    "perm_csv": "outputs/tables/perm_smoke.csv",
    "runlog": "outputs/logs/runlog.md",
}


def ensure_directories() -> None:
    """Ensure all required output directories exist."""
    dirs = [
        Path("outputs/figures_v1"),
        Path("outputs/tables"),
        Path("outputs/logs"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Ensure runlog exists under outputs/
    runlog = Path(EXPECTED_FILES["runlog"])
    if not runlog.exists():
        runlog.parent.mkdir(parents=True, exist_ok=True)
        runlog.touch()


def run_experiment(name: str, script: str, args: list[str]) -> tuple[float, str]:
    """Run a single experiment script and return timing + stdout.

    Args:
        name: Human-readable experiment name.
        script: Python script to run (e.g., 'scripts/rsp_smoke.py').
        args: Additional arguments to pass.

    Returns:
        Tuple of (duration in seconds, stdout text).
    """
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, script] + args
    start = time.perf_counter()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        duration = time.perf_counter() - start
        
        # Print stdout
        if result.stdout:
            print(result.stdout)
        
        return duration, result.stdout
    
    except subprocess.CalledProcessError as e:
        duration = time.perf_counter() - start
        print(f"❌ FAILED after {duration:.2f}s")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        sys.exit(1)


def validate_output(filepath: str, min_size_bytes: int = 0) -> bool:
    """Check that an output file exists and has sufficient size.

    Args:
        filepath: Path to file.
        min_size_bytes: Minimum expected size (default 0).

    Returns:
        True if file exists and meets size requirements.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"❌ Expected output missing: {filepath}")
        return False
    
    size = path.stat().st_size
    if size < min_size_bytes:
        print(f"❌ Output too small: {filepath} ({size} bytes < {min_size_bytes})")
        return False
    
    print(f"✅ Validated: {filepath} ({size} bytes)")
    return True


def parse_stdout_value(stdout: str, key: str) -> str | None:
    """Extract a value from stdout like 'key=value'.

    Args:
        stdout: Stdout text to parse.
        key: Key to search for.

    Returns:
        Value string if found, otherwise None.
    """
    pattern = rf"{key}=([^\s]+)"
    match = re.search(pattern, stdout)
    return match.group(1) if match else None


def main() -> int:
    """Run all experiments and print summary.

    Returns:
        Exit code (0 for success).
    """
    warnings.warn(
        "scripts/smoke_suite.py is deprecated; run canonical CLI entrypoints directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    print("BioRSP Experiment Runner")
    print("=" * 70)
    
    # Setup
    ensure_directories()
    
    # Determine input file
    h5ad_path = "data/processed/HT_pca_umap.h5ad"
    if not Path(h5ad_path).exists():
        print(f"❌ Input file not found: {h5ad_path}")
        return 1
    
    print(f"Input: {h5ad_path}")
    
    # Storage for results
    timings: Dict[str, float] = {}
    rsp_metrics: Dict[str, Any] = {}
    
    # =========================================================================
    # Experiment 1: RSP smoke test
    # =========================================================================
    duration, stdout = run_experiment(
        "RSP Smoke Test",
        "scripts/rsp_smoke.py",
        ["--h5ad", h5ad_path, "--outdir", ".", "--bins", "72", "--seed", "0"]
    )
    timings["rsp"] = duration
    
    # Parse RSP metrics
    rsp_metrics["gene"] = parse_stdout_value(stdout, "gene")
    rsp_metrics["E_max"] = parse_stdout_value(stdout, "E_max")
    rsp_metrics["phi_max_rad"] = parse_stdout_value(stdout, "phi_max_rad")
    rsp_metrics["phi_max_deg"] = parse_stdout_value(stdout, "phi_max_deg")
    
    # Validate output
    if not validate_output(EXPECTED_FILES["rsp"], min_size_bytes=1024):
        return 1
    
    # Quality gate: runtime check
    if duration > 300:  # 5 minutes
        print(f"⚠️  WARNING: RSP took {duration:.2f}s (> 5 min threshold)")
    
    print(f"⏱️  Duration: {duration:.2f}s")
    
    # =========================================================================
    # Experiment 2: Moran's I baseline
    # =========================================================================
    duration, stdout = run_experiment(
        "Moran's I Baseline",
        "scripts/moran_smoke.py",
        ["--h5ad", h5ad_path, "--outdir", ".", "--seed", "0"]
    )
    timings["moran"] = duration
    
    # Validate outputs
    if not validate_output(EXPECTED_FILES["moran"]):
        return 1
    if not validate_output(EXPECTED_FILES["runlog"]):
        return 1
    
    print(f"⏱️  Duration: {duration:.2f}s")
    
    # =========================================================================
    # Experiment 3: Donor-stratified permutation test
    # =========================================================================
    duration, stdout = run_experiment(
        "Permutation Test (n=100)",
        "scripts/perm_smoke.py",
        [
            "--h5ad", h5ad_path,
            "--outdir", ".",
            "--bins", "72",
            "--seed", "0",
            "--n-perm", "100",
            "--donor-col", "hubmap_id"
        ]
    )
    timings["perm"] = duration
    
    # Validate outputs
    if not validate_output(EXPECTED_FILES["perm_fig"], min_size_bytes=1024):
        return 1
    if not validate_output(EXPECTED_FILES["perm_csv"]):
        return 1
    
    print(f"⏱️  Duration: {duration:.2f}s")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    # Read data files
    moran_df = pd.read_csv(EXPECTED_FILES["moran"])
    perm_df = pd.read_csv(EXPECTED_FILES["perm_csv"])
    
    print(f"\n📊 Marker Gene: {rsp_metrics.get('gene', 'N/A')}")
    print(f"\n🎯 RSP Metrics:")
    print(f"  E_max:        {rsp_metrics.get('E_max', 'N/A')}")
    print(f"  phi_max (rad): {rsp_metrics.get('phi_max_rad', 'N/A')}")
    print(f"  phi_max (deg): {rsp_metrics.get('phi_max_deg', 'N/A')}")
    
    print(f"\n📈 Moran's I:")
    for _, row in moran_df.iterrows():
        print(f"  {str(row['gene']):15s}: {row['moran_I']:.6f}")
    
    print(f"\n🎲 Permutation Test (p-values):")
    for _, row in perm_df.iterrows():
        print(f"  {str(row['gene']):15s}: {row['p_perm_100']:.4f}")
    
    print(f"\n⏱️  Timings:")
    total_time = sum(timings.values())
    for exp, dur in timings.items():
        print(f"  {exp:15s}: {dur:6.2f}s")
    print(f"  {'TOTAL':15s}: {total_time:6.2f}s")
    
    # Quality gates
    print(f"\n🚦 Quality Gates:")
    
    # Check housekeeping p-value
    # Look for the second entry (housekeeping) since marker is first
    if len(perm_df) >= 2:
        hk_p = perm_df.iloc[1]['p_perm_100']
        hk_gene = perm_df.iloc[1]['gene']
        if hk_p >= 0.2:
            print(f"  ✅ Housekeeping ({hk_gene}) p_perm >= 0.2: {hk_p:.4f}")
        else:
            print(f"  ⚠️  WARNING: Housekeeping ({hk_gene}) p_perm < 0.2: {hk_p:.4f}")
            print(f"      Check donor stratification and expression threshold.")
            print(f"      Housekeeping genes should usually be diffuse (non-significant).")
    
    # Check runtime
    if timings.get("rsp", 0) <= 300:
        print(f"  ✅ RSP runtime < 5 min")
    else:
        print(f"  ⚠️  RSP runtime > 5 min: {timings['rsp']:.2f}s")
    
    print("\n✅ All experiments completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
