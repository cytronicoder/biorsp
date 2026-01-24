import subprocess
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


def _write_gene_scores(path: Path, label: str, offset: float = 0.0) -> None:
    data = {
        "gene": ["G1", "G2", "G3"],
        "gene_symbol": ["G1", "G2", "G3"],
        "coverage_expr": [0.2, 0.1, 0.05],
        "spatial_score": [0.3 + offset, 0.1 + offset, 0.05 + offset],
        "p_value": [0.01, 0.2, 0.5],
        "q_value": [0.02, 0.25, 0.55],
        "n_cells_total": [100, 100, 100],
        "warnings": [np.nan, np.nan, np.nan],
        "Archetype": ["Gradient", "Patchy", "Basal"],
    }
    df = pd.DataFrame(data)
    disease_dir = path / label
    disease_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(disease_dir / "gene_scores.csv", index=False)


def test_compare_disease_results_smoke(tmp_path: Path) -> None:
    results_dir = tmp_path / "disease_results"
    _write_gene_scores(results_dir, "healthy_reference", offset=0.0)
    _write_gene_scores(results_dir, "acute_kidney_injury", offset=0.2)

    # Minimal AnnData for plotting
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.1, 0.0],
            [0.0, 0.3, 0.1],
            [0.0, 0.0, 0.2],
        ]
    )
    adata = anndata.AnnData(X=X)
    adata.var_names = ["G1", "G2", "G3"]
    adata.obsm["X_umap"] = np.random.default_rng(0).normal(size=(adata.n_obs, 2))
    h5_path = tmp_path / "toy.h5ad"
    adata.write_h5ad(h5_path)

    script = Path(__file__).resolve().parents[2] / "analysis" / "kidney_atlas" / "utils" / "compare_disease_results.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(results_dir),
            "--h5ad",
            str(h5_path),
            "--top_k",
            "1",
            "--rank_by",
            "effect",
            "--min_expr_cells",
            "1",
            "--seed",
            "0",
        ],
        check=True,
    )

    output_dir = results_dir / "comparison_outputs" / "global"
    assert (output_dir / "ranked_genes.csv").exists()
    assert (output_dir / "figures" / "rank_scatter.png").exists()
