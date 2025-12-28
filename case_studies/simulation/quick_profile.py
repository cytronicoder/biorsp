import numpy as np
from line_profiler import LineProfiler
import sys
sys.path.append('/Users/cytronicoder/Documents/GitHub/biorsp-swordfish')

from biorsp.config import BioRSPConfig
from simulation import SimulationConfig, analyze_gene, generate_geometry_elliptical, generate_umis, generate_expression_null_B

# Quick profiling of analyze_gene
config = SimulationConfig(n_cells=1000, n_genes=1, n_permutations=10)  # Very small
biorsp_config = config.to_biorsp_config()

# Generate small data
z = generate_geometry_elliptical(1000, seed=42)
umis, _ = generate_umis(1000, seed=42)
x = generate_expression_null_B(umis, seed=42)

# Profile the function
lp = LineProfiler()
analyze_gene_profiled = lp(analyze_gene)
result = analyze_gene_profiled(z, x, umis, biorsp_config)

# Print stats
lp.print_stats()