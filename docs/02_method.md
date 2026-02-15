# Method

## Scientific Framing

- Geometry and directionality are representation-conditional.
- Angles are interpreted only in the analyzed embedding coordinate frame.
- This workflow is a heart case study for method validation, not biological subtype discovery.

## RSP Profile

For each feature, foreground is defined by detection (`expr > 0`) and directional contrast is:

- `R(theta_b) = pF_b - pB_b`

Primary per-feature outputs:

- `anisotropy = max_b |R(theta_b)|`
- `peak_direction_rad`
- permutation-calibrated `p_T`, BH-adjusted `q_T`
- Moran's I baseline (`continuous`, `binary detection`)

## Coordinate Convention

- `theta = atan2(y - y0, x - x0) mod 2*pi`
- `theta = 0` at East (`+UMAP1`)
- `theta` increases counterclockwise
- 90 degrees is the top of the plot (`+UMAP2`)

## A Priori Marker Panel

- Cardiomyocyte: `TNNT2`, `TTN`, `MYH6`, `MYH7`, `ACTC1`, `PLN`, `RYR2`
- Fibroblast/ECM: `COL1A1`, `COL1A2`, `DCN`, `LUM`
- Endothelial: `PECAM1`, `VWF`, `KDR`
- Pericyte/Smooth muscle: `RGS5`, `ACTA2`, `TAGLN`
- Immune: `PTPRC`, `LST1`, `LYZ`

Resolver order is `hugo_symbol -> gene_name -> gene_symbol -> var_names`.
If fewer than 12 panel genes resolve, auto-selected fallback genes are added and explicitly labeled as `auto_gene`.
