# Methods: Definitions

## 1. Foreground Definition

A cell $i$ is defined as "foreground" for gene $g$ if its detected expression count is greater than zero.

- **Rule**: $F_g = \{i \mid X_{i,g} > 0\}$
- **Threshold**: Strictly positive ($> 0$).
- **Unit**: Raw or normalized counts (binary detection).

## 2. Vantage Point

The vantage point $V = (x_c, y_c)$ is the coordinate-wise median of the embedding coordinates for all $N$ cells in the analysis scope (e.g., cluster or full dataset).

- **Formula**: $x_c = \text{median}(x_1, \dots, x_N)$, $y_c = \text{median}(y_1, \dots, y_N)$
- **Unit**: Embedding coordinate space (dimensionless).

## 3. Angle Calculation

For each cell $i$ with coordinates $(x_i, y_i)$, the angle $\theta_i$ relative to the vantage point is calculated using the 2-argument arctangent, mapped to the range $[0, 2\pi)$.

- **Formula**: $\theta_{raw} = \text{atan2}(y_i - y_c, x_i - x_c)$
- **Mapping**: $\theta_i = (\theta_{raw} + 2\pi) \pmod{2\pi}$
- **Unit**: Radians $[0, 2\pi)$.

## 4. Binning Strategy

The angular space is divided into $K$ equal-width sectoral bins.

- **Count ($K$)**: $60$ bins.
- **Width**: $\Delta\theta = \frac{2\pi}{60} \approx 0.1047$ radians ($6^{\circ}$).
- **Assignment**: Cell $i$ belongs to bin $k \in \{0, \dots, 59\}$ if $\lfloor \frac{\theta_i}{\Delta\theta} \rfloor = k$.
- **Edge Convention**: Lower bound inclusive, upper bound exclusive $[a, b)$.

## 5. Enrichment Score $E(\phi)$

The directional enrichment $E(\phi_k)$ for bin $k$ is defined as the ratio of the observed proportion of foreground cells in the bin to the proportion of total cells in the bin (background density).

- **Formula**:
  $$ E(\phi*k) = \frac{n*{k, fg} / N*{fg}}{n*{k, total} / N\_{total}} $$
  Where:
  - $n_{k, fg}$: Number of foreground cells in bin $k$.
  - $N_{fg}$: Total number of foreground cells.
  - $n_{k, total}$: Total number of cells (background) in bin $k$.
  - $N_{total}$: Total number of cells in the analysis.
- **Description**: This metric captures the density of gene-expressing cells in a specific direction relative to the overall cell density in that direction, normalizing for non-uniform sampling of the angular space.

## 6. Outputs

The primary outputs for each gene are:

1. **$E_{max}$**: The maximum enrichment score across all bins.
   - Formula: $E_{max} = \max_{k} E(\phi_k)$
2. **$\phi_{max}$**: The center angle of the bin $k$ where the maximum enrichment occurs.
   - Unit: Radians $[0, 2\pi)$.
3. **$p_{perm}$**: Empirical p-value derived from $M$ permutations (default $M=1000$) of gene expression values.
   - Formula: $p = (R + 1) / (M + 1)$, where $R$ is the count of permuted $E_{max}$ values $\ge$ observed $E_{max}$.
4. **Moran's I**: Spatial autocorrelation score calculated on the k-nearest neighbor graph of the embedding or high-dimensional space.
