# BioRSP Developer Notes

## Coordinate Systems and Conventions

### Angles
- All internal angles are in **radians**.
- The angular range is $[-\pi, \pi)$.
- Angular windows (sectors) are defined by a center $\phi_b$ and a width $\delta$.
- A cell $i$ with angle $\theta_i$ is in sector $b$ if the circular distance between $\theta_i$ and $\phi_b$ is $\le \delta/2$.

### Radial Distances
- Radial distances $r$ are typically normalized using robust scaling: $\hat{r} = \frac{r - \text{median}(r)}{\text{IQR}(r)}$.
- This ensures that the RSP statistic is invariant to the absolute scale of the embedding.

## Statistical Definitions

### RSP Radar Function $R_g(\theta)$
The radar function at angle $\theta$ is defined as:
$$R_g(\theta) = \text{sgn}(\text{med}(r_{fg}) - \text{med}(r_{bg})) \cdot W_1(r_{fg}, r_{bg})$$
where $W_1$ is the Wasserstein-1 distance between the foreground and background radial distributions within the angular window.

### RMS Anisotropy $A_g$
The global anisotropy score is the Root Mean Square of the radar function across all valid sectors:
$$A_g = \sqrt{\frac{1}{|B_{valid}|} \sum_{b \in B_{valid}} R_g(\phi_b)^2}$$

### P-value Calculation
We use a permutation test with finite-permutation correction:
$$p = \frac{1 + \sum_{k=1}^K [A_{null, k} \ge A_{obs}]}{K + 1}$$
where $K$ is the number of permutations. This ensures that the p-value is never zero and is strictly conservative.

## Implementation Details

### Determinism and RNG
- All random operations must use a `numpy.random.Generator` object.
- Permutation seeds are generated from a master seed and stored in the `InferenceResult`.
- Tie-breaking in quantiles and foreground selection is handled deterministically using the provided RNG.

### Performance
- Angular windowing is optimized using a two-pointer sliding window approach on sorted angles, reducing complexity from $O(N \cdot B)$ to $O(N \log N + B)$.
- Sector indices can be precomputed and reused across multiple features or permutations.

### Adequacy
- A feature is considered "adequate" if it has sufficient foreground mass and a minimum fraction of sectors meet the foreground/background count requirements.
- The `valid_mask` (which sectors are used for anisotropy) is fixed based on the observed data and reused for all permutations to ensure a consistent null distribution.
