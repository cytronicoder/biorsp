from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Sequence, Tuple, List, Union, TYPE_CHECKING
import logging
import warnings

import numpy as np

from . import preprocessing as prep
from . import stats
from . import null_models
from . import utils

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ScanParams:
    """Configuration parameters for RadarScanner."""

    # Coordinate frame
    center_mode: Literal["geom_median", "cluster", "user"] = "geom_median"
    user_center: Optional[Tuple[float, float]] = None
    B: int = 180  # angular bins
    offset_rad: float = 0.0  # visualization alignment only

    # Radial handling
    radial_mode: Literal["all", "quantile", "width"] = "all"
    n_bands: int = 1
    bands: Optional[Sequence[Tuple[float, float]]] = None  # explicit annuli

    # Window bank / kernel
    widths_deg: Sequence[float] = (15, 30, 60, 90, 120, 180)
    kernel_kind: Literal["boxcar", "vonmises"] = "boxcar"
    kappa: Optional[float] = None  # von Mises taper; None = auto
    normalize: Literal["area", "l1", "l2"] = "area"

    # Feature processing
    standardize: Literal["z", "rank", "mad", "none"] = "z"
    residualize: Literal["none", "ols", "ridge"] = "none"
    ridge_alpha: float = 1.0

    # Density correction
    density_correction: Literal["none", "2d", "ratio"] = "none"
    k_hd: int = 30
    k_2d: int = 30

    # Variance & testing
    var_mode: Literal["binomial", "plugin"] = "binomial"
    overdispersion: float = 0.0
    ESS_min: float = 25.0  # minimum effective sample size in sector

    # Null model
    null_model: Literal["rotation", "within_batch_rotation", "permutation"] = "rotation"
    R: int = 500  # resamples
    fdr: Literal["BH"] = "BH"  # keep simple; extend later

    # Performance / runtime
    engine: Literal["fft", "prefix"] = "fft"
    random_state: Optional[int] = None
    log_level: int = 20  # logging.INFO


@dataclass
class FeatureResult:
    """Result object for a single feature scan."""

    name: Optional[str] = None
    Z_max: float = 0.0
    p_value: float = 1.0
    q_value: Optional[float] = None
    phi_star: float = 0.0  # peak angle in radians
    width_idx: int = 0
    center_idx: int = 0
    ER: float = 0.0  # enrichment ratio
    R_conc: float = 0.0  # concentration ratio
    Z_heat: Optional[np.ndarray] = None  # (J, B) heatmap
    flags: dict = field(default_factory=dict)
    runtime: float = 0.0
    params: dict = field(default_factory=dict)


class RadarScanner:
    """Scanner for radial spatial polarization analysis."""

    def __init__(self, params: ScanParams):
        """
        Create a scanner with fixed parameters.

        Args:
            params (ScanParams): Configuration parameters for the scanner.

        Notes:
            - `params` are copied to internal dict and stored in every result.
            - Logger is initialized with `params.log_level`.
        """
        self.params = asdict(params)
        self._params_obj = params
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(params.log_level)
        self.N: Optional[int] = None
        self.B: int = params.B
        self.center: Optional[np.ndarray] = None
        self.theta: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None
        self.wedge_idx: Optional[np.ndarray] = None
        self.band_idx: Optional[np.ndarray] = None
        self.band_edges: Optional[np.ndarray] = None
        self.batches: Optional[np.ndarray] = None
        self.density_w: Optional[np.ndarray] = None
        self.kernels: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        coords_2d: np.ndarray,
        *,
        feature_matrix: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
        covariates: Optional[np.ndarray] = None,
        highd: Optional[np.ndarray] = None,
        cluster_ids: Optional[np.ndarray] = None,
        batches: Optional[np.ndarray] = None,
    ) -> "RadarScanner":
        """
        Prepare reusable structures for scanning.

        Args:
            coords_2d (np.ndarray): 2D embedding coordinates, shape (N, 2).
            feature_matrix (Optional[np.ndarray], optional): Reserved for future use.
            covariates (Optional[np.ndarray], optional): Covariate matrix for residualization.
            highd (Optional[np.ndarray], optional): High-dimensional features for density ratio.
            cluster_ids (Optional[np.ndarray], optional): Cluster labels for center computation.
            batches (Optional[np.ndarray], optional): Batch assignments for batch-aware nulls.

        Returns:
            RadarScanner: The fitted scanner instance (self).

        Notes:
            - Validates inputs and parameters.
            - Chooses `center` and converts coords to polar: theta (0..2π), radius r.
            - Discretizes angles into B wedges; assigns radial bands.
            - Prepares density weights if requested:
              * '2d' -> inverse 2D crowding weights
              * 'ratio' -> rho_HD / rho_2D ratio
            - Builds and caches the window bank (kernels) for all widths.
            - Stores: self.N, self.B, self.center, self.theta, self.r,
              self.wedge_idx, self.band_idx, self.band_edges, self.batches,
              self.density_w, self.kernels, self.params.
            - Warns if N < 100 or if many radii are identical (likely duplicates).
            - If density correction is 'ratio' but `highd` is missing, falls back to '2d'.
        """
        coords_2d = np.asarray(coords_2d)
        prep.validate_inputs(
            coords_2d,
            feature=None,
            covariates=covariates,
            bands=self._params_obj.bands,
            B=self.B,
        )

        self.N = coords_2d.shape[0]
        self.center = self._compute_center(coords_2d, cluster_ids)
        self.theta, self.r = self._to_polar(coords_2d, self.center)
        self.wedge_idx = self._discretize_angles(self.theta)
        self.band_idx, self.band_edges = self._assign_bands(self.r)

        if batches is not None:
            self.batches = np.asarray(batches)
            if self.batches.shape[0] != self.N:
                raise ValueError(f"batches must have length N={self.N}")

        self.density_w = self._compute_density_weights(coords_2d, highd)
        self.kernels = self._build_kernels()

        self._fitted = True
        self.logger.info(
            "Fitted RadarScanner: N=%d, B=%d, bands=%d, kernels=%d",
            self.N,
            self.B,
            len(np.unique(self.band_idx)),
            len(self.kernels),
        )

        return self

    def scan_feature(
        self,
        feature: np.ndarray,  # (N,)
        *,
        name: Optional[str] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> FeatureResult:
        """
        Run the full RSP pipeline for a single feature.

        Args:
            feature (np.ndarray): Feature values, shape (N,).
            name (Optional[str], optional): Feature name for result provenance.
            covariates (Optional[np.ndarray], optional): Covariates for residualization.

        Returns:
            FeatureResult: Complete summary including Z_heat and provenance.

        Raises:
            RuntimeError: If fit() has not been called before scan_feature().

        Notes:
            - Pipeline steps:
              1. Preprocess feature -> weights (standardize, residualize, apply density).
              2. Bin weights to per-band wedge sums S[a,b] via fast bincount.
              3. For each window kernel, compute Obs/Exp/Var and Z grid across centers.
              4. Take global max over widths × centers; record (width_idx*, center_idx*),
                 Z_max, ER, R_conc.
              5. Calibrate significance with chosen null model (rotation/permutation).
              6. Compute QC flags.
            - QC flags:
              * `low_ess`: effective sample size in winning sector < ESS_min.
              * `unstable_width`: neighboring widths disagree strongly (delta Z > 2).
              * `density_extreme`: top/bottom 1% density weights dominate the sector.
            - Runtime: O(N) to bin, O(J * B log B) to convolve (FFT),
              plus O(R * J * B log B) for nulls.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before scan_feature()")

        import time as time_module

        start_time = time_module.time()

        feature = np.asarray(feature)
        if feature.shape[0] != self.N:
            raise ValueError(f"feature must have length N={self.N}")

        weights = self._preprocess_feature(feature, covariates)
        S = self._bin_to_wedges(weights)
        Z_grids = []
        for kernel in self.kernels:
            Z_grid = self._compute_Z_grid(S, kernel)
            Z_grids.append(Z_grid)

        Z_heat = np.array(Z_grids)
        max_idx = np.unravel_index(np.argmax(Z_heat), Z_heat.shape)
        width_idx, center_idx = max_idx  # pylint: disable=unbalanced-tuple-unpacking
        Z_max = Z_heat[width_idx, center_idx]
        ER, R_conc = self._compute_enrichment_stats(S, width_idx, center_idx)

        p_value = self._calibrate_pvalue(weights, Z_max)
        flags = self._compute_qc_flags(width_idx, center_idx, Z_heat)
        phi_star = (center_idx * 2 * np.pi / self.B + self._params_obj.offset_rad) % (
            2 * np.pi
        )

        runtime = time_module.time() - start_time

        return FeatureResult(
            name=name,
            Z_max=Z_max,
            p_value=p_value,
            q_value=None,  # computed later in batch
            phi_star=phi_star,
            width_idx=width_idx,
            center_idx=center_idx,
            ER=ER,
            R_conc=R_conc,
            Z_heat=Z_heat,
            flags=flags,
            runtime=runtime,
            params=self.params.copy(),
        )

    def scan_matrix(
        self,
        X: np.ndarray,  # (N, G)
        *,
        names: Optional[Sequence[str]] = None,
        covariates: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,  # pylint: disable=unused-argument
        progress: bool = True,
        return_table: bool = False,
    ) -> Union[List[FeatureResult], "pd.DataFrame"]:
        """
        Scan many features efficiently.

        Args:
            X (np.ndarray): Feature matrix, shape (N, G).
            names (Optional[Sequence[str]], optional): Feature names for provenance.
                Defaults to "feature_0", "feature_1", etc.
            covariates (Optional[np.ndarray], optional): Covariates for residualization.
            batch_size (Optional[int], optional): Reserved for future mini-batching.
            progress (bool, optional): Whether to log progress. Defaults to True.
            return_table (bool, optional): If True and pandas available, return DataFrame.
                Defaults to False.

        Returns:
            Union[List[FeatureResult], pd.DataFrame]: List of FeatureResult objects,
                or DataFrame if return_table=True and pandas is available.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If X is not 2D or has wrong number of rows.

        Notes:
            - Iterates columns of X, calling `scan_feature` with shared caches.
            - If `params.fdr == "BH"`, q-values are computed across the batch at the end.
            - Progress logging is routed to the logger.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before scan_matrix()")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != self.N:
            raise ValueError(f"X must have N={self.N} rows")

        G = X.shape[1]

        if names is None:
            names = [f"feature_{i}" for i in range(G)]
        elif len(names) != G:
            raise ValueError(f"names must have length G={G}")

        results = []

        for i in range(G):
            if progress and i % 100 == 0:
                self.logger.info("Scanning feature %d/%d", i + 1, G)

            result = self.scan_feature(X[:, i], name=names[i], covariates=covariates)
            results.append(result)

        if self._params_obj.fdr == "BH":
            p_values = np.array([r.p_value for r in results])
            q_values = self._benjamini_hochberg(p_values)
            for i, result in enumerate(results):
                result.q_value = q_values[i]

        if return_table:
            return self.to_dataframe(results)

        return results

    def consensus(
        self,
        results_across_embeddings: Sequence[Sequence[FeatureResult]],
        *,
        rule: Literal["median_effect", "vote_q<0.05"] = "vote_q<0.05",
    ) -> "pd.DataFrame":
        """
        Combine results from multiple embeddings/seeds into a stability summary.

        Args:
            results_across_embeddings (Sequence[Sequence[FeatureResult]]): Results from
                multiple runs, where each run is a list of FeatureResult objects.
            rule (Literal["median_effect", "vote_q<0.05"], optional): Decision rule
                for consensus calls. Defaults to "vote_q<0.05".

        Returns:
            pd.DataFrame: Consensus table with per-feature vote fractions, median Z_max,
                median phi*, circular dispersion of phi*, and recommended call.

        Raises:
            ImportError: If pandas is not installed.

        Notes:
            - Groups results by feature name across runs.
            - Computes vote fraction (q < 0.05), median Z_max, circular median phi*,
              and circular dispersion.
            - Rule "vote_q<0.05": call if > 50% of runs have q < 0.05.
            - Rule "median_effect": call if median Z_max > 2.0.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for consensus()") from exc

        feature_names = [r.name for r in results_across_embeddings[0]]
        consensus_data = []
        for fname in feature_names:
            feature_results = [
                [r for r in run if r.name == fname][0]
                for run in results_across_embeddings
            ]

            q_values = [r.q_value for r in feature_results if r.q_value is not None]
            vote_sig = np.mean([q < 0.05 for q in q_values]) if q_values else 0.0

            z_maxes = [r.Z_max for r in feature_results]
            median_z = np.median(z_maxes)

            phis = [r.phi_star for r in feature_results]
            median_phi = self._circular_median(phis)
            phi_dispersion = self._circular_dispersion(phis)

            if rule == "vote_q<0.05":
                call = vote_sig > 0.5
            else:  # median_effect
                call = median_z > 2.0

            consensus_data.append(
                {
                    "feature": fname,
                    "vote_q<0.05": vote_sig,
                    "median_Z_max": median_z,
                    "median_phi": median_phi,
                    "phi_dispersion": phi_dispersion,
                    "call": call,
                }
            )

        return pd.DataFrame(consensus_data)

    def to_dataframe(self, results: Sequence[FeatureResult]) -> "pd.DataFrame":
        """
        Flatten FeatureResult objects into a tidy DataFrame.

        Args:
            results (Sequence[FeatureResult]): List of feature scan results.

        Returns:
            pd.DataFrame: Tidy DataFrame with one row per feature, including
                Z_max, p_value, q_value, phi_star, width_idx, center_idx,
                ER, R_conc, runtime, and expanded flag columns.

        Raises:
            ImportError: If pandas is not installed.

        Notes:
            - Computes BH q-values if necessary.
            - Expands flags into boolean columns (e.g., flag_low_ess).
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        data = []
        for r in results:
            row = {
                "name": r.name,
                "Z_max": r.Z_max,
                "p_value": r.p_value,
                "q_value": r.q_value,
                "phi_star": r.phi_star,
                "width_idx": r.width_idx,
                "center_idx": r.center_idx,
                "ER": r.ER,
                "R_conc": r.R_conc,
                "runtime": r.runtime,
            }

            for flag_name, flag_value in r.flags.items():
                row[f"flag_{flag_name}"] = flag_value
            data.append(row)

        return pd.DataFrame(data)

    def get_config(self) -> dict:
        """
        Return a JSON-serializable dict of ScanParams and derived runtime options.

        Returns:
            dict: Configuration dictionary including B, widths, kernel_kind,
                null_model, engine, and if fitted, N and n_kernels.

        Notes:
            - Useful for provenance tracking and reproducibility.
        """
        config = self.params.copy()
        config["B"] = self.B
        if self._fitted:
            config["N"] = self.N
            config["n_kernels"] = len(self.kernels) if self.kernels is not None else 0
        return config

    def _compute_center(
        self, coords: np.ndarray, cluster_ids: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute center based on center_mode.

        Args:
            coords (np.ndarray): 2D coordinates, shape (N, 2).
            cluster_ids (Optional[np.ndarray]): Cluster labels for 'cluster' mode.

        Returns:
            np.ndarray: Center point (x, y).
        """
        return prep.choose_center(
            coords,
            mode=self._params_obj.center_mode,
            cluster_ids=cluster_ids,
            user_center=self._params_obj.user_center,
        )

    def _to_polar(
        self, coords: np.ndarray, center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Cartesian to polar coordinates.

        Args:
            coords (np.ndarray): Cartesian coordinates, shape (N, 2).
            center (np.ndarray): Center point (x, y).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Angles theta in [0, 2π) and radii r.
        """
        return prep.to_polar(coords, tuple(center))

    def _discretize_angles(self, theta: np.ndarray) -> np.ndarray:
        """
        Discretize angles into B wedges (rotation-invariant for discovery).

        Args:
            theta (np.ndarray): Angles in radians, shape (N,).

        Returns:
            np.ndarray: Wedge indices in [0, B-1], shape (N,).

        Notes:
            - Uses offset=0.0 to ensure rotation-invariant discovery.
            - The offset_rad parameter is only applied when reporting phi_star.
        """
        return prep.bin_angles(theta, self.B, offset=0.0)

    def _assign_bands(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign radial bands based on radial_mode.

        Args:
            r (np.ndarray): Radii from center, shape (N,).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Band indices (N,) and band edges (A, 2).
        """
        band_idx, edges = prep.assign_radial_bands(
            r,
            bands=self._params_obj.bands,
            mode=self._params_obj.radial_mode,
            n_bands=self._params_obj.n_bands,
        )
        return band_idx, edges

    def _compute_density_weights(
        self, coords: np.ndarray, highd: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute density correction weights.

        Args:
            coords (np.ndarray): 2D coordinates, shape (N, 2).
            highd (Optional[np.ndarray]): High-dimensional features for 'ratio' mode.

        Returns:
            Optional[np.ndarray]: Density weights, shape (N,), or None if mode is 'none'.

        Notes:
            - Mode '2d': inverse 2D crowding weights.
            - Mode 'ratio': rho_HD / rho_2D ratio.
            - Falls back to '2d' if 'ratio' requested but highd is None.
        """
        mode = self._params_obj.density_correction

        if mode == "none":
            return None
        elif mode == "2d":
            return prep.density_weights(
                coords, k=self._params_obj.k_2d, return_density=False
            )
        elif mode == "ratio":
            if highd is None:
                warnings.warn(
                    "highd not provided for density_correction='ratio', falling back to '2d'",
                    UserWarning,
                )
                return prep.density_weights(
                    coords, k=self._params_obj.k_2d, return_density=False
                )
            return prep.density_ratio(
                highd=highd,
                coords_2d=coords,
                k_hd=self._params_obj.k_hd,
                k_2d=self._params_obj.k_2d,
            )
        else:
            raise ValueError(f"Unknown density_correction: {mode}")

    def _build_kernels(self) -> np.ndarray:
        """
        Build window bank (kernels) for all widths.

        Returns:
            np.ndarray: Kernel array, shape (J, B), where J is the number of widths.

        Notes:
            - Each kernel is normalized according to params.normalize.
            - Kernel type is determined by params.kernel_kind.
        """
        widths_deg = self._params_obj.widths_deg
        kernels = []
        for width_deg in widths_deg:
            kernel = stats.make_kernel(
                B=self.B,
                width_deg=width_deg,
                kind=self._params_obj.kernel_kind,
                kappa=self._params_obj.kappa,
                normalize=self._params_obj.normalize,
            )
            kernels.append(kernel)
        return np.array(kernels)

    def _preprocess_feature(
        self, feature: np.ndarray, covariates: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess feature: standardize, residualize, apply density weights.

        Args:
            feature (np.ndarray): Raw feature values, shape (N,).
            covariates (Optional[np.ndarray]): Covariates for residualization.

        Returns:
            np.ndarray: Preprocessed weights, shape (N,).

        Notes:
            - Applies standardization according to params.standardize.
            - Optionally residualizes against covariates if params.residualize != 'none'.
            - Multiplies by density_w if available.
        """
        weights = prep.standardize_feature(feature, method=self._params_obj.standardize)

        if self._params_obj.residualize != "none" and covariates is not None:
            weights = prep.residualize_feature(
                weights,
                covariates,
                method=self._params_obj.residualize,
                alpha=self._params_obj.ridge_alpha,
            )

        if self.density_w is not None:
            weights = weights * self.density_w

        return weights

    def _bin_to_wedges(self, weights: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Bin weights to wedge sums, supporting multiple radial bands.

        Args:
            weights (np.ndarray): Per-cell weights, shape (N,).
            mask (np.ndarray, optional): Boolean mask for subsetting, shape (N,).

        Returns:
            np.ndarray: Wedge sums, shape (B,) if single band or (A, B) if multiple bands.

        Notes:
            - Uses fast bincount for efficiency.
            - Automatically detects number of bands from self.band_idx.
        """
        # Apply mask if provided
        if mask is not None:
            weights = weights[mask]
            wedge_idx = self.wedge_idx[mask]
            band_idx = self.band_idx[mask] if self.band_idx is not None else None
        else:
            wedge_idx = self.wedge_idx
            band_idx = self.band_idx
        
        A = int(np.max(band_idx)) + 1 if band_idx is not None else 1
        if A == 1:
            return np.bincount(wedge_idx, weights=weights, minlength=self.B)
        S = np.zeros((A, self.B), dtype=float)
        for a in range(A):
            m = band_idx == a
            if m.any():
                S[a] = np.bincount(
                    wedge_idx[m], weights=weights[m], minlength=self.B
                )
        return S

    def _compute_Z_grid(self, S: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Compute Z-scores across all centers using stats.compute_Z_grid.

        Args:
            S (np.ndarray): Wedge sums, shape (B,) or (A, B).
            kernel (np.ndarray): Circular kernel, shape (B,).

        Returns:
            np.ndarray: Z-scores across centers, shape (B,).

        Notes:
            - Ensures S is in (A, B) shape internally.
            - Aggregates across bands using fixed-effects model if A > 1.
            - Delegates to stats.compute_Z_grid for proper variance computation.
        """
        # Track if input was 1D to squeeze output
        squeeze_output = False
        if S.ndim == 1:
            S = S[None, :]
            squeeze_output = True
        totals = S.sum(axis=1)

        _obs, _exp, _var, Z = stats.compute_Z_grid(
            wedge_sums=S,
            totals_per_band=totals,
            kernel=kernel,
            B=self.B,
            var_mode=self._params_obj.var_mode,
            overdispersion=self._params_obj.overdispersion,
            engine=self._params_obj.engine,
        )

        if S.shape[0] > 1:
            Z = stats.aggregate_bands(Z, method="fixed")
        elif squeeze_output:
            # If input was 1D, return 1D output
            Z = Z.squeeze()

        return Z

    def _compute_enrichment_stats(
        self, S: np.ndarray, width_idx: int, center_idx: int
    ) -> Tuple[float, float]:
        """
        Compute enrichment ratio and concentration ratio.

        Args:
            S (np.ndarray): Wedge sums, shape (B,) or (A, B).
            width_idx (int): Index of the winning kernel width.
            center_idx (int): Index of the winning center.

        Returns:
            Tuple[float, float]: Enrichment ratio (ER) and concentration ratio (R_conc).

        Notes:
            - ER = Obs / Exp, where Obs is observed sum in sector, Exp is expected
              under uniformity.
            - R_conc = mean resultant length (circular concentration metric).
            - Collapses bands by summing if S is (A, B).
        """
        kernel = self.kernels[width_idx]
        S_use = S if S.ndim == 1 else S.sum(axis=0)
        kernel_rolled = np.roll(kernel, center_idx)
        obs = float(np.sum(S_use * kernel_rolled))
        total = float(np.sum(S_use))
        exp = stats.expected_from_uniform(total, np.sum(kernel), self.B)
        ER = float(stats.enrichment_ratio(obs, exp))
        R_conc = float(stats.mean_resultant_length(S_use, B=self.B))
        return ER, R_conc

    def _calibrate_pvalue(self, weights: np.ndarray, Z_obs: float) -> float:
        """
        Calibrate p-value using null model.

        Args:
            weights (np.ndarray): Preprocessed per-cell weights, shape (N,).
            Z_obs (float): Observed maximum Z-score.

        Returns:
            float: Empirical p-value from the null distribution.

        Raises:
            ValueError: If batches are required but not provided.

        Notes:
            - Bins weights to wedge sums, ensuring proper (A, B) shape.
            - Delegates to appropriate null_models function based on params.null_model.
            - Supports 'rotation', 'within_batch_rotation', and 'permutation' null models.
        """
        S = self._bin_to_wedges(weights)
        if S.ndim == 1:
            wedge_sums = S[None, :]
            totals_per_band = np.array([np.sum(S)])
        else:
            wedge_sums = S
            totals_per_band = S.sum(axis=1)

        null_mode = self._params_obj.null_model
        rng = utils.check_random_state(self._params_obj.random_state)

        if null_mode == "rotation":
            p_value, _ = null_models.rotation_null_pvalue(
                wedge_sums=wedge_sums,
                totals_per_band=totals_per_band,
                kernels=self.kernels,
                B=self.B,
                R=self._params_obj.R,
                var_mode=self._params_obj.var_mode,
                overdispersion=self._params_obj.overdispersion,
                engine=self._params_obj.engine,
                zmax_obs=Z_obs,
                random_state=rng,
            )
        elif null_mode == "within_batch_rotation":
            if self.batches is None:
                raise ValueError("batches required for within_batch_rotation")

            unique_batches = np.unique(self.batches)
            G = len(unique_batches)
            A = wedge_sums.shape[0]
            wedge_sums_per_batch = np.zeros((G, A, self.B))
            totals_per_band_per_batch = np.zeros((G, A))

            for g, batch in enumerate(unique_batches):
                mask = self.batches == batch
                S_batch = self._bin_to_wedges(weights, mask=mask)
                if S_batch.ndim == 1:
                    S_batch = S_batch[None, :]
                wedge_sums_per_batch[g] = S_batch
                totals_per_band_per_batch[g] = S_batch.sum(axis=1)

            p_value, _ = null_models.within_batch_rotation_pvalue(
                wedge_sums_per_batch=wedge_sums_per_batch,
                totals_per_band_per_batch=totals_per_band_per_batch,
                kernels=self.kernels,
                B=self.B,
                R=self._params_obj.R,
                var_mode=self._params_obj.var_mode,
                overdispersion=self._params_obj.overdispersion,
                engine=self._params_obj.engine,
                zmax_obs=Z_obs,
                random_state=rng,
            )
        elif null_mode == "permutation":
            p_value, _ = null_models.permutation_null_pvalue(
                weights=weights,
                wedge_idx=self.wedge_idx,
                band_idx=self.band_idx,
                kernels=self.kernels,
                B=self.B,
                A=len(np.unique(self.band_idx)),
                R=self._params_obj.R,
                var_mode=self._params_obj.var_mode,
                overdispersion=self._params_obj.overdispersion,
                engine=self._params_obj.engine,
                batches=self.batches,
                zmax_obs=Z_obs,
                random_state=rng,
            )
        else:
            raise ValueError(f"Unknown null_model: {null_mode}")

        return p_value

    def _in_sector_mask(self, width_idx: int, center_idx: int) -> np.ndarray:
        """
        Generate per-cell boolean mask for the winning sector.

        Args:
            width_idx (int): Index of the winning kernel width.
            center_idx (int): Index of the winning center.

        Returns:
            np.ndarray: Boolean mask, shape (N,), True for cells in the sector.

        Notes:
            - Includes any bin with positive kernel weight after rolling to center.
            - Used for per-cell QC analysis.
        """
        K = np.roll(self.kernels[width_idx], center_idx)
        return K[self.wedge_idx] > 0

    def _compute_qc_flags(
        self, width_idx: int, center_idx: int, Z_heat: np.ndarray
    ) -> dict:
        """
        Compute QC flags using per-cell membership.

        Args:
            width_idx (int): Index of the winning kernel width.
            center_idx (int): Index of the winning center.
            Z_heat (np.ndarray): Z-score heatmap, shape (J, B).

        Returns:
            dict: Dictionary of QC flags with boolean values.

        Notes:
            - Flags computed:
              * low_ess: effective sample size < ESS_min.
              * unstable_width: neighboring widths differ by > 2 Z-units.
              * density_extreme: top/bottom 1% density weights exceed 10% of sector.
            - Uses per-cell sector mask for accurate ESS and density checks.
        """
        flags = {}
        m = self._in_sector_mask(width_idx, center_idx)
        ess = utils.effective_sample_size(np.ones(self.N)[m])
        flags["low_ess"] = ess < self._params_obj.ESS_min

        if 0 < width_idx < (len(self.kernels) - 1):
            z_prev = Z_heat[width_idx - 1, center_idx]
            z_curr = Z_heat[width_idx, center_idx]
            z_next = Z_heat[width_idx + 1, center_idx]
            flags["unstable_width"] = (abs(z_curr - z_prev) > 2.0) or (
                abs(z_curr - z_next) > 2.0
            )
        else:
            flags["unstable_width"] = False

        if self.density_w is not None and m.any():
            dw = self.density_w[m]
            if dw.size >= 20:
                top, bot = np.percentile(dw, 99), np.percentile(dw, 1)
                flags["density_extreme"] = (np.mean(dw >= top) > 0.10) or (
                    np.mean(dw <= bot) > 0.10
                )
            else:
                flags["density_extreme"] = False
        else:
            flags["density_extreme"] = False

        return flags

    def _benjamini_hochberg(self, p_values: np.ndarray) -> np.ndarray:
        """
        Compute Benjamini-Hochberg FDR q-values.

        Args:
            p_values (np.ndarray): Array of p-values.

        Returns:
            np.ndarray: Array of q-values (FDR-adjusted p-values).
        """
        return null_models.bh_qvalues(p_values)

    def _circular_median(self, angles: Sequence[float]) -> float:
        """
        Compute circular mean of angles.

        Args:
            angles (Sequence[float]): Angles in radians.

        Returns:
            float: Circular mean angle in [0, 2π).

        Notes:
            - Despite the name, this computes the circular mean, not median.
            - Uses complex number representation for robustness.
        """
        angles = np.array(angles)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        return np.arctan2(sin_mean, cos_mean) % (2 * np.pi)

    def _circular_dispersion(self, angles: Sequence[float]) -> float:
        """
        Compute circular dispersion of angles.

        Args:
            angles (Sequence[float]): Angles in radians.

        Returns:
            float: Circular dispersion (1 - R), where R is mean resultant length.

        Notes:
            - Values near 0 indicate concentrated angles.
            - Values near 1 indicate dispersed angles.
        """
        angles = np.array(angles)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        R = np.sqrt(sin_mean**2 + cos_mean**2)
        return 1 - R
