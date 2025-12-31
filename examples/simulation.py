import argparse
import concurrent.futures
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from biorsp.config import BioRSPConfig
from biorsp.geometry import geometric_median
from biorsp.inference import compute_p_value
from biorsp.radar import compute_rsp_radar
from biorsp.summaries import compute_scalar_summaries

# --- Configuration ---


@dataclass
class SimulationConfig:
    # Reverted to original large defaults
    n_cells: int = 10000
    n_genes: int = 100
    n_permutations: int = 1000
    seed: int = 42

    # BioRSP defaults
    B: int = 360
    delta_deg: float = 20.0
    n_fg_min: int = 10
    n_bg_min: int = 50
    n_fg_tot_min: int = 100

    # Debug-friendly override: set environment var SIM_DEBUG to enable quick runs
    debug_override: Optional[Tuple[int, int]] = None  # (n_genes, n_permutations)

    def to_biorsp_config(self) -> BioRSPConfig:
        return BioRSPConfig(
            n_angles=self.B,
            sector_width_deg=self.delta_deg,
            min_fg_sector=self.n_fg_min,
            min_bg_sector=self.n_bg_min,
            min_fg_total=self.n_fg_tot_min,
            n_permutations=self.n_permutations,
            seed=self.seed,
        )


# --- Geometry Generators ---


def generate_geometry_elliptical(
    n: int, sigma_ratio: float = 1.5, gamma: float = 0.5, seed: int = 0
) -> np.ndarray:
    """
    Family 1: Elliptical cloud with mild density gradient.
    sigma_ratio: sigma_x / sigma_y
    gamma: density gradient strength exp(gamma * x)
    """
    rng = np.random.default_rng(seed)

    # Rejection sampling for density gradient
    # Proposal: Gaussian
    # Target: Gaussian * exp(gamma * x)
    # This is just a shifted Gaussian, but let's follow the spec's rejection steps if needed.
    # Actually, Gaussian * exp(gamma * x) is proportional to a shifted Gaussian.
    # f(x) ~ exp(-x^2/2s^2 + gamma*x) = exp(-(x^2 - 2s^2 gamma x)/2s^2)
    # = exp(-(x - s^2 gamma)^2 / 2s^2) * const.
    # So we can just sample from shifted Gaussian directly.

    sigma_y = 1.0
    sigma_x = sigma_ratio * sigma_y

    # Shifted mean for x due to gradient exp(gamma * z_x)
    # If z_x ~ N(0, sigma_x^2), then z_x * exp(gamma * z_x) ~ N(gamma * sigma_x^2, sigma_x^2)
    mu_x = gamma * (sigma_x**2)

    x = rng.normal(mu_x, sigma_x, n)
    y = rng.normal(0, sigma_y, n)

    z = np.column_stack((x, y))

    # Standardize to unit median radius
    radii = np.linalg.norm(z, axis=1)
    med_r = np.median(radii)
    if med_r > 0:
        z = z / med_r

    return z


def generate_geometry_crescent(
    n: int,
    phi_range: Tuple[float, float] = (0, np.pi),
    r_range: Tuple[float, float] = (1.0, 3.0),
    sigma_eta: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    Family 2: Non-convex crescent/arc.
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(phi_range[0], phi_range[1], n)
    rho = rng.uniform(r_range[0], r_range[1], n)

    z_clean = np.column_stack((rho * np.cos(phi), rho * np.sin(phi)))
    eta = rng.normal(0, sigma_eta, (n, 2))

    return z_clean + eta


def generate_geometry_peanut(n: int, separation: float = 1.5, seed: int = 0) -> np.ndarray:
    """
    Family 3: Two-lobed peanut density.
    Mixture of two anisotropic Gaussians.
    """
    rng = np.random.default_rng(seed)
    n1 = rng.binomial(n, 0.5)
    n2 = n - n1

    # Lobe 1: centered at (-sep/2, 0)
    z1 = rng.normal(0, 1, (n1, 2))
    z1[:, 0] = z1[:, 0] * 0.7 + (-separation / 2)  # narrower in x
    z1[:, 1] = z1[:, 1] * 1.2  # taller in y

    # Lobe 2: centered at (sep/2, 0)
    z2 = rng.normal(0, 1, (n2, 2))
    z2[:, 0] = z2[:, 0] * 0.7 + (separation / 2)
    z2[:, 1] = z2[:, 1] * 1.2

    z = np.vstack((z1, z2))
    rng.shuffle(z)

    # Mild nonlinear warp to remove obvious separability?
    # "apply a mild nonlinear warp"
    # Let's bend it slightly: y' = y + 0.2 * x^2
    z[:, 1] += 0.2 * (z[:, 0] ** 2)

    return z


# --- Distortion Operators ---


def apply_distortion_radial(z: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """z_i = z_i * (1 + alpha * ||z_i||)"""
    if alpha == 0:
        return z
    r = np.linalg.norm(z, axis=1, keepdims=True)
    return z * (1 + alpha * r)


def apply_distortion_shear(z: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Convert to polar, phi' = phi + beta * rho, map back."""
    if beta == 0:
        return z
    r = np.linalg.norm(z, axis=1)
    phi = np.arctan2(z[:, 1], z[:, 0])

    phi_new = phi + beta * r
    return np.column_stack((r * np.cos(phi_new), r * np.sin(phi_new)))


def apply_distortion_truncation(z: np.ndarray, c: float = 1.0, seed: int = 0) -> np.ndarray:
    """Remove points with x > c and resample to keep size fixed."""
    rng = np.random.default_rng(seed)
    mask = z[:, 0] <= c
    z_keep = z[mask]

    # Resample to restore size n
    n_target = z.shape[0]
    if len(z_keep) == 0:
        return z  # Should not happen in reasonable settings

    indices = rng.choice(len(z_keep), size=n_target, replace=True)
    return z_keep[indices]


def apply_donor_shifts(
    z: np.ndarray, donors: np.ndarray, shift_scale: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Apply random spatial shifts to each donor's cells."""
    rng = np.random.default_rng(seed)
    n_donors = donors.max() + 1
    shifts = rng.normal(0, shift_scale, (n_donors, 2))
    return z + shifts[donors]


# --- Expression Generators ---


def generate_umis(
    n: int,
    mu_u: float = 8.0,
    sigma_u: float = 0.5,
    n_donors: int = 1,
    donor_shift: float = 0.0,
    z: Optional[np.ndarray] = None,
    delta_spatial: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate library sizes and donor labels.
    If n_donors > 1, assigns donors and applies donor_shift to mu_u for half the donors.
    If z and delta_spatial are provided, induces spatial structure:
    log u_i = mu + delta_spatial * r_i + epsilon
    Returns (umis, donor_labels)
    """
    rng = np.random.default_rng(seed)

    donors = rng.integers(0, n_donors, n)

    # Donor-specific means
    mus = np.full(n, mu_u)
    if n_donors > 1 and donor_shift != 0:
        # Shift even donors up, odd donors down (or just shift half)
        is_shifted = donors % 2 == 1
        mus[is_shifted] += donor_shift

    # Spatial structure
    if z is not None and delta_spatial != 0:
        r = np.linalg.norm(z, axis=1)
        # Standardize r to have mean 0, std 1 for predictable delta effect
        r_std = (r - np.mean(r)) / (np.std(r) + 1e-8)
        mus += delta_spatial * r_std

    log_u = rng.normal(mus, sigma_u)
    umis = np.exp(log_u)

    return umis, donors


def sample_nb(mu: np.ndarray, phi: float, seed: int) -> np.ndarray:
    """
    Sample from Negative Binomial parameterized by mean mu and dispersion phi.
    Var = mu + mu^2 / phi  (or similar, depending on parameterization).
    Here we use numpy's negative_binomial(n, p).
    Mean = n(1-p)/p.
    Let's map mu, phi to n, p.
    Common param: Var = mu + alpha * mu^2. Here phi is likely 1/alpha or similar.
    Spec says "dispersion phi in {10, 50}". Usually implies size parameter n in NB(n, p).
    If n=phi, then Var = mu + mu^2/phi.
    p = n / (n + mu)
    """
    rng = np.random.default_rng(seed)
    n_param = phi
    p_param = n_param / (n_param + mu)

    # Handle mu=0 case
    # If mu is 0, p=1, result is 0.
    # p_param can be 0 if mu is inf, but mu is finite.

    return rng.negative_binomial(n_param, p_param)


def generate_expression_null_A(
    umis: np.ndarray, lambda_0: float = 0.1, phi: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Null A: Exchangeable. lambda = lambda_0 * u_i (implicitly, usually).
    Spec says: lambda_i = lambda_0 independent of z_i and u_i?
    Wait, "lambda_i = lambda_0 independent of z_i and u_i".
    If it's independent of u_i, then it's not library size normalized counts?
    "We then apply the same normalization used in the real pipeline (e.g., log1p counts-per-10k)"
    Usually raw counts depend on u_i.
    If lambda_i is truly constant, then counts ~ NB(lambda_0, phi).
    But usually in scRNA-seq, mean count is proportional to library size.
    Let's re-read carefully: "lambda_i = lambda_0 independent of z_i and u_i".
    This implies absolute expression is constant, so counts are NOT correlated with depth?
    That would be very weird for scRNA-seq.
    Null B says: "lambda_i = lambda_0 * (u_i / median(u))^kappa".
    If kappa=0, we get Null A.
    So Null A implies kappa=0.
    However, usually "null" means "no biological signal", but technical signal (depth) is always present.
    If Null A has NO depth effect, it's an idealized theoretical null.
    Let's implement generic Null with kappa.
    """
    # Implementing as generic null with kappa=0
    return generate_expression_null_B(umis, lambda_0, kappa=0.0, phi=phi, seed=seed)


def generate_expression_null_B(
    umis: np.ndarray,
    lambda_0: float = 0.1,
    kappa: float = 1.0,
    phi: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Null B: Depth-driven. lambda_i = lambda_0 * (u_i / median(u))^kappa."""
    med_u = np.median(umis)
    lam = lambda_0 * (umis / med_u) ** kappa
    return sample_nb(lam, phi, seed)


def generate_expression_null_C(
    umis: np.ndarray,
    donors: np.ndarray,
    lambda_0: float = 0.1,
    donor_effects: Optional[np.ndarray] = None,
    phi: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Null C: Donor-driven. lambda_i = lambda_0,d(i)."""
    if donor_effects is None:
        # Default: random multipliers for each donor
        rng_d = np.random.default_rng(seed)  # distinct seed for effects?
        n_donors = donors.max() + 1
        donor_effects = rng_d.lognormal(0, 0.5, n_donors)

    lam = lambda_0 * donor_effects[donors]
    # Usually still scales with library size? Spec says "independent of z_i".
    # Doesn't explicitly say independent of u_i, but Null B was the depth one.
    # Let's assume it scales with u_i as well, otherwise it's just donor batch effect.
    # "lambda_i = lambda_0,d(i) ... independent of z_i".
    # Let's assume standard depth scaling is present unless kappa is specified.
    # But let's stick to the formula: lambda_i = lambda_0,d(i).
    # If we want depth scaling, we should probably multiply by u_i/med_u.
    # Let's add depth scaling implicitly or assume lambda_0,d includes it?
    # Given Null B is "depth only", Null C is likely "donor only" or "donor + depth".
    # Let's assume it includes depth scaling (kappa=1) because that's realistic.
    med_u = np.median(umis)
    lam = lam * (umis / med_u)

    return sample_nb(lam, phi, seed)


def generate_expression_alt(
    z: np.ndarray,
    umis: np.ndarray,
    variant: str = "wedge",  # wedge, rim, bipolar
    theta_dagger: float = 0.0,
    sigma_theta: float = np.deg2rad(20),
    beta_theta: float = 1.0,
    beta_r: float = 1.0,  # usually 1 or -1 implicitly in spec, but we can make it explicit
    kappa: float = 1.0,
    lambda_0: float = 0.1,
    phi: float = 10.0,
    seed: int = 0,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unified Alternative Generator.
    log lambda = beta_0 + beta_theta * h(theta) + beta_r * q(r) + kappa * log u
    """
    # Compute theta, r relative to vantage point v (center)
    if center is None:
        center = np.zeros(2)

    z_centered = z - center
    r = np.linalg.norm(z_centered, axis=1)
    theta = np.arctan2(z_centered[:, 1], z_centered[:, 0])

    # h(theta)
    def dist_s1(t1, t2):
        d = np.abs(t1 - t2)
        return np.minimum(d, 2 * np.pi - d)

    if variant == "wedge":
        # h = exp(-0.5 * (dist/sigma)^2)
        # q = -r
        d = dist_s1(theta, theta_dagger)
        h = np.exp(-0.5 * (d / sigma_theta) ** 2)
        q = -r
    elif variant == "rim":
        # h same
        # q = +r
        d = dist_s1(theta, theta_dagger)
        h = np.exp(-0.5 * (d / sigma_theta) ** 2)
        q = r
    elif variant == "bipolar":
        # h = exp(...) - exp(... + pi)
        # q = 0 (purely angular)
        d1 = dist_s1(theta, theta_dagger)
        d2 = dist_s1(theta, theta_dagger + np.pi)
        h = np.exp(-0.5 * (d1 / sigma_theta) ** 2) - np.exp(-0.5 * (d2 / sigma_theta) ** 2)
        q = np.zeros_like(r)
    else:
        raise ValueError(f"Unknown variant {variant}")

    # log lambda
    # beta_0 = log(lambda_0)
    # log u term: kappa * log(u/med_u) to keep scale
    med_u = np.median(umis)
    log_lam = np.log(lambda_0) + beta_theta * h + beta_r * q + kappa * np.log(umis / med_u)
    lam = np.exp(log_lam)

    return sample_nb(lam, phi, seed)


# --- Core Analysis Function ---


@dataclass
class GeneResult:
    gene_id: int
    A_g: float
    theta_g: float
    P_g: float
    p_naive: float
    p_strat: float
    adequacy_fraction: float
    n_fg: int
    is_adequate: bool
    beta: float = 0.0
    theta_0: float = 0.0
    variant: str = ""


def analyze_gene(
    z: np.ndarray,
    x: np.ndarray,
    umis: np.ndarray,
    config: BioRSPConfig,
    gene_id: int = 0,
    beta: float = 0.0,
    theta_0: float = 0.0,
    variant: str = "",
    compute_stratified: bool = True,
    center: Optional[np.ndarray] = None,
) -> GeneResult:

    # 1. Compute Vantage and Angles
    if center is None:
        if config.vantage == "geometric_median":
            center, _, _ = geometric_median(
                z, tol=config.geom_median_tol, max_iter=config.geom_median_max_iter
            )
        else:
            center = np.mean(z, axis=0)

    z_centered = z - center
    r = np.linalg.norm(z_centered, axis=1)
    theta = np.arctan2(z_centered[:, 1], z_centered[:, 0])
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2pi)

    # 2. Define Foreground
    # Using quantile
    threshold = np.quantile(x, config.foreground_quantile)
    if threshold == 0:
        y = x > 0
    else:
        y = x >= threshold

    n_fg = np.sum(y)

    # Check total adequacy
    if n_fg < config.min_fg_total:
        return GeneResult(
            gene_id=gene_id,
            A_g=np.nan,
            theta_g=np.nan,
            P_g=np.nan,
            p_naive=np.nan,
            p_strat=np.nan,
            adequacy_fraction=0.0,
            n_fg=n_fg,
            is_adequate=False,
            beta=beta,
            theta_0=theta_0,
            variant=variant,
        )

    # 3. Compute Observed Radar
    # Now passing r, theta, y to compute_rsp_radar
    radar = compute_rsp_radar(
        r,
        theta,
        y,
        B=config.n_angles,
        delta_deg=config.sector_width_deg,
        min_fg_sector=config.min_fg_sector,
        min_bg_sector=config.min_bg_sector,
    )

    # Note: compute_rsp_radar now returns NaNs for inadequate sectors.
    # We do NOT set them to 0.0 anymore.

    summaries = compute_scalar_summaries(radar)

    # Adequacy fraction (fraction of sectors that are not NaN)
    adequate_mask = ~np.isnan(radar.rsp)
    adequacy_fraction = np.mean(adequate_mask)

    # 4. Compute P-values
    # 4. Compute P-values
    # Naive
    p_naive, _, _, _, _ = compute_p_value(
        r,
        theta,
        y,
        B=config.n_angles,
        delta_deg=config.sector_width_deg,
        n_perm=config.n_permutations,
        umi_counts=None,  # Naive
        seed=config.seed + gene_id,
        min_fg_sector=config.min_fg_sector,
        min_bg_sector=config.min_bg_sector,
    )

    p_strat = np.nan
    if compute_stratified:
        p_strat, _, _, _, _ = compute_p_value(
            r,
            theta,
            y,
            B=config.n_angles,
            delta_deg=config.sector_width_deg,
            n_perm=config.n_permutations,
            umi_counts=umis,  # Stratified
            seed=config.seed + gene_id,
            min_fg_sector=config.min_fg_sector,
            min_bg_sector=config.min_bg_sector,
        )

    return GeneResult(
        gene_id=gene_id,
        A_g=summaries.rms_anisotropy,
        theta_g=summaries.peak_angle,
        P_g=summaries.peak_strength,
        p_naive=p_naive,
        p_strat=p_strat,
        adequacy_fraction=adequacy_fraction,
        n_fg=n_fg,
        is_adequate=True,
        beta=beta,
        theta_0=theta_0,
        variant=variant,
    )


# --- Worker wrappers for parallel execution ---


def worker_analyze_null(args: Dict) -> Dict:
    """Worker wrapper for null gene analysis."""
    # Import here so top-level import cost is minimal and ruff doesn't complain
    from multiprocessing import shared_memory

    # Attach geometry
    if "shm_z" in args:
        info_z = args["shm_z"]
        shm_z = shared_memory.SharedMemory(name=info_z["name"])
        z = np.ndarray(tuple(info_z["shape"]), dtype=info_z["dtype"], buffer=shm_z.buf)
    else:
        z = args["z"]

    # Attach umis
    if "shm_umis" in args:
        info_u = args["shm_umis"]
        shm_u = shared_memory.SharedMemory(name=info_u["name"])
        umis = np.ndarray(tuple(info_u["shape"]), dtype=info_u["dtype"], buffer=shm_u.buf)
    else:
        umis = args["umis"]

    # Attach donors if present
    donors = args.get("donors", None)

    config = args["config"]
    gene_id = args["gene_id"]
    geometry_name = args["geometry"]
    seed = args["seed"]
    null_type = args.get("null_type", "A")

    # Wrap work in try/finally to ensure shared memory is always closed
    try:
        # Generate expression based on null type
        if null_type == "A":
            x = generate_expression_null_A(umis, seed=seed)
        elif null_type == "B":
            x = generate_expression_null_B(umis, kappa=1.0, seed=seed)
        elif null_type == "C":
            if donors is None:
                # Fallback if no donors provided
                x = generate_expression_null_B(umis, seed=seed)
            else:
                x = generate_expression_null_C(umis, donors, seed=seed)
        else:
            x = generate_expression_null_A(umis, seed=seed)

        res = analyze_gene(z, x, umis, config, gene_id=gene_id)
        d = asdict(res)
        d["geometry"] = geometry_name
        d["null_type"] = null_type
        d["seed"] = seed
        return d
    finally:
        # Close shared memory handles in child when attached
        try:
            if "shm_z" in args:
                shm_z.close()
        except Exception:
            pass
        try:
            if "shm_umis" in args:
                shm_u.close()
        except Exception:
            pass


def worker_analyze_planted(args: Dict) -> Dict:
    """Worker wrapper for planted-signal gene analysis."""
    from multiprocessing import shared_memory

    if "shm_z" in args:
        info_z = args["shm_z"]
        shm_z = shared_memory.SharedMemory(name=info_z["name"])
        z = np.ndarray(tuple(info_z["shape"]), dtype=np.dtype(info_z["dtype"]), buffer=shm_z.buf)
    else:
        z = args["z"]

    if "shm_umis" in args:
        info_u = args["shm_umis"]
        shm_u = shared_memory.SharedMemory(name=info_u["name"])
        umis = np.ndarray(tuple(info_u["shape"]), dtype=np.dtype(info_u["dtype"]), buffer=shm_u.buf)
    else:
        umis = args["umis"]

    config = args["config"]
    gene_id = args["gene_id"]
    beta = args["beta"]
    theta_0 = args["theta_0"]
    variant = args.get("variant", "wedge")
    sigma_theta = args.get("sigma_theta", np.deg2rad(20))
    seed = args["seed"]
    center = args.get("center", None)

    # Generate expression
    x = generate_expression_alt(
        z=z,
        umis=umis,
        variant=variant,
        theta_dagger=theta_0,
        sigma_theta=sigma_theta,
        beta_theta=beta,
        seed=seed,
        center=center,
    )

    try:
        res = analyze_gene(
            z,
            x,
            umis,
            config,
            gene_id=gene_id,
            beta=beta,
            theta_0=theta_0,
            variant=variant,
            center=center,
        )

        d = asdict(res)
        # Propagate meta inputs so plotting can group by them
        d["sigma_deg"] = args.get("sigma_deg", None)
        # Unique job key for checkpointing
        d["job_key"] = args.get(
            "job_key", f"{variant}_b{beta}_s{args.get('sigma_deg', '')}_g{gene_id}"
        )
        return d
    finally:
        try:
            if "shm_z" in args:
                shm_z.close()
        except Exception:
            pass
        try:
            if "shm_umis" in args:
                shm_u.close()
        except Exception:
            pass


# --- Shared-memory helpers ---


def _create_shared_arrays(z: np.ndarray, umis: np.ndarray) -> Dict:
    """Create shared memory blocks for z and umis and return info dicts and shm objects.

    Returns:
        {"shm_z": info_z, "shm_umis": info_u, "shm_objs": [shm_z, shm_u]}
    """
    from multiprocessing import shared_memory

    shm_z = shared_memory.SharedMemory(create=True, size=z.nbytes)
    arr_z = np.ndarray(z.shape, dtype=z.dtype, buffer=shm_z.buf)
    arr_z[:] = z[:]
    info_z = {"name": shm_z.name, "shape": z.shape, "dtype": str(z.dtype)}

    shm_u = shared_memory.SharedMemory(create=True, size=umis.nbytes)
    arr_u = np.ndarray(umis.shape, dtype=umis.dtype, buffer=shm_u.buf)
    arr_u[:] = umis[:]
    info_u = {"name": shm_u.name, "shape": umis.shape, "dtype": str(umis.dtype)}

    return {"shm_z": info_z, "shm_umis": info_u, "shm_objs": [shm_z, shm_u]}


def _cleanup_shared(shm_objs: List):
    for s in shm_objs:
        try:
            s.close()
            s.unlink()
        except Exception:
            pass


# --- Checkpointing & robustness helpers ---


def write_batch_to_csv(results: List[Dict], outpath: str):
    """Append a list of result dicts to CSV (creates file if missing)."""
    if not results:
        return
    df = pd.DataFrame(results)
    write_header = not os.path.exists(outpath)
    df.to_csv(outpath, mode="a", header=write_header, index=False)


def get_completed_job_keys(outpath: str) -> set:
    """Return a set of job_key strings already present in CSV (if exists)."""
    if not os.path.exists(outpath):
        return set()
    try:
        df = (
            pd.read_csv(outpath, usecols=["job_key"])
            if "job_key" in pd.read_csv(outpath, nrows=0).columns
            else None
        )
    except Exception:
        return set()

    if df is None:
        return set()
    return set(df["job_key"].dropna().astype(str).tolist())


BATCH_SIZE = 200  # number of genes/results to accumulate before flushing to disk


# --- Simulation Families ---


def run_family_1_null_calibration(
    config: SimulationConfig,
    workers: int = 1,
    executor: str = "process",
    use_shared: bool = False,
    outdir: str = ".",
    resume: bool = False,
):
    print("Running Family 1: Null Calibration (Null A, B, C)...")

    # Grid parameters
    n_sizes = [1000, 3000]  # Reduced for default run, spec says 10000 too
    if config.n_cells not in n_sizes:
        n_sizes = [config.n_cells]

    geometries = ["Elliptical", "Crescent", "Peanut"]
    null_types = ["A", "B", "C"]

    results = []

    for n in n_sizes:
        print(f"  Size n={n}")

        # Generate geometries once per size
        geom_data = {}
        geom_data["Elliptical"] = generate_geometry_elliptical(n, seed=config.seed)
        geom_data["Crescent"] = generate_geometry_crescent(n, seed=config.seed)
        geom_data["Peanut"] = generate_geometry_peanut(n, seed=config.seed)

        # Generate UMIs and Donors
        # Case 1: Balanced donors
        umis_bal, donors_bal = generate_umis(n, n_donors=4, donor_shift=0.0, seed=config.seed)
        # Case 2: Shifted donors (for Null C stress test)
        umis_shift, donors_shift = generate_umis(n, n_donors=4, donor_shift=1.0, seed=config.seed)

        for geom_name in geometries:
            z = geom_data[geom_name]

            # Generate structured UMIs for THIS geometry for Null B
            # This ensures depth confounding is aligned with the specific geometry
            umis_struct, _ = generate_umis(n, z=z, delta_spatial=1.0, seed=config.seed)

            # Generate shifted Z for Null C
            # This creates donor-structured geometry to test donor confounding
            z_shifted = apply_donor_shifts(z, donors_shift, seed=config.seed)

            for null_type in null_types:
                # Select UMI/Donor/Z set
                if null_type == "C":
                    current_z = z_shifted
                    current_umis = umis_shift
                    current_donors = donors_shift
                    scenario = f"{geom_name}_Null{null_type}_Shifted"
                elif null_type == "B":
                    # Use structured UMIs for Null B on ALL geometries
                    current_z = z
                    current_umis = umis_struct
                    current_donors = donors_bal
                    scenario = f"{geom_name}_Null{null_type}_Structured"
                else:
                    current_z = z
                    current_umis = umis_bal
                    current_donors = donors_bal
                    scenario = f"{geom_name}_Null{null_type}"

                print(f"    Scenario: {scenario}")

                # Save inputs for this scenario so plots can reconstruct signals later
                inputs_dir = os.path.join(outdir, "inputs")
                os.makedirs(inputs_dir, exist_ok=True)
                zfile = os.path.join(inputs_dir, f"{scenario}_z.npy")
                ufile = os.path.join(inputs_dir, f"{scenario}_umis.npy")
                if not os.path.exists(zfile):
                    np.save(zfile, current_z)
                if not os.path.exists(ufile):
                    np.save(ufile, current_umis)

                if use_shared and executor == "process":
                    shm_info = _create_shared_arrays(current_z, current_umis)
                    jobs = [
                        {
                            "shm_z": shm_info["shm_z"],
                            "shm_umis": shm_info["shm_umis"],
                            "donors": current_donors,
                            "config": config.to_biorsp_config(),
                            "gene_id": g,
                            "geometry": scenario,
                            "null_type": null_type,
                            "seed": config.seed + g + n,
                            "job_key": f"{scenario}_g{g}",
                        }
                        for g in range(config.n_genes)
                    ]
                else:
                    jobs = [
                        {
                            "z": current_z,
                            "umis": current_umis,
                            "donors": current_donors,
                            "config": config.to_biorsp_config(),
                            "gene_id": g,
                            "geometry": scenario,
                            "null_type": null_type,
                            "seed": config.seed + g + n,
                            "job_key": f"{scenario}_g{g}",
                        }
                        for g in range(config.n_genes)
                    ]

                outcsv = os.path.join(outdir, "family_1_null_calibration.csv")
                completed = get_completed_job_keys(outcsv) if resume else set()

                try:
                    if workers == 1:
                        for args in tqdm(jobs, desc=f"Processing {scenario}"):
                            if args.get("job_key") in completed:
                                continue
                            try:
                                res = worker_analyze_null(args)
                            except Exception as e:
                                print(f"Job failed {args.get('job_key')}: {e}")
                                # Log error to disk for auditing
                                try:
                                    with open(os.path.join(outdir, "errors.log"), "a") as fh:
                                        fh.write(f"{args.get('job_key')}\t{e}\n")
                                except Exception:
                                    pass
                                continue
                            res["n_cells"] = n
                            results.append(res)
                            if len(results) >= BATCH_SIZE:
                                write_batch_to_csv(results, outcsv)
                                results = []
                    else:
                        if executor == "thread":
                            pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
                        else:
                            pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

                        futures = [
                            pool.submit(worker_analyze_null, a)
                            for a in jobs
                            if a.get("job_key") not in completed
                        ]
                        for fut in tqdm(
                            concurrent.futures.as_completed(futures),
                            total=len(futures),
                            desc=f"Processing {scenario}",
                        ):
                            try:
                                res = fut.result()
                            except Exception as e:
                                print(f"Job failed: {e}")
                                try:
                                    with open(os.path.join(outdir, "errors.log"), "a") as fh:
                                        fh.write(f"future_job\t{e}\n")
                                except Exception:
                                    pass
                                continue

                            res["n_cells"] = n
                            results.append(res)
                            if len(results) >= BATCH_SIZE:
                                write_batch_to_csv(results, outcsv)
                                results = []
                        pool.shutdown(wait=True)
                finally:
                    # flush remaining
                    if results:
                        write_batch_to_csv(results, outcsv)
                        results = []
                    if use_shared and executor == "process":
                        _cleanup_shared(shm_info["shm_objs"])

    outpath = os.path.join(outdir, "family_1_null_calibration.csv")
    # Flush any remaining in-memory results
    if results:
        write_batch_to_csv(results, outpath)
        results = []

    df = pd.read_csv(outpath) if os.path.exists(outpath) else pd.DataFrame([])

    # Plotting: QQ Plots for Null B (Depth)
    # Filter for Null B Structured
    df_b = df[df["geometry"].str.contains("NullB_Structured")]
    if not df_b.empty:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Naive
        p_naive = df_b["p_naive"].values
        p_naive = p_naive[~np.isnan(p_naive)]
        if len(p_naive) > 0:
            expected = np.linspace(0, 1, len(p_naive))
            ax.plot(
                expected, np.sort(p_naive), label="Naive (Confounded)", color="red", linestyle="--"
            )

        # Stratified
        p_strat = df_b["p_strat"].values
        p_strat = p_strat[~np.isnan(p_strat)]
        if len(p_strat) > 0:
            expected = np.linspace(0, 1, len(p_strat))
            ax.plot(expected, np.sort(p_strat), label="Stratified (Corrected)", color="blue")

        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Expected P-value")
        ax.set_ylabel("Observed P-value")
        ax.set_title("QQ Plot: Structured Confounding (Null B')")
        ax.legend()
        plt.savefig(os.path.join(outdir, "family_1_qq_nullB_comparison.png"))

    print(f"Family 1 complete. Saved outputs to {outdir}")
    return df


def run_family_2_planted_signal(
    config: SimulationConfig,
    workers: int = 1,
    executor: str = "process",
    use_shared: bool = False,
    outdir: str = ".",
    resume: bool = False,
):
    print("Running Family 2: Power and Recovery...")

    # Fixed geometry for power analysis (usually Elliptical or Gaussian)
    n = config.n_cells
    z = generate_geometry_elliptical(n, seed=config.seed)
    umis, _ = generate_umis(n, seed=config.seed)

    # Compute center once to ensure generation and analysis use the same vantage
    biorsp_config = config.to_biorsp_config()
    if biorsp_config.vantage == "geometric_median":
        center, _, _ = geometric_median(z)
    else:
        center = np.mean(z, axis=0)

    # Save inputs for family 2
    inputs_dir = os.path.join(outdir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    zfile = os.path.join(inputs_dir, "family_2_z.npy")
    ufile = os.path.join(inputs_dir, "family_2_umis.npy")
    if not os.path.exists(zfile):
        np.save(zfile, z)
    if not os.path.exists(ufile):
        np.save(ufile, umis)

    # Grid
    variants = ["wedge", "rim", "bipolar"]
    betas = [0.5, 1.0, 1.5]
    sigmas = [10, 20, 40]  # degrees

    theta_0 = np.pi / 2  # 90 degrees

    results = []

    if use_shared and executor == "process":
        shm_info = _create_shared_arrays(z, umis)

    jobs = []
    for variant in variants:
        for beta in betas:
            for sigma_deg in sigmas:
                sigma_rad = np.deg2rad(sigma_deg)

                # Build jobs for this combination
                for g in range(config.n_genes):
                    job = {
                        "config": config.to_biorsp_config(),
                        "gene_id": g,
                        "beta": beta,
                        "theta_0": theta_0,
                        "variant": variant,
                        "sigma_theta": sigma_rad,
                        "sigma_deg": sigma_deg,
                        "seed": config.seed + g + int(beta * 100),
                        "job_key": f"{variant}_b{beta}_s{sigma_deg}_g{g}",
                        "center": center,
                    }

                    if use_shared and executor == "process":
                        job["shm_z"] = shm_info["shm_z"]
                        job["shm_umis"] = shm_info["shm_umis"]
                    else:
                        job["z"] = z
                        job["umis"] = umis

                    jobs.append(job)

    outcsv = os.path.join(outdir, "family_2_power.csv")
    completed = get_completed_job_keys(outcsv) if resume else set()

    try:
        if workers == 1:
            for args in tqdm(jobs, desc="Processing planted signal genes"):
                if args.get("job_key") in completed:
                    continue
                try:
                    res = worker_analyze_planted(args)
                except Exception as e:
                    print(f"Job failed {args.get('job_key')}: {e}")
                    continue
                results.append(res)
                if len(results) >= BATCH_SIZE:
                    write_batch_to_csv(results, outcsv)
                    results = []
        else:
            if executor == "thread":
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            else:
                pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

            futures = [
                pool.submit(worker_analyze_planted, a)
                for a in jobs
                if a.get("job_key") not in completed
            ]
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing planted signal genes",
            ):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"Job failed: {e}")
                    try:
                        with open(os.path.join(outdir, "errors.log"), "a") as fh:
                            fh.write(f"future_job\t{e}\n")
                    except Exception:
                        pass
                    continue
                results.append(res)
                if len(results) >= BATCH_SIZE:
                    write_batch_to_csv(results, outcsv)
                    results = []
            pool.shutdown(wait=True)
    finally:
        if results:
            write_batch_to_csv(results, outcsv)
            results = []
        if use_shared and executor == "process":
            _cleanup_shared(shm_info["shm_objs"])

    if use_shared and executor == "process":
        _cleanup_shared(shm_info["shm_objs"])

    outpath = os.path.join(outdir, "family_2_power.csv")
    # Flush any remaining results (already written by batches)
    if results:
        write_batch_to_csv(results, outpath)
        results = []

    df = pd.read_csv(outpath) if os.path.exists(outpath) else pd.DataFrame([])

    # Plotting: Power vs Beta for each variant
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, variant in enumerate(variants):
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue

        # Group by beta and sigma
        # We want curves for each sigma
        # TODO: ensure 'sigma' is present in result CSV for per-sigma plotting
        pass

    print(f"Family 2 complete. Saved CSV to {outdir}")
    return df


def run_family_3_robustness(
    config: SimulationConfig,
    outdir: str = ".",
    resume: bool = False,
):
    print("Running Family 3: Robustness to Distortions...")

    n = config.n_cells
    # Latent geometry
    z_latent = generate_geometry_elliptical(n, seed=config.seed)
    umis, _ = generate_umis(n, seed=config.seed)

    # Plant a signal
    theta_0 = np.pi / 2
    beta = 1.5

    # We'll simulate multiple genes to get a distribution of robustness
    n_robust_genes = 100
    results = []

    outcsv = os.path.join(outdir, "family_3_robustness.csv")
    completed = get_completed_job_keys(outcsv) if resume else set()

    for g in tqdm(range(n_robust_genes), desc="Robustness genes"):
        # Generate expression for this gene
        if f"robust_g{g}" in completed:
            continue

        x = generate_expression_alt(
            z_latent,
            umis,
            variant="wedge",
            theta_dagger=theta_0,
            beta_theta=beta,
            seed=config.seed + g,
        )

        # Transformations
        transforms = {}
        transforms["Original"] = z_latent
        transforms["RadialWarp"] = apply_distortion_radial(z_latent, alpha=0.25)
        transforms["Shear"] = apply_distortion_shear(z_latent, beta=0.5)
        # Truncation
        mask = z_latent[:, 0] <= 1.0
        transforms["Truncation"] = z_latent[mask]

        for name, z_trans in transforms.items():
            if name == "Truncation":
                z_curr = z_trans
                x_curr = x[mask]
                umis_curr = umis[mask]
            else:
                z_curr = z_trans
                x_curr = x
                umis_curr = umis

            res = analyze_gene(z_curr, x_curr, umis_curr, config.to_biorsp_config(), gene_id=g)
            d = asdict(res)
            d["transform"] = name
            d["job_key"] = f"robust_g{g}"
            results.append(d)

        if len(results) >= BATCH_SIZE:
            write_batch_to_csv(results, outcsv)
            results = []

    if results:
        write_batch_to_csv(results, outcsv)
        results = []

    df = pd.read_csv(outcsv) if os.path.exists(outcsv) else pd.DataFrame([])

    # Summary stats
    if not df.empty:
        summary = df.groupby("transform")[["A_g", "adequacy_fraction"]].mean()
        print(summary)

    print(f"Family 3 complete. Saved results to {outdir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run biorsp simulation families")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use (default: 1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (small n_genes / n_permutations)",
    )
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Executor type to use for parallel jobs (process|thread)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="sim_outputs",
        help="Directory to save outputs (plots/CSVs)",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Use shared memory for large arrays to reduce pickling overhead (process executor only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing CSV outputs and skip completed jobs",
    )

    args = parser.parse_args()

    # Configure. Debug mode must be explicitly requested via --debug (no longer using SIM_DEBUG env var).
    if args.debug:
        config = SimulationConfig()
        config.n_genes, config.n_permutations = (5, 50)
        print("DEBUG mode enabled: using small test configuration (n_genes=5, n_permutations=50)")
    else:
        config = SimulationConfig()

    # Ensure outdir exists
    os.makedirs(args.outdir, exist_ok=True)

    # Print configuration summary so the user knows what will run
    print(
        f"Configuration: n_cells={config.n_cells}, n_genes={config.n_genes}, "
        f"n_permutations={config.n_permutations}, workers={args.workers}, "
        f"executor={args.executor}, shared={args.shared}, outdir={args.outdir}"
    )

    # Run all families with requested workers
    run_family_1_null_calibration(
        config,
        workers=args.workers,
        executor=args.executor,
        use_shared=args.shared,
        outdir=args.outdir,
        resume=args.resume,
    )
    run_family_2_planted_signal(
        config,
        workers=args.workers,
        executor=args.executor,
        use_shared=args.shared,
        outdir=args.outdir,
        resume=args.resume,
    )
    df3 = run_family_3_robustness(
        config,
        outdir=args.outdir,
        resume=args.resume,
    )

    print("All simulation families complete.")

    # Basic verification of outputs
    for fname in ["family_1_null_calibration.csv", "family_2_power.csv", "family_3_robustness.csv"]:
        p = os.path.join(args.outdir, fname)
        if os.path.exists(p):
            try:
                n = len(pd.read_csv(p))
            except Exception:
                n = "(unreadable)"
            print(f"  {fname}: exists, rows={n}")
        else:
            print(f"  {fname}: MISSING")

    print(
        f"To generate publication-style figures from the CSV outputs, run:\n"
        f"  python examples/plot_simulation_results.py --indir {args.outdir} --outdir {args.outdir}/plots"
    )
