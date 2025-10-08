# Create three independent figures (plus two for density scenarios)
# with biologically more realistic simulation patterns.
# - No captions
# - Matplotlib only, no seaborn
# - No explicit colors set (uses defaults)
# - Each chart in its own figure (no subplots)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ---------------- helpers ----------------
def style_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------- Figure 1: COVERAGE ----------------
def generate_coverage(path):
    # Two biologically plausible clusters (think cell types) - arranged in square bounds
    n1, n2 = 2400, 2100
    mu1, mu2 = np.array([0.0, 1.2]), np.array([0.0, -1.2])
    cov1 = np.array([[0.30, 0.00], [0.00, 0.30]])
    cov2 = np.array([[0.35, 0.00], [0.00, 0.35]])

    c1 = np.random.multivariate_normal(mu1, cov1, n1)
    c2 = np.random.multivariate_normal(mu2, cov2, n2)
    all_cells = np.vstack([c1, c2])

    # Simulate a "feature" with cluster-dependent probability and within-cluster spatial variation
    # Higher coverage in cluster 1, lower in cluster 2; decay with distance from respective centers
    def radial_probs(points, center, base, peak, scale):
        r = np.linalg.norm(points - center, axis=1)
        # smoothly decaying probability from center; add small floor to mimic biological dropout/false positives
        p = base + peak * np.exp(-((r / scale) ** 2))
        return np.clip(p, 0, 1)

    p1 = radial_probs(c1, mu1, base=0.05, peak=0.90, scale=0.9)  # high-coverage cluster
    p2 = radial_probs(c2, mu2, base=0.02, peak=0.08, scale=1.0)  # low-coverage cluster
    expr1 = np.random.rand(n1) < p1
    expr2 = np.random.rand(n2) < p2
    expressing = np.vstack([c1[expr1], c2[expr2]])

    # Plot: background first (all cells), then expressing cells on top
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(
        all_cells[:, 0], all_cells[:, 1], s=5, alpha=0.25, linewidths=0, color="gray"
    )
    ax.scatter(
        expressing[:, 0], expressing[:, 1], s=7, alpha=0.9, linewidths=0, color="red"
    )
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    style_axes(ax)
    plt.savefig(path, dpi=300)
    plt.close(fig)


generate_coverage("coverage.png")


# ---------------- Figure 2: DENSITY (4 bands with decreasing density) ----------------
def generate_density(path):
    # Create 4 horizontal bands with decreasing density - fit to square bounds
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Define 4 bands with different density levels and progressively darker red shades
    bands = [
        {
            "y_range": (-2.0, -1.0),
            "n_points": 4500,
            "color": "#660000",
        },  # densest = darkest red
        {"y_range": (-1.0, 0.0), "n_points": 2400, "color": "#cc0000"},
        {"y_range": (0.0, 1.0), "n_points": 1000, "color": "#ff6666"},
        {
            "y_range": (1.0, 2.0),
            "n_points": 400,
            "color": "#ffcccc",
        },  # sparsest = lightest red
    ]

    for band in bands:
        y_min, y_max = band["y_range"]
        n = band["n_points"]

        # Generate points within the band - constrained x to match y range
        x = np.random.normal(loc=0.0, scale=0.8, size=n)
        y = np.random.uniform(y_min, y_max, size=n)

        # Add some curvature
        y += 0.1 * np.tanh(0.6 * x)

        # Plot with specific red shade
        ax.scatter(x, y, s=5, alpha=0.7, linewidths=0, color=band["color"])

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    style_axes(ax)
    plt.savefig(path, dpi=300)
    plt.close(fig)


generate_density("density.png")


# ---------------- Figure 3: SHAPE (more biologically plausible shapes) ----------------
def elongated_cluster(center, n, major, minor, angle_deg):
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    D = np.diag([major, minor])
    cov = R @ D @ D @ R.T
    return np.random.multivariate_normal(center, cov, n)


def rotated_rect(center, w, h, angle_deg, n):
    cx, cy = center
    pts = (np.random.rand(n, 2) - 0.5) * np.array([w, h])
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = pts @ R.T + np.array([cx, cy])
    return pts


def branching_trajectory(root, branches, n_per_branch=220, noise=0.05):
    pts = []
    for dx, dy in branches:
        t = np.random.beta(
            1.8, 1.2, n_per_branch
        )  # more terminal cells than progenitors
        curve = np.column_stack([root[0] + dx * t, root[1] + dy * t])
        curve += noise * np.random.randn(n_per_branch, 2)
        pts.append(curve)
    return np.vstack(pts)


def generate_shape(path):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Round cluster
    pts = elongated_cluster((-2.0, 2.0), 240, 0.20, 0.20, 0)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.95, linewidths=0)

    # Elongated cluster
    pts = elongated_cluster((2.0, 2.0), 230, 0.60, 0.12, 28)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.95, linewidths=0)

    # Ring (cell cycle-like)
    theta = np.random.uniform(0, 2 * np.pi, 420)
    r = np.sqrt(np.random.uniform(0.30**2, 0.45**2, 420))
    ring = np.vstack([r * np.cos(theta) + 2.0, r * np.sin(theta) + 0.0]).T
    ax.scatter(ring[:, 0], ring[:, 1], s=6, alpha=0.95, linewidths=0)

    # Crescent (trajectory around a void)
    theta_c = np.random.uniform(-2.6, 1.6, 460)
    rc = 1.0 + 0.05 * np.random.randn(theta_c.size)
    cres = np.vstack([rc * np.cos(theta_c) - 2.0, 0.7 * rc * np.sin(theta_c) - 2.0]).T
    ax.scatter(cres[:, 0], cres[:, 1], s=6, alpha=0.95, linewidths=0)

    # Rotated rectangle (batch-like slab)
    rect = rotated_rect(center=(2.0, -2.0), w=1.2, h=0.25, angle_deg=15, n=260)
    ax.scatter(rect[:, 0], rect[:, 1], s=6, alpha=0.95, linewidths=0)

    # Branching lineage (bifurcation)
    branch_pts = branching_trajectory(
        root=(-0.2, -0.5),
        branches=[(1.2, 0.8), (1.0, -1.2), (-1.2, 0.8)],
        n_per_branch=260,
        noise=0.06,
    )
    ax.scatter(branch_pts[:, 0], branch_pts[:, 1], s=6, alpha=0.9, linewidths=0)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    style_axes(ax)
    plt.savefig(path, dpi=300)
    plt.close(fig)


generate_shape("shape.png")
