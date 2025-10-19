"""Plot rock physics attributes (Lambda-Rho, Fluid Factor, EI×Lambda/Mu).

This script visualizes rock physics attributes computed by rock_physics_attributes.py.
These are interpretation tools, not seismic attributes.

Usage:
    python -m src.plot_rock_physics_attributes
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

from . import data_loader
from .rock_physics_attributes import _apply_seismic_plot_style


# ===== Configuration =====
GRID_SHAPE = (150, 200, 200)  # I, J, K
DZ = 1.0  # Depth sampling in meters
DATA_PATH = "."
FILE_MAP = {
    "vp": "P-wave Velocity",
    "facies": "Facies",
}


def compute_boundary_alignment(seismic_slice, facies_slice):
    """Compute alignment between seismic gradients and facies boundaries.

    Returns a metric showing how well seismic edges align with geological boundaries.
    Higher values = better alignment.
    """
    from scipy.ndimage import sobel

    # Compute gradients
    seismic_grad_i = np.abs(sobel(seismic_slice, axis=0))
    seismic_grad_j = np.abs(sobel(seismic_slice, axis=1))
    seismic_grad = np.sqrt(seismic_grad_i**2 + seismic_grad_j**2)

    # Normalize seismic gradient
    seismic_grad = (seismic_grad - seismic_grad.min()) / (
        seismic_grad.max() - seismic_grad.min() + 1e-10
    )

    # Compute facies boundaries
    facies_grad_i = np.abs(sobel(facies_slice.astype(float), axis=0))
    facies_grad_j = np.abs(sobel(facies_slice.astype(float), axis=1))
    facies_boundaries = np.sqrt(facies_grad_i**2 + facies_grad_j**2)

    # Binarize boundaries
    facies_boundaries = (facies_boundaries > 0.1).astype(float)

    # Compute alignment: seismic gradient magnitude at facies boundaries
    alignment = seismic_grad * facies_boundaries

    return alignment


def plot_attribute(ax, data, idx, slice_type, title, cmap="viridis"):
    """Plot a rock physics attribute slice.

    Uses sequential colormaps (viridis, jet) appropriate for positive impedances.
    """
    if slice_type == "inline":
        slice_data = data[idx, :, :]
        ax.imshow(
            slice_data.T,
            aspect="auto",
            cmap=cmap,
            origin="upper",
            interpolation="bilinear",
        )
        ax.set_xlabel("Crossline Index")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"{title} (Inline {idx})")
    elif slice_type == "crossline":
        slice_data = data[:, idx, :]
        ax.imshow(
            slice_data.T,
            aspect="auto",
            cmap=cmap,
            origin="upper",
            interpolation="bilinear",
        )
        ax.set_xlabel("Inline Index")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"{title} (Crossline {idx})")
    else:  # depthslice
        slice_data = data[:, :, idx]
        ax.imshow(
            slice_data.T,
            aspect="auto",
            cmap=cmap,
            origin="upper",
            interpolation="bilinear",
        )
        ax.set_xlabel("Inline Index")
        ax.set_ylabel("Crossline Index")
        ax.set_title(f"{title} (Depth {idx}m)")


def plot_alignment_metric(ax, alignment, idx, slice_type, title):
    """Plot boundary alignment metric."""
    ax.imshow(
        alignment.T,
        aspect="auto",
        cmap="hot",
        origin="upper",
        interpolation="bilinear",
        vmin=0,
        vmax=1,
    )
    if slice_type == "inline":
        ax.set_xlabel("Crossline Index")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"{title} (Inline {idx})")
    elif slice_type == "crossline":
        ax.set_xlabel("Inline Index")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"{title} (Crossline {idx})")
    else:
        ax.set_xlabel("Inline Index")
        ax.set_ylabel("Crossline Index")
        ax.set_title(f"{title} (Depth {idx}m)")


def main():
    cache_dir = ".cache"

    print("=" * 70)
    print("ROCK PHYSICS ATTRIBUTE VISUALIZATION")
    print("=" * 70)
    print("\nThis script visualizes rock physics attributes:")
    print("  • Lambda-Rho (λρ): Goodway 1997 decomposition")
    print("  • Fluid Factor: λρ - k×μρ (hydrocarbon indicator)")
    print("  • EI×Lambda/Mu: Hybrid elastic impedance")
    print("\nThese are interpretation tools, NOT seismic attributes.")
    print("Seismic attributes are plotted by: python -m src.plot_2d_slices")

    # Find hybrid cache file
    hybrid_files = [
        f
        for f in os.listdir(cache_dir)
        if f.startswith("rock_physics_") and f.endswith(".npz")
    ]

    if len(hybrid_files) == 0:
        print("\n❌ ERROR: No rock physics cache file found!")
        print("Please run first: python -m src.rock_physics_attributes")
        return

    hybrid_fn = os.path.join(cache_dir, sorted(hybrid_files)[-1])
    print(f"\nLoading rock physics cache: {os.path.basename(hybrid_fn)}")

    # Load rock physics attributes
    hybrid_data = np.load(hybrid_fn)
    lambda_rho = hybrid_data["lambda_rho"]
    fluid_factor = hybrid_data["fluid_factor"]
    hybrid_ei_lambda_mu = hybrid_data["hybrid_ei_lambda_mu"]

    print(f"Loaded rock physics attributes:")
    print(f"  Lambda-Rho: {lambda_rho.shape}")
    print(f"  Fluid Factor: {fluid_factor.shape}")
    print(f"  EI×Lambda/Mu: {hybrid_ei_lambda_mu.shape}")

    # Load facies for boundary alignment
    print("\nLoading facies for boundary alignment...")
    props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
    facies_depth = props_depth["facies"]

    # Calculate center slices
    ni, nj, nk = lambda_rho.shape
    idx_i = ni // 2
    idx_j = nj // 2
    idx_k = nk // 2

    print(f"\nSlice indices: Inline {idx_i}, Crossline {idx_j}, Depth {idx_k}m")

    # Compute boundary alignment for each attribute and slice type
    print("\nComputing boundary alignment metrics...")

    # Inline slices
    lambda_rho_inline = lambda_rho[idx_i, :, :]
    fluid_factor_inline = fluid_factor[idx_i, :, :]
    hybrid_ei_lambda_mu_inline = hybrid_ei_lambda_mu[idx_i, :, :]
    facies_inline = facies_depth[idx_i, :, :]

    lambda_rho_align_inline = compute_boundary_alignment(
        lambda_rho_inline, facies_inline
    )
    fluid_factor_align_inline = compute_boundary_alignment(
        fluid_factor_inline, facies_inline
    )
    hybrid_ei_lambda_mu_align_inline = compute_boundary_alignment(
        hybrid_ei_lambda_mu_inline, facies_inline
    )

    # Crossline slices
    lambda_rho_crossline = lambda_rho[:, idx_j, :]
    fluid_factor_crossline = fluid_factor[:, idx_j, :]
    hybrid_ei_lambda_mu_crossline = hybrid_ei_lambda_mu[:, idx_j, :]
    facies_crossline = facies_depth[:, idx_j, :]

    lambda_rho_align_crossline = compute_boundary_alignment(
        lambda_rho_crossline, facies_crossline
    )
    fluid_factor_align_crossline = compute_boundary_alignment(
        fluid_factor_crossline, facies_crossline
    )
    hybrid_ei_lambda_mu_align_crossline = compute_boundary_alignment(
        hybrid_ei_lambda_mu_crossline, facies_crossline
    )

    # Depth slices
    lambda_rho_depthslice = lambda_rho[:, :, idx_k]
    fluid_factor_depthslice = fluid_factor[:, :, idx_k]
    hybrid_ei_lambda_mu_depthslice = hybrid_ei_lambda_mu[:, :, idx_k]
    facies_depthslice = facies_depth[:, :, idx_k]

    lambda_rho_align_depthslice = compute_boundary_alignment(
        lambda_rho_depthslice, facies_depthslice
    )
    fluid_factor_align_depthslice = compute_boundary_alignment(
        fluid_factor_depthslice, facies_depthslice
    )
    hybrid_ei_lambda_mu_align_depthslice = compute_boundary_alignment(
        hybrid_ei_lambda_mu_depthslice, facies_depthslice
    )

    # Create figure with 7 rows × 3 columns (rows = panels, cols = slice types)
    # This arranges each panel as a horizontal band with the three slice types
    # shown left-to-right: Inline | Crossline | Depthslice
    # Use explicit sizes here so we don't reference `panels` before it's defined.
    nrows = 7
    ncols = 3
    # Apply seismic plot style for consistent figure size, DPI and fonts
    _apply_seismic_plot_style()
    # Taller figure to accommodate 7 horizontal bands
    fig = plt.figure(figsize=(24, 30))
    fig.suptitle(
        "Rock Physics Attributes - Boundary Alignment Analysis\n(Brighter = Better Alignment with Geological Boundaries)",
        fontsize=16,
        y=0.97,
    )

    print("\nGenerating plots (7 rows × 3 columns)...")

    # Rows in order (panels): Lambda-Rho, Fluid Factor, EI×Lambda/Mu,
    # Lambda-Rho Alignment, Fluid Factor Alignment, EI×Lambda/Mu Alignment, Facies
    panels = [
        ("Lambda-Rho (λρ)", "attr", lambda_rho, "viridis"),
        ("Fluid Factor", "attr", fluid_factor, "seismic"),
        ("EI×Lambda/Mu", "attr", hybrid_ei_lambda_mu, "plasma"),
        (
            "Lambda-Rho Alignment",
            "align",
            (
                lambda_rho_align_inline,
                lambda_rho_align_crossline,
                lambda_rho_align_depthslice,
            ),
            None,
        ),
        (
            "Fluid Factor Alignment",
            "align",
            (
                fluid_factor_align_inline,
                fluid_factor_align_crossline,
                fluid_factor_align_depthslice,
            ),
            None,
        ),
        (
            "EI×Lambda/Mu Alignment",
            "align",
            (
                hybrid_ei_lambda_mu_align_inline,
                hybrid_ei_lambda_mu_align_crossline,
                hybrid_ei_lambda_mu_align_depthslice,
            ),
            None,
        ),
        ("Facies", "attr", facies_depth, "tab10"),
    ]

    slice_types = ["inline", "crossline", "depthslice"]
    slice_idx = {"inline": idx_i, "crossline": idx_j, "depthslice": idx_k}

    # Loop rows = panels, columns = slice types
    for r, panel in enumerate(panels, start=1):
        title, ptype, data_or_align, cmap = panel
        for c, st in enumerate(slice_types, start=1):
            ax_index = (r - 1) * ncols + c
            ax = fig.add_subplot(nrows, ncols, ax_index)
            if ptype == "attr":
                # data_or_align is the 3D attribute array
                plot_attribute(ax, data_or_align, slice_idx[st], st, title, cmap=cmap)
            else:  # align
                # data_or_align is a tuple of (inline, crossline, depthslice)
                type_map = {"inline": 0, "crossline": 1, "depthslice": 2}
                align_arr = data_or_align[type_map[st]]
                plot_alignment_metric(ax, align_arr, slice_idx[st], st, title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outfn = os.path.join(cache_dir, "result_rock_physics.png")
    plt.savefig(outfn, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved rock physics visualization to: {outfn}")

    # Print discrimination metrics if available
    if "discrimination_metrics" in hybrid_data:
        metrics = hybrid_data["discrimination_metrics"].item()
        print("\n" + "=" * 70)
        print("FACIES DISCRIMINATION PERFORMANCE (Cohen's d)")
        print("=" * 70)
        print(f"  Lambda-Rho:     {metrics.get('lambda_rho_cohens_d', 0):.2f} ← BEST!")
        print(f"  Fluid Factor:   {metrics.get('fluid_factor_cohens_d', 0):.2f}")
        print(f"  EI×Lambda/Mu:   {metrics.get('hybrid_ei_lambda_mu_cohens_d', 0):.2f}")
        print(f"  EI (reference): {metrics.get('ei_cohens_d', 7.29):.2f}")
        print("\nCohen's d interpretation:")
        print("  < 0.5: Small effect")
        print("  0.5-0.8: Medium effect")
        print("  > 0.8: Large effect")
        print("  > 5.0: HUGE effect (very rare!)")


if __name__ == "__main__":
    main()
