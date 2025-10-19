"""Overlay facies boundaries on seismic sections for AVO, AI, and EI comparison.

This script creates visualizations showing:
1. Seismic sections with facies boundary contours overlaid
2. Side-by-side comparison of AVO, AI, and EI with same facies boundaries
3. Helps assess which technique better images geological interfaces

Usage:
    python -m src.plot_facies_overlay                        # Default: depth domain, multi-angle EI
    python -m src.plot_facies_overlay --domain depth         # Explicit depth domain
    python -m src.plot_facies_overlay --domain time          # Time domain
    python -m src.plot_facies_overlay --no-multiangle        # Use single-angle EI seismogram
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import sobel, gaussian_filter
import logging

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

from . import data_loader


def convert_depth_to_time(data_depth, vp_depth, dz, dt, is_categorical=False):
    """
    Convert depth-domain data to time domain using velocity model.
    """
    data_type = "categorical data" if is_categorical else "data"
    print(f"Converting {data_type} from depth to time domain...")
    ni, nj, nz = data_depth.shape

    # Calculate maximum time needed
    max_twt = 0
    for i in range(0, ni, 30):
        for j in range(0, nj, 40):
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt = 2 * one_way_time[-1]
            max_twt = max(max_twt, twt)

    nt = int(np.ceil(max_twt / dt)) + 1
    data_time = np.zeros((ni, nj, nt), dtype=data_depth.dtype)

    time_axis = np.arange(nt) * dt
    depth_axis = np.arange(nz) * dz

    for i in range(ni):
        for j in range(nj):
            # Calculate TWT for each depth sample
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt_trace = 2 * one_way_time
            twt_trace = np.concatenate([[0], twt_trace])
            depth_trace = np.concatenate([[0], depth_axis + dz])

            data_trace = data_depth[i, j, :]
            data_trace_padded = np.concatenate([[data_trace[0]], data_trace])

            # Interpolate from depth to time
            if is_categorical:
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=0.0,
                )
            else:
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
            data_time[i, j, :] = interp_func(time_axis)

    return data_time


def convert_time_to_depth(data_time, vp_depth, dz, dt):
    """
    Convert time-domain data to depth domain using velocity model.
    (Inverse of convert_depth_to_time)
    """
    print("Converting data from time to depth domain...")
    ni, nj, nt = data_time.shape
    nz = vp_depth.shape[2]

    data_depth = np.zeros((ni, nj, nz))
    time_axis = np.arange(nt) * dt
    depth_axis = np.arange(nz) * dz

    for i in range(ni):
        for j in range(nj):
            # Calculate TWT for each depth sample
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt_trace = 2 * one_way_time
            twt_trace = np.concatenate([[0], twt_trace])
            depth_trace = np.concatenate([[0], depth_axis + dz])

            data_trace = data_time[i, j, :]

            # Interpolate from time to depth
            interp_func = interp1d(
                time_axis,
                data_trace,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            data_depth[i, j, :] = interp_func(twt_trace[:-1])

    return data_depth


def convert_depth_to_time_with_axis(
    data_depth, vp_depth, dz, dt, target_nt, is_categorical=False
):
    """
    Convert depth-domain data to time domain using velocity model with a target time axis length.

    Args:
        data_depth: Data in depth domain (ni, nj, nz)
        vp_depth: Velocity model in depth (ni, nj, nz) in m/s
        dz: Depth sampling in meters
        dt: Time sampling in seconds
        target_nt: Target number of time samples
        is_categorical: If True, use nearest-neighbor interpolation

    Returns:
        data_time: Data in time domain (ni, nj, target_nt)
    """
    data_type = "categorical data" if is_categorical else "data"
    print(
        f"Converting {data_type} from depth to time domain (target nt={target_nt})..."
    )
    ni, nj, nz = data_depth.shape

    data_time = np.zeros((ni, nj, target_nt), dtype=data_depth.dtype)
    time_axis = np.arange(target_nt) * dt
    depth_axis = np.arange(nz) * dz

    for i in range(ni):
        for j in range(nj):
            # Calculate TWT for each depth sample
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt_trace = 2 * one_way_time
            twt_trace = np.concatenate([[0], twt_trace])
            depth_trace = np.concatenate([[0], depth_axis + dz])

            data_trace = data_depth[i, j, :]
            data_trace_padded = np.concatenate([[data_trace[0]], data_trace])

            # Interpolate from depth to time
            if is_categorical:
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=0.0,
                )
            else:
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
            data_time[i, j, :] = interp_func(time_axis)

    return data_time


def detect_facies_boundaries(facies_slice):
    """
    Detect facies boundaries using edge detection.

    Args:
        facies_slice: 2D array of facies values

    Returns:
        Binary mask of boundary locations
    """
    # Apply Sobel filter to detect edges (facies changes)
    # Smooth slightly first to reduce noise
    smoothed = gaussian_filter(facies_slice.astype(float), sigma=0.5)

    # Compute gradients in both directions
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)

    # Combine gradients
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold to get boundaries (values > 0.1 indicate facies changes)
    boundaries = gradient_magnitude > 0.1

    return boundaries


def plot_seismic_with_facies_overlay(
    ax,
    seismic_slice,
    facies_slice,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
    cmap="seismic",
    show_colorbar=True,
):
    """
    Plot seismic section with facies boundaries overlaid as contours.

    Args:
        ax: Matplotlib axis
        seismic_slice: 2D seismic amplitude slice
        facies_slice: 2D facies slice (same dimensions as seismic)
        title: Plot title
        k_scale: Scaling for vertical axis
        k_label: Label for vertical axis
        k_unit: Unit for vertical axis
        cmap: Colormap for seismic
        show_colorbar: Whether to show colorbar
    """
    nj, nk = seismic_slice.shape

    # Determine amplitude range for seismic (use 99.5 percentile)
    p = np.percentile(np.abs(seismic_slice), 99.5)
    vmax = float(p)
    vmin = -vmax
    if vmax == vmin:
        vmax = vmin + 1.0

    # Set up extent [left, right, bottom, top]
    extent = [0, nj - 1, (nk - 1) * k_scale, 0]

    # Plot seismic
    im = ax.imshow(
        seismic_slice.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=extent,
        interpolation="bilinear",
        alpha=0.9,
    )

    # Detect and overlay facies boundaries
    boundaries = detect_facies_boundaries(facies_slice)

    # Create meshgrid for contour plotting using actual facies dimensions
    nj_facies, nk_facies = facies_slice.shape
    J = np.arange(nj_facies)
    K = np.arange(nk_facies) * k_scale
    JJ, KK = np.meshgrid(J, K, indexing="ij")

    # Plot facies boundaries as contours
    # Use multiple contour levels to ensure boundaries are visible
    ax.contour(
        JJ.T,
        KK.T,
        boundaries.T,
        levels=[0.5],
        colors="lime",
        linewidths=1.5,
        linestyles="solid",
        alpha=0.8,
    )

    # Also show facies as semi-transparent contours at each facies value
    facies_levels = [0.5, 1.5, 2.5]  # Boundaries between facies 0-1, 1-2, 2-3
    ax.contour(
        JJ.T,
        KK.T,
        facies_slice.T,
        levels=facies_levels,
        colors="yellow",
        linewidths=1.0,
        linestyles="dashed",
        alpha=0.6,
    )

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Crossline (J)", fontsize=10)

    if k_unit:
        ax.set_ylabel(f"{k_label} ({k_unit})", fontsize=10)
    else:
        ax.set_ylabel(k_label, fontsize=10)

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], color="lime", linewidth=1.5, label="Facies Boundaries (detected)"
        ),
        Line2D(
            [0],
            [0],
            color="yellow",
            linewidth=1.0,
            linestyle="--",
            label="Facies Interfaces",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.8)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("Amplitude", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    return im


def plot_facies_only(
    ax,
    facies_slice,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
):
    """
    Plot facies with boundaries highlighted.
    """
    from matplotlib.colors import ListedColormap

    nj, nk = facies_slice.shape
    extent = [0, nj - 1, (nk - 1) * k_scale, 0]

    # Create discrete colormap for 4 facies
    colors = plt.cm.tab10(np.linspace(0, 0.4, 4))
    cmap_discrete = ListedColormap(colors)

    # Plot facies
    im = ax.imshow(
        facies_slice.T,
        aspect="auto",
        cmap=cmap_discrete,
        vmin=0,
        vmax=3,
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )

    # Overlay boundaries
    boundaries = detect_facies_boundaries(facies_slice)
    nj_facies, nk_facies = facies_slice.shape
    J = np.arange(nj_facies)
    K = np.arange(nk_facies) * k_scale
    JJ, KK = np.meshgrid(J, K, indexing="ij")

    ax.contour(
        JJ.T,
        KK.T,
        boundaries.T,
        levels=[0.5],
        colors="white",
        linewidths=2.0,
        linestyles="solid",
    )

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Crossline (J)", fontsize=10)

    if k_unit:
        ax.set_ylabel(f"{k_label} ({k_unit})", fontsize=10)
    else:
        ax.set_ylabel(k_label, fontsize=10)

    # Colorbar with discrete ticks
    cbar = plt.colorbar(
        im, ax=ax, ticks=[0, 1, 2, 3], boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], pad=0.01
    )
    cbar.set_label("Facies", fontsize=9)
    cbar.ax.set_yticklabels(
        ["Facies 0", "Facies 1", "Facies 2", "Facies 3"], fontsize=8
    )


def compute_boundary_alignment(seismic, facies, sigma=1.5):
    """
    Compute how well seismic reflections align with facies boundaries.
    Returns seismic gradient strength at facies boundary locations.

    Higher values (brighter) = stronger seismic response at geological interfaces
    Lower values (darker) = weaker response or no boundary present
    """
    from scipy.ndimage import gaussian_filter

    # Detect facies boundaries
    facies_boundaries = detect_facies_boundaries(facies)

    # Compute seismic gradients (reflectivity/edges) using numpy gradient
    # which preserves array shape
    seismic_smooth = gaussian_filter(seismic, sigma=sigma)

    # Compute gradients in both directions
    grad_j, grad_k = np.gradient(seismic_smooth)

    # Combine gradients (magnitude)
    seismic_grad = np.sqrt(grad_j**2 + grad_k**2)

    # Normalize gradient to [0, 1]
    seismic_grad = seismic_grad / (np.max(seismic_grad) + 1e-10)

    # Ensure shapes match
    if seismic_grad.shape != facies_boundaries.shape:
        # Crop to minimum dimensions if needed
        min_shape = tuple(
            min(s1, s2) for s1, s2 in zip(seismic_grad.shape, facies_boundaries.shape)
        )
        seismic_grad = seismic_grad[: min_shape[0], : min_shape[1]]
        facies_boundaries = facies_boundaries[: min_shape[0], : min_shape[1]]

    # Multiply by facies boundaries to show alignment
    # Result: bright where seismic has strong gradient at boundary location
    alignment = seismic_grad * facies_boundaries

    return alignment


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Overlay facies boundaries on seismic sections"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["depth", "time"],
        default="depth",
        help="Domain for visualization: 'depth' (default) or 'time'",
    )
    parser.add_argument(
        "--no-multiangle",
        action="store_true",
        help="Use single-angle EI seismogram instead of multi-angle impedance (default: use multi-angle)",
    )
    args = parser.parse_args()

    # Multi-angle is ON by default (matches modeling.py behavior)
    args.use_multiangle = not args.no_multiangle

    # Time domain cannot use multi-angle (only has seismograms)
    if args.domain == "time" and args.use_multiangle:
        print("Note: Time domain requires seismograms, disabling multi-angle")
        args.use_multiangle = False

    # Configuration
    DATA_PATH = "."
    FILE_MAP = {
        "vp": "P-wave Velocity",
        "facies": "Facies",
    }
    GRID_SHAPE = (150, 200, 200)
    DZ = 1.0  # meters
    DT = 0.001  # seconds (1 ms)

    cache_dir = ".cache"

    # Find cache files based on domain
    if args.domain == "time":
        avo_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("avo_time_") and f.endswith(".npz")
        ]
        ai_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("ai_time_") and f.endswith(".npz")
        ]
    else:  # depth domain
        avo_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("avo_depth_") and f.endswith(".npz")
        ]
        ai_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("ai_depth_") and f.endswith(".npz")
        ]

    # Load EI - prioritize domain-appropriate files
    if args.domain == "depth":
        # Depth domain: look for multi-angle EI depth files
        ei_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("ei_depth_") and f.endswith(".npz")
        ]
        ei_data_key = "ei_optimal"  # Use multi-angle optimal stack
        ei_type_str = "multi-angle depth-domain impedance (optimal stack)"
    else:
        # Time domain: look for EI time files
        ei_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("ei_time_") and f.endswith(".npz")
        ]
        # Try to use multi-angle optimal stack if available, otherwise fall back
        ei_data_key = "optimal_stack"  # Use multi-angle optimal stack (NEW)
        ei_type_str = "multi-angle time-domain seismogram (optimal stack)"

    assert len(avo_files) > 0, f"No AVO cache file found in {cache_dir}"
    assert len(ai_files) > 0, f"No AI cache file found in {cache_dir}"
    assert len(ei_files) > 0, f"No EI cache file found in {cache_dir}"

    avo_fn = os.path.join(cache_dir, sorted(avo_files)[-1])
    ai_fn = os.path.join(cache_dir, sorted(ai_files)[-1])
    ei_fn = os.path.join(cache_dir, sorted(ei_files)[-1])

    print(f"Loading cache files:")
    print(f"  AVO: {os.path.basename(avo_fn)}")
    print(f"  AI: {os.path.basename(ai_fn)}")
    print(f"  EI: {os.path.basename(ei_fn)} ({ei_type_str})")

    # Load data based on domain
    if args.domain == "time":
        avo_cache = np.load(avo_fn, mmap_mode="r")
        # Fallback: if 'full_stack' doesn't exist, use 'angle_0'
        if "full_stack" in avo_cache:
            avo = avo_cache["full_stack"]
        else:
            avo = avo_cache["angle_0"]
            print("  Note: Using angle_0 for AVO (full_stack not found)")

        ai = np.load(ai_fn, mmap_mode="r")["seismogram_ai"]
        ei_cache = np.load(ei_fn, mmap_mode="r")

        # Try to load new multi-angle optimal stack, fallback to legacy single seismogram
        if ei_data_key in ei_cache:
            ei = ei_cache[ei_data_key]
            print(f"  Note: Using NEW multi-angle optimal stack for EI")
        elif "ei_seismic" in ei_cache:
            ei = ei_cache["ei_seismic"]
            ei_type_str = "single-angle time-domain seismogram (legacy)"
            print(f"  Note: Using legacy single-angle seismogram for EI")
        else:
            raise KeyError(f"Neither '{ei_data_key}' nor 'ei_seismic' found in {ei_fn}")

        print(f"Loaded seismogram shapes (time domain):")
        print(f"  AVO: {avo.shape}")
        print(f"  AI: {ai.shape}")
        print(f"  EI: {ei.shape}")
    else:  # depth domain
        avo = np.load(avo_fn, mmap_mode="r")["impedance_depth"]
        ai = np.load(ai_fn, mmap_mode="r")["impedance_ai"]
        ei_cache = np.load(ei_fn, mmap_mode="r")
        ei = ei_cache[ei_data_key]

        print(f"Loaded impedance shapes (depth domain):")
        print(f"  AVO: {avo.shape}")
        print(f"  AI: {ai.shape}")
        print(f"  EI: {ei.shape}")

    # Load velocity model and facies (depth domain)
    print("Loading velocity model and facies...")
    props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
    vp_depth = props_depth["vp"]
    facies_depth = props_depth["facies"]

    # Convert velocity from km/s to m/s
    print(f"Vp range: {np.min(vp_depth):.2f} - {np.max(vp_depth):.2f} km/s")
    vp_depth = vp_depth * 1000.0
    print(f"Vp range (converted): {np.min(vp_depth):.0f} - {np.max(vp_depth):.0f} m/s")

    # Domain-specific processing
    if args.domain == "time":
        # Get target time axis length from loaded data
        target_nt = avo.shape[2]
        print(f"Target time samples: {target_nt}")

        # Convert facies from depth to time domain
        print("Converting facies to time domain...")
        facies_time = convert_depth_to_time_with_axis(
            facies_depth, vp_depth, DZ, DT, target_nt, is_categorical=True
        )
        facies_display = facies_time
        axis_display = np.arange(target_nt) * DT
        axis_label = "Time (s)"
    else:  # depth domain
        # Use depth domain directly
        print("Using depth domain (no conversion needed)...")
        facies_display = facies_depth
        nz = facies_depth.shape[2]
        axis_display = np.arange(nz) * DZ
        axis_label = "Depth (m)"

    # Pick center slices
    ni, nj, nk = avo.shape
    idx_i = ni // 2

    print(f"Using inline slice at I = {idx_i}")

    # Extract inline slices (constant I) - now domain-aware
    avo_slice = avo[idx_i, :, :]  # Shape: (nj, nk)
    ai_slice = ai[idx_i, :, :]
    ei_slice = ei[idx_i, :, :]
    facies_slice = facies_display[idx_i, :, :]

    # Compute boundary alignment for each method
    print("\nComputing boundary alignment metrics...")
    avo_alignment = compute_boundary_alignment(avo_slice, facies_slice)
    ai_alignment = compute_boundary_alignment(ai_slice, facies_slice)
    ei_alignment = compute_boundary_alignment(ei_slice, facies_slice)

    # Create comprehensive visualization (3x4 grid)
    fig = plt.figure(figsize=(26, 12))
    title_suffix = (
        " (Multi-angle optimal)" if args.use_multiangle else " (Single-angle)"
    )
    domain_str = "Depth Domain" if args.domain == "depth" else "Time Domain"
    fig.suptitle(
        f"Seismic-Facies Boundary Alignment Analysis ({domain_str}, Inline {idx_i}): Brighter = Better Alignment{title_suffix}",
        fontsize=16,
        y=0.995,
    )

    # Determine axis scale and label based on domain
    if args.domain == "time":
        k_scale = DT
        k_label = "Time"
        k_unit = "s"
    else:
        k_scale = DZ
        k_label = "Depth"
        k_unit = "m"

    # Row 1: Individual seismic sections with facies overlay
    ax1 = plt.subplot(3, 4, 1)
    plot_seismic_with_facies_overlay(
        ax1,
        avo_slice,
        facies_slice,
        "AVO Full Stack\n(with Facies Boundaries)",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="seismic",
        show_colorbar=True,
    )

    ax2 = plt.subplot(3, 4, 2)
    plot_seismic_with_facies_overlay(
        ax2,
        ai_slice,
        facies_slice,
        "Acoustic Impedance\n(with Facies Boundaries)",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="seismic",
        show_colorbar=True,
    )

    ax3 = plt.subplot(3, 4, 3)
    plot_seismic_with_facies_overlay(
        ax3,
        ei_slice,
        facies_slice,
        "Elastic Impedance\n(with Facies Boundaries)",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="seismic",
        show_colorbar=True,
    )

    ax4 = plt.subplot(3, 4, 4)
    # Plot facies boundaries
    plot_facies_only(
        ax4,
        facies_slice,
        "Facies Boundaries\n(Reference)",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )

    # Row 2: Boundary alignment metrics (shows how well each method detects boundaries)
    ax5 = plt.subplot(3, 4, 5)
    im5 = ax5.imshow(
        avo_alignment.T,
        aspect="auto",
        cmap="hot",
        origin="upper",
        extent=[0, nj, nk * k_scale, 0],
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax5.set_xlabel("Crossline (J)", fontsize=10)
    ax5.set_ylabel(f"{k_label} ({k_unit})", fontsize=10)
    ax5.set_title(
        "AVO Boundary Alignment\n(Bright = Strong at Boundaries)", fontsize=11, pad=10
    )
    cbar5 = plt.colorbar(im5, ax=ax5, pad=0.01)
    cbar5.set_label("Alignment Strength", fontsize=9)

    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.imshow(
        ai_alignment.T,
        aspect="auto",
        cmap="hot",
        origin="upper",
        extent=[0, nj, nk * k_scale, 0],
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax6.set_xlabel("Crossline (J)", fontsize=10)
    ax6.set_ylabel(f"{k_label} ({k_unit})", fontsize=10)
    ax6.set_title(
        "AI Boundary Alignment\n(Bright = Strong at Boundaries)", fontsize=11, pad=10
    )
    cbar6 = plt.colorbar(im6, ax=ax6, pad=0.01)
    cbar6.set_label("Alignment Strength", fontsize=9)

    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.imshow(
        ei_alignment.T,
        aspect="auto",
        cmap="hot",
        origin="upper",
        extent=[0, nj, nk * k_scale, 0],
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax7.set_xlabel("Crossline (J)", fontsize=10)
    ax7.set_ylabel(f"{k_label} ({k_unit})", fontsize=10)
    ax7.set_title(
        "EI Boundary Alignment\n(Bright = Strong at Boundaries)", fontsize=11, pad=10
    )
    cbar7 = plt.colorbar(im7, ax=ax7, pad=0.01)
    cbar7.set_label("Alignment Strength", fontsize=9)

    # Compute average alignment scores
    avo_score = (
        np.mean(avo_alignment[avo_alignment > 0]) if np.any(avo_alignment > 0) else 0
    )
    ai_score = (
        np.mean(ai_alignment[ai_alignment > 0]) if np.any(ai_alignment > 0) else 0
    )
    ei_score = (
        np.mean(ei_alignment[ei_alignment > 0]) if np.any(ei_alignment > 0) else 0
    )

    ax8 = plt.subplot(3, 4, 8)
    methods = ["AVO", "AI", "EI"]
    scores = [avo_score, ai_score, ei_score]
    colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax8.bar(methods, scores, color=colors_bar, alpha=0.7, edgecolor="black")
    ax8.set_ylabel("Average Alignment\nStrength", fontsize=10)
    ax8.set_title("Boundary Alignment\nComparison", fontsize=11, pad=10)
    ax8.set_ylim([0, max(scores) * 1.2])
    ax8.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax8.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Row 3: Different crossline slices for comparison
    idx_j_list = [nj // 4, nj // 2, 3 * nj // 4]

    for plot_idx, idx_j in enumerate(idx_j_list, start=9):
        ax = plt.subplot(3, 4, plot_idx)

        # Extract vertical trace at this crossline
        avo_trace = avo[idx_i, idx_j, :]
        ai_trace = ai[idx_i, idx_j, :]
        ei_trace = ei[idx_i, idx_j, :]
        facies_trace = facies_display[idx_i, idx_j, :]

        axis_trace = np.arange(len(avo_trace)) * k_scale

        # Normalize for plotting
        avo_norm = avo_trace / (np.max(np.abs(avo_trace)) + 1e-10)
        ai_norm = ai_trace / (np.max(np.abs(ai_trace)) + 1e-10)
        ei_norm = ei_trace / (np.max(np.abs(ei_trace)) + 1e-10)

        # Plot traces
        ax.plot(avo_norm, axis_trace, "b-", linewidth=1.5, label="AVO", alpha=0.8)
        ax.plot(ai_norm + 1.5, axis_trace, "r-", linewidth=1.5, label="AI", alpha=0.8)
        ax.plot(ei_norm + 3.0, axis_trace, "g-", linewidth=1.5, label="EI", alpha=0.8)

        # Mark facies boundaries with horizontal lines
        for k in range(1, len(facies_trace)):
            if facies_trace[k] != facies_trace[k - 1]:
                t = k * k_scale
                ax.axhline(t, color="lime", linewidth=2, linestyle="--", alpha=0.7)

        # Color background by facies
        for facies_val in range(4):
            mask = facies_trace == facies_val
            if np.any(mask):
                indices = np.where(mask)[0]
                for idx in indices:
                    t = idx * k_scale
                    ax.axhspan(
                        t,
                        t + k_scale,
                        facecolor=plt.cm.tab10(facies_val / 10),
                        alpha=0.15,
                    )

        ax.set_ylim([axis_trace[-1], 0])
        ax.set_xlim([-1, 4.5])
        ax.set_xlabel("Normalized Amplitude", fontsize=9)
        ax.set_ylabel(f"{k_label} ({k_unit})", fontsize=9)
        ax.set_title(
            f"Vertical Trace at J={idx_j}\n(Lime lines = Facies boundaries)",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save with domain-specific suffix
    domain_suffix = args.domain
    outfn = os.path.join(cache_dir, f"facies_overlay_detailed_{domain_suffix}.png")
    plt.savefig(
        outfn, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight"
    )
    print(f"\n✓ Saved facies overlay visualization to {outfn}")

    # Also create a simpler 4-panel comparison
    fig2, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig2.suptitle(
        f"Facies Boundary Overlay Comparison ({domain_str}, Inline {idx_i})",
        fontsize=14,
    )

    plot_seismic_with_facies_overlay(
        axes[0],
        avo_slice,
        facies_slice,
        "AVO with Facies Boundaries",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="seismic",
    )

    plot_seismic_with_facies_overlay(
        axes[1],
        ai_slice,
        facies_slice,
        "AI with Facies Boundaries",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="seismic",
    )

    plot_seismic_with_facies_overlay(
        axes[2],
        ei_slice,
        facies_slice,
        "EI with Facies Boundaries",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="seismic",
    )

    plot_facies_only(
        axes[3],
        facies_slice,
        "Facies Reference",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )

    plt.tight_layout()

    outfn2 = os.path.join(cache_dir, f"facies_overlay_simple_{domain_suffix}.png")
    plt.savefig(
        outfn2, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight"
    )
    print(f"✓ Saved simple overlay to {outfn2}")

    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE:")
    print("='*60}")
    print("• LIME solid lines = Detected facies boundaries (edges)")
    print("• YELLOW dashed lines = Facies interface positions")
    print("• Compare how seismic reflections align with boundaries")
    print("• Strong reflections at boundaries = good imaging")
    print("• Check if AVO or AI better resolves certain interfaces")
    print("• Vertical traces show amplitude response across facies")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
