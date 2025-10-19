"""Generate 3D visualization with inline, crossline, and depth/time slices.

Shows orthogonal slices through the 3D seismic volumes in actual 3D space.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import logging

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

from . import data_loader
from . import reflectivity as refl_module
from . import wavelets


def plot_3d_volume(
    ax,
    cube,
    slice_indices,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
    cmap="seismic",
    is_categorical=False,
):
    """
    Plot three orthogonal slices in 3D space.

    Args:
        ax: Matplotlib 3D axis object
        cube: 3D data cube (ni, nj, nk)
        slice_indices: Tuple of (i, j, k) indices for slicing
        title: Plot title
        k_scale: Scaling factor for k-axis (e.g., DT for time, DZ for depth)
        k_label: Label for k-axis (e.g., "TWT", "Depth")
        k_unit: Unit string (e.g., "s", "m")
        cmap: Colormap to use
        is_categorical: If True, use discrete colormap
    """
    ni, nj, nk = cube.shape
    idx_i, idx_j, idx_k = slice_indices

    # Determine vmin/vmax and colormap
    if is_categorical:
        vmin = 0
        vmax = 3
        colors = plt.cm.tab10(np.linspace(0, 0.4, 4))
        cmap_discrete = ListedColormap(colors)
        cmap_to_use = cmap_discrete
    else:
        # Use 99.5 percentile for better dynamic range (seismogram data)
        p_i = np.percentile(np.abs(cube), 99.5)
        vmax = float(p_i)
        vmin = -vmax
        if vmax == vmin:
            vmax = vmin + 1.0
        cmap_to_use = plt.get_cmap(cmap)

    # Create coordinate grids for each slice
    # Inline slice (constant I = idx_i)
    J_inline, K_inline = np.meshgrid(np.arange(nj), np.arange(nk) * k_scale)
    I_inline = np.full_like(J_inline, idx_i, dtype=float)
    inline_data = cube[idx_i, :, :].T  # Shape: (nk, nj)

    # Crossline slice (constant J = idx_j)
    I_cross, K_cross = np.meshgrid(np.arange(ni), np.arange(nk) * k_scale)
    J_cross = np.full_like(I_cross, idx_j, dtype=float)
    cross_data = cube[:, idx_j, :].T  # Shape: (nk, ni)

    # Z-slice (constant K = idx_k)
    I_z, J_z = np.meshgrid(np.arange(ni), np.arange(nj))
    K_z = np.full_like(I_z, idx_k * k_scale, dtype=float)
    z_data = cube[:, :, idx_k].T  # Shape: (nj, ni)

    # Plot the three slices with high transparency and coarser sampling
    # Using rstride/cstride > 1 reduces density and helps with overlap visibility
    stride = 2  # Sample every 2nd point

    # Inline slice (YZ plane at x=idx_i)
    ax.plot_surface(
        I_inline,
        J_inline,
        K_inline,
        facecolors=cmap_to_use((inline_data - vmin) / (vmax - vmin)),
        shade=False,
        antialiased=True,
        rstride=stride,
        cstride=stride,
        alpha=0.6,
        edgecolor="k",  # Add thin black edges
        linewidth=0.1,
    )

    # Crossline slice (XZ plane at y=idx_j)
    ax.plot_surface(
        I_cross,
        J_cross,
        K_cross,
        facecolors=cmap_to_use((cross_data - vmin) / (vmax - vmin)),
        shade=False,
        antialiased=True,
        rstride=stride,
        cstride=stride,
        alpha=0.6,
        edgecolor="k",  # Add thin black edges
        linewidth=0.1,
    )

    # Z-slice (XY plane at z=idx_k*k_scale)
    ax.plot_surface(
        I_z,
        J_z,
        K_z,
        facecolors=cmap_to_use((z_data - vmin) / (vmax - vmin)),
        shade=False,
        antialiased=True,
        rstride=stride,
        cstride=stride,
        alpha=0.6,
        edgecolor="k",  # Add thin black edges
        linewidth=0.1,
    )

    # Set labels
    ax.set_xlabel("Inline (I)")
    ax.set_ylabel("Crossline (J)")
    if k_unit:
        ax.set_zlabel(f"{k_label} ({k_unit})")
    else:
        ax.set_zlabel(k_label)

    ax.set_title(title)

    # Set axis limits
    ax.set_xlim(0, ni - 1)
    ax.set_ylim(0, nj - 1)
    ax.set_zlim((nk - 1) * k_scale, 0)  # Inverted for seismic convention

    # Better viewing angle
    ax.view_init(elev=20, azim=45)


def impedance_to_seismogram_depth(impedance, dz, f_peak=30):
    """
    Convert depth-domain impedance cube to seismogram by computing reflectivity
    and convolving with wavelet (in depth domain).

    Args:
        impedance: 3D impedance cube in depth (ni, nj, nk)
        dz: Depth sampling interval (meters)
        f_peak: Peak frequency for Ricker wavelet (Hz)

    Returns:
        seismogram: 3D seismogram cube in depth (ni, nj, nk)
    """
    from scipy.signal import fftconvolve

    print(
        f"    Converting impedance to seismogram (f_peak={f_peak} Hz, depth domain)..."
    )

    # Compute reflectivity in depth domain
    refl = refl_module.reflectivity_from_ai(impedance)

    # Clip extreme reflectivity values
    refl = np.clip(refl, -0.99, 0.99)

    print(f"      Reflectivity range: [{refl.min():.6f}, {refl.max():.6f}]")

    # Generate wavelet - use equivalent time sampling for depth domain
    # dt_equiv ~ dz / average_velocity (rough approximation)
    dt_equiv = dz / 2500.0  # Assuming average velocity ~2500 m/s
    wavelet = wavelets.ricker_wavelet(f_peak=f_peak, dt=dt_equiv)
    print(
        f"      Wavelet: {len(wavelet)} samples at {f_peak} Hz (dt_equiv={dt_equiv:.6f}s)"
    )

    # Convolve each trace with wavelet
    ni, nj, nk = impedance.shape
    seismogram = np.zeros_like(impedance)

    for i in range(ni):
        if i % 30 == 0:
            print(f"        Progress: {i}/{ni} ({i*100//ni}%)")
        for j in range(nj):
            trace = fftconvolve(refl[i, j, :], wavelet, mode="same")
            seismogram[i, j, :] = trace

    print(f"      Seismogram range: [{seismogram.min():.6f}, {seismogram.max():.6f}]")
    return seismogram


def impedance_to_seismogram(impedance, dt, f_peak=30):
    """
    Convert impedance cube to seismogram by computing reflectivity and convolving with wavelet.

    Args:
        impedance: 3D impedance cube (ni, nj, nk)
        dt: Time sampling interval (seconds)
        f_peak: Peak frequency for Ricker wavelet (Hz)

    Returns:
        seismogram: 3D seismogram cube (ni, nj, nk)
    """
    from scipy.signal import fftconvolve

    print(f"  Converting impedance to seismogram (f_peak={f_peak} Hz)...")

    # Compute reflectivity
    refl = refl_module.reflectivity_from_ai(impedance)

    # Clip extreme reflectivity values to avoid numerical issues
    # Reflectivity should be between -1 and 1, but extreme contrasts can exceed this
    refl = np.clip(refl, -0.99, 0.99)

    print(
        f"    Reflectivity range (after clipping): [{refl.min():.6f}, {refl.max():.6f}]"
    )

    # Generate wavelet
    wavelet = wavelets.ricker_wavelet(f_peak=f_peak, dt=dt)
    print(f"    Wavelet: {len(wavelet)} samples at {f_peak} Hz")

    # Convolve each trace with wavelet
    ni, nj, nk = impedance.shape
    seismogram = np.zeros_like(impedance)

    for i in range(ni):
        if i % 30 == 0:
            print(f"      Progress: {i}/{ni} ({i*100//ni}%)")
        for j in range(nj):
            trace = fftconvolve(refl[i, j, :], wavelet, mode="same")
            seismogram[i, j, :] = trace

    print(f"    Seismogram range: [{seismogram.min():.6f}, {seismogram.max():.6f}]")
    return seismogram


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate 3D orthogonal slice visualizations"
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

    # Configuration parameters
    DATA_PATH = "."
    FILE_MAP = {
        "vp": "P-wave Velocity",
        "facies": "Facies",
    }
    GRID_SHAPE = (150, 200, 200)
    DZ = 1.0  # meters
    DT = 0.001  # seconds (1 ms)

    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Determine domain suffix for filenames and titles
    domain_suffix = "_depth" if args.domain == "depth" else "_time"
    domain_label = "Depth Domain" if args.domain == "depth" else "Time Domain"

    # Set scale and labels based on domain
    if args.domain == "depth":
        k_scale = DZ
        k_label = "Depth"
        k_unit = "m"
    else:
        k_scale = DT
        k_label = "Time"
        k_unit = "s"

    # Find cache files based on domain
    if args.domain == "time":
        # Time domain: look for seismogram caches (with "_time_" infix)
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
    else:
        # Depth domain: look for depth impedance caches
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

    # Find EI files - use single-angle files
    ei_depth_files = [
        f
        for f in os.listdir(cache_dir)
        if f.startswith("ei_depth_") and f.endswith(".npz")
    ]
    ei_time_files = [
        f
        for f in os.listdir(cache_dir)
        if f.startswith("ei_time_") and f.endswith(".npz")
    ]

    # Prioritize appropriate domain
    if args.domain == "depth" and ei_depth_files:
        ei_files = ei_depth_files
        ei_data_key = "ei_optimal"
        ei_type_str = "multi-angle depth-domain impedance (optimal stack)"
        ei_is_depth_domain = True
    elif args.domain == "time" and ei_time_files:
        ei_files = ei_time_files
        ei_data_key = "ei_seismic"
        ei_type_str = "multi-angle time-domain seismogram"
        ei_is_depth_domain = False
    elif ei_depth_files:
        ei_files = ei_depth_files
        ei_data_key = "ei_optimal"
        ei_type_str = "multi-angle depth-domain impedance (optimal stack)"
        ei_is_depth_domain = True
    else:
        ei_files = ei_time_files
        ei_data_key = "ei_seismic"
        ei_type_str = "multi-angle time-domain seismogram"
        ei_is_depth_domain = False

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

    # Load data based on selected domain
    if args.domain == "time":
        # Load time-domain seismograms
        avo_cache = np.load(avo_fn, mmap_mode="r")
        if "full_stack" in avo_cache:
            avo = avo_cache["full_stack"]
        else:
            avo = avo_cache["angle_0"]

        ai_cache = np.load(ai_fn, mmap_mode="r")
        if "seismogram_ai" in ai_cache:
            ai = ai_cache["seismogram_ai"]
        else:
            ai = ai_cache["impedance_ai"]

        ei_cache = np.load(ei_fn, mmap_mode="r")
        ei_raw = ei_cache[ei_data_key]

        print(f"Loaded data shapes:")
        print(f"  AVO: {avo.shape} (time domain)")
        print(f"  AI: {ai.shape} (time domain)")
        if ei_is_depth_domain:
            print(f"  EI: {ei_raw.shape} (depth-domain impedance)")
        else:
            print(f"  EI: {ei_raw.shape} (time-domain seismogram)")
    else:
        # Depth domain: load from cached depth files
        print("Loading depth-domain impedances from cache...")

        # Find depth domain cache files
        avo_depth_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("avo_depth_") and f.endswith(".npz")
        ]
        ai_depth_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("ai_depth_") and f.endswith(".npz")
        ]

        # Load AVO depth data
        if len(avo_depth_files) > 0:
            avo_depth_fn = os.path.join(cache_dir, sorted(avo_depth_files)[-1])
            avo_depth_cache = np.load(avo_depth_fn, mmap_mode="r")
            avo = avo_depth_cache["impedance_depth"]
            print(f"  Loaded AVO from: {os.path.basename(avo_depth_fn)}")
        else:
            # Fallback: compute from rock properties
            print("  AVO depth cache not found, computing from rock properties...")
            props_depth = data_loader.load_stanfordsix_data(
                DATA_PATH, FILE_MAP, GRID_SHAPE
            )
            vp = props_depth["vp"] * 1000.0
            rho = props_depth["rho"] * 1000.0
            avo = vp * rho

        # Load AI depth data
        if len(ai_depth_files) > 0:
            ai_depth_fn = os.path.join(cache_dir, sorted(ai_depth_files)[-1])
            ai_depth_cache = np.load(ai_depth_fn, mmap_mode="r")
            ai = ai_depth_cache["impedance_ai"]
            print(f"  Loaded AI from: {os.path.basename(ai_depth_fn)}")
        else:
            # Fallback: compute from rock properties
            print("  AI depth cache not found, computing from rock properties...")
            if "props_depth" not in locals():
                props_depth = data_loader.load_stanfordsix_data(
                    DATA_PATH, FILE_MAP, GRID_SHAPE
                )
                vp = props_depth["vp"] * 1000.0
                rho = props_depth["rho"] * 1000.0
            ai = vp * rho

        # Load EI impedance
        ei_cache = np.load(ei_fn, mmap_mode="r")
        ei_raw = ei_cache[ei_data_key]
        ei_is_depth_domain = True

        print(f"  Loaded EI from: {os.path.basename(ei_fn)}")

        print(f"\nLoaded depth-domain data shapes:")
        print(f"  AVO: {avo.shape} (depth domain impedance)")
        print(f"  AI: {ai.shape} (depth domain impedance)")
        print(f"  EI: {ei_raw.shape} (depth domain impedance)")

        # Convert depth-domain impedances to seismograms for visualization
        print("\nConverting depth-domain impedances to seismograms...")
        print("  (Computing reflectivity series in depth domain)")

        # Convert AVO impedance to seismogram
        print("  Converting AVO...")
        avo = impedance_to_seismogram_depth(avo, DZ, f_peak=26)

        # Convert AI impedance to seismogram
        print("  Converting AI...")
        ai = impedance_to_seismogram_depth(ai, DZ, f_peak=30)

        # Convert EI impedance to seismogram
        print("  Converting EI...")
        ei = impedance_to_seismogram_depth(ei_raw, DZ, f_peak=30)

        print(f"\nConverted to seismograms:")
        print(f"  AVO: {avo.shape} (depth domain seismogram)")
        print(f"  AI: {ai.shape} (depth domain seismogram)")
        print(f"  EI: {ei.shape} (depth domain seismogram)")

    # Handle EI conversion if needed (for time domain when EI is depth-domain impedance)
    if args.domain == "time" and ei_is_depth_domain:
        print(
            "Multi-angle mode: Converting depth-domain impedance to time-domain seismogram..."
        )
        # For 3D visualization, we'll work in time domain to match AVO/AI
        # Need to convert depth impedance -> time impedance -> seismogram

        # Load velocity for conversion
        print("  Loading velocity model...")
        props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
        vp_depth = props_depth["vp"] * 1000.0  # Convert km/s to m/s

        # Convert impedance from depth to time domain
        from scipy.interpolate import interp1d

        ni, nj, nz = ei_raw.shape
        target_nt = avo.shape[2]
        ei_impedance_time = np.zeros((ni, nj, target_nt))

        print(f"  Converting {ni}x{nj}x{nz} (depth) -> {ni}x{nj}x{target_nt} (time)...")
        time_axis = np.arange(target_nt) * DT
        depth_axis = np.arange(nz) * DZ

        for i in range(ni):
            if i % 30 == 0:
                print(f"    Progress: {i}/{ni} ({i*100//ni}%)")
            for j in range(nj):
                # Calculate TWT
                vp_trace = vp_depth[i, j, :]
                slowness = 1.0 / vp_trace
                one_way_time = np.cumsum(slowness * DZ)
                twt_trace = 2 * one_way_time
                twt_trace = np.concatenate([[0], twt_trace])

                data_trace = ei_raw[i, j, :]
                data_trace_padded = np.concatenate([[data_trace[0]], data_trace])

                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                ei_impedance_time[i, j, :] = interp_func(time_axis)

        # Convert impedance to seismogram
        ei = impedance_to_seismogram(ei_impedance_time, DT, f_peak=30)
    elif args.domain == "time":
        # Time domain - already seismograms (ei_raw is already a seismogram)
        ei = ei_raw
    # else: depth domain - ei was already converted above in the depth domain block

    print(f"\nFinal EI shape: {ei.shape}")

    # Load facies
    print("Loading facies...")
    if "props_depth" not in locals():
        props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
    facies_depth = props_depth["facies"]

    # Calculate slice indices (center of volume)
    ni, nj, nk = avo.shape
    slice_indices = (ni // 2, nj // 2, nk // 2)

    ni_d, nj_d, nk_d = facies_depth.shape
    slice_indices_facies = (ni_d // 2, nj_d // 2, nk_d // 2)

    # Create figure with 2×3 grid (AVO, AI, EI, Difference, Facies, and one more)
    fig = plt.figure(figsize=(30, 20))
    ncols = 3

    ei_mode_str = " (Multi-angle optimal)" if args.use_multiangle else " (Single-angle)"
    domain_label = "Depth Domain" if args.domain == "depth" else "Time Domain"
    fig.suptitle(
        f"3D Seismic Volumes: Orthogonal Slices in 3D Space ({domain_label}){ei_mode_str}",
        fontsize=18,
    )

    # Row 1: AVO, AI, EI
    # Subplot 1: AVO Full Stack
    print("Plotting AVO 3D volume...")
    ax1 = fig.add_subplot(2, ncols, 1, projection="3d")
    plot_3d_volume(
        ax1,
        avo,
        slice_indices,
        "AVO Full Stack",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )

    # Subplot 2: Acoustic Impedance
    print("Plotting AI 3D volume...")
    ax2 = fig.add_subplot(2, ncols, 2, projection="3d")
    plot_3d_volume(
        ax2,
        ai,
        slice_indices,
        "Acoustic Impedance",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )

    # Subplot 3: Elastic Impedance
    print("Plotting EI 3D volume...")
    print(
        f"  EI data for plotting - shape: {ei.shape}, range: [{ei.min():.6f}, {ei.max():.6f}]"
    )
    print(f"  For comparison - AVO range: [{avo.min():.6f}, {avo.max():.6f}]")
    print(f"  For comparison - AI range: [{ai.min():.6f}, {ai.max():.6f}]")
    ax3 = fig.add_subplot(2, ncols, 3, projection="3d")
    plot_3d_volume(
        ax3,
        ei,
        slice_indices,
        "Elastic Impedance",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )

    # Row 2: Differences and Facies
    # Subplot 4: Difference (AVO - AI)
    print("Plotting AVO-AI difference 3D volume...")
    ax4 = fig.add_subplot(2, ncols, 4, projection="3d")
    diff_avo_ai = avo - ai
    plot_3d_volume(
        ax4,
        diff_avo_ai,
        slice_indices,
        "Difference (AVO - AI)",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )

    # Subplot 5: Difference (EI - AI)
    print("Plotting EI-AI difference 3D volume...")
    ax5 = fig.add_subplot(2, ncols, 5, projection="3d")
    diff_ei_ai = ei - ai
    plot_3d_volume(
        ax5,
        diff_ei_ai,
        slice_indices,
        "Difference (EI - AI)",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )

    # Subplot 6: Facies
    print("Plotting facies 3D volume (depth domain)...")
    ax6 = fig.add_subplot(2, ncols, 6, projection="3d")
    plot_3d_volume(
        ax6,
        facies_depth,
        slice_indices_facies,
        "Facies",
        k_scale=DZ,
        k_label="Depth",
        k_unit="m",
        cmap="tab10",
        is_categorical=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Use domain suffix for filename
    outfn = os.path.join(cache_dir, f"orthogonal_slices{domain_suffix}.png")
    plt.savefig(outfn, dpi=200, facecolor="white", edgecolor="none")
    print(f"\nSaved 3D visualization to {outfn} at 200 DPI")


if __name__ == "__main__":
    main()
