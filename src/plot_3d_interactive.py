"""Generate interactive 3D visualization with inline, crossline, and depth/time slices using Plotly.

Shows orthogonal slices through the 3D seismic volumes in actual 3D space with interactive controls.

Usage:
    python -m src.plot_3d_interactive                 # Default: depth domain
    python -m src.plot_3d_interactive --domain depth  # Explicit depth domain
    python -m src.plot_3d_interactive --domain time   # Time domain

The depth domain is the natural geological representation (200 samples at 1m spacing).
The time domain shows seismic data as typically acquired (148 samples at 1ms spacing).
"""

import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

from . import data_loader
from . import reflectivity as refl_module
from . import wavelets


def convert_time_to_depth(seismogram_time, vp_depth, dz, dt):
    """
    Convert a time-domain seismogram back to depth domain.

    Args:
        seismogram_time: 3D seismogram in time domain (ni, nj, nt)
        vp_depth: 3D P-wave velocity in depth domain (ni, nj, nz)
        dz: Vertical spacing in depth domain (meters)
        dt: Time sampling interval (seconds)

    Returns:
        3D seismogram in depth domain (ni, nj, nz)
    """
    print("Converting seismogram from time to depth domain...")
    ni, nj, nt = seismogram_time.shape
    _, _, nz = vp_depth.shape

    # Create time axis for input seismogram
    time_axis = np.arange(nt) * dt

    # Initialize output
    seismogram_depth = np.zeros((ni, nj, nz), dtype=seismogram_time.dtype)

    # Convert each trace
    for i in range(ni):
        for j in range(nj):
            # Calculate TWT at each depth sample
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt_at_depth = 2 * one_way_time

            # Interpolate seismogram from time axis to depth samples
            seism_trace = seismogram_time[i, j, :]
            interp_func = interp1d(
                time_axis,
                seism_trace,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            # Sample the seismogram at the TWT corresponding to each depth
            seismogram_depth[i, j, :] = interp_func(twt_at_depth)

    return seismogram_depth


def convert_depth_to_time_with_target(data_depth, vp_depth, dz, dt, target_nt):
    """
    Convert depth-domain data to time domain with a target number of time samples.

    Args:
        data_depth: 3D data in depth domain (ni, nj, nz)
        vp_depth: 3D P-wave velocity in depth domain (ni, nj, nz)
        dz: Vertical spacing in depth domain (meters)
        dt: Time sampling interval (seconds)
        target_nt: Target number of time samples

    Returns:
        3D data in time domain (ni, nj, target_nt)
    """
    ni, nj, nz = data_depth.shape

    # Create time axis (matching AVO/AI)
    time_axis = np.arange(target_nt) * dt

    # Initialize output
    data_time = np.zeros((ni, nj, target_nt), dtype=data_depth.dtype)

    # Convert each trace from depth to time
    for i in range(ni):
        for j in range(nj):
            # Calculate TWT for this trace
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt_trace = 2 * one_way_time
            twt_trace = np.insert(twt_trace, 0, 0)  # Add zero at start

            # Get depth-domain data trace
            data_trace = data_depth[i, j, :]
            data_trace_ext = np.append(data_trace, data_trace[-1])  # Extend by one

            # Interpolate from depth to time
            interp_func = interp1d(
                twt_trace,
                data_trace_ext,
                kind="linear",
                bounds_error=False,
                fill_value=data_trace[-1],
            )
            data_time[i, j, :] = interp_func(time_axis)

    return data_time


def impedance_to_seismogram(impedance, dt, f_peak=30):
    """
    Convert depth-domain impedance to time-domain seismogram.

    Args:
        impedance: 3D impedance data (ni, nj, nt)
        dt: Time sampling in seconds
        f_peak: Peak frequency for Ricker wavelet (Hz)

    Returns:
        3D seismogram (ni, nj, nt)
    """
    print(f"Converting impedance to seismogram (f_peak={f_peak} Hz)...")

    # Compute reflectivity from impedance
    reflectivity = refl_module.reflectivity_from_ai(impedance)
    print(f"  Reflectivity range: [{reflectivity.min():.6f}, {reflectivity.max():.6f}]")

    # Generate Ricker wavelet (with correct parameter order: f_peak, length, dt)
    wavelet = wavelets.ricker_wavelet(f_peak, length=0.128, dt=dt)
    print(f"  Wavelet: {len(wavelet)} samples at {f_peak} Hz")

    # Convolve reflectivity with wavelet
    ni, nj, nt = reflectivity.shape
    seismogram = np.zeros_like(reflectivity)

    for i in range(ni):
        for j in range(nj):
            # Convolve this trace with the wavelet
            seismogram[i, j, :] = fftconvolve(
                reflectivity[i, j, :], wavelet, mode="same"
            )

    print(f"  Seismogram range: [{seismogram.min():.6f}, {seismogram.max():.6f}]")

    return seismogram


def create_3d_volume_plotly(
    cube,
    slice_indices,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
    colorscale="RdBu",
    is_categorical=False,
    is_impedance=False,
    colorbar_x=1.02,
    colorbar_title="Amplitude",
    show_colorbar=True,
):
    """
    Create three orthogonal slices in 3D space using Plotly for interactivity.

    Args:
        cube: 3D data cube (ni, nj, nk)
        slice_indices: Tuple of (i, j, k) indices for slicing
        title: Plot title
        k_scale: Scaling factor for k-axis (e.g., DT for time, DZ for depth)
        k_label: Label for k-axis (e.g., "TWT", "Depth")
        k_unit: Unit string (e.g., "s", "m")
        colorscale: Plotly colorscale name
        is_categorical: If True, use discrete colormap
        colorbar_x: X position of colorbar
        colorbar_title: Title for colorbar

    Returns:
        list: List of plotly Surface traces
    """
    ni, nj, nk = cube.shape
    idx_i, idx_j, idx_k = slice_indices

    # Determine colorscale and range
    if is_categorical:
        # For categorical data (facies)
        colorscale = [
            [0, "rgb(31, 119, 180)"],
            [0.33, "rgb(255, 127, 14)"],
            [0.67, "rgb(44, 160, 44)"],
            [1, "rgb(214, 39, 40)"],
        ]
        cmin = 0
        cmax = 3
    elif is_impedance:
        # For impedance data - use non-symmetric range (all positive values)
        # Calculate range from the three slices that will be displayed
        slice_inline = cube[idx_i, :, :]
        slice_crossline = cube[:, idx_j, :]
        slice_k = cube[:, :, idx_k]

        # Use percentiles to avoid outliers affecting the color scale
        all_slices = np.concatenate(
            [slice_inline.ravel(), slice_crossline.ravel(), slice_k.ravel()]
        )
        cmin = float(np.percentile(all_slices, 2))  # 2nd percentile
        cmax = float(np.percentile(all_slices, 98))  # 98th percentile

        if cmax == cmin:
            cmax = cmin + 1.0

        # Use a sequential colorscale for impedance (showing variation in positive values)
        # Turbo is similar to Jet but perceptually better
        colorscale = "Turbo"
    else:
        # For seismic data - use symmetric range centered at zero
        # Use 99.5 percentile of absolute values (same as matplotlib plot_2d_slices)
        # Calculate range from the three slices that will be displayed
        slice_inline = cube[idx_i, :, :]
        slice_crossline = cube[:, idx_j, :]
        slice_k = cube[:, :, idx_k]

        # Compute 99.5th percentile of absolute values for each slice
        p_inline = np.percentile(np.abs(slice_inline), 99.5)
        p_crossline = np.percentile(np.abs(slice_crossline), 99.5)
        p_k = np.percentile(np.abs(slice_k), 99.5)

        # Use the maximum across all three slices for consistent colorscale
        vmax = max(p_inline, p_crossline, p_k)
        cmax = float(vmax)
        cmin = -cmax

        if cmax == 0:
            cmax = 1.0
            cmin = -1.0

        # Use Plotly's built-in seismic colorscale (blue-white-red)
        # RdBu_r is reversed Red-Blue, giving blue (negative) to red (positive)
        colorscale = "RdBu_r"

    traces = []

    # === Inline slice (constant I = idx_i) ===
    # Create meshgrid for inline slice
    j_range = np.arange(nj)
    k_range = np.arange(nk) * k_scale
    J_inline, K_inline = np.meshgrid(j_range, k_range)
    I_inline = np.full_like(J_inline, idx_i, dtype=float)
    inline_data = cube[idx_i, :, :].T  # Shape: (nk, nj)

    trace_inline = go.Surface(
        x=I_inline,
        y=J_inline,
        z=K_inline,
        surfacecolor=inline_data,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        cauto=False,  # Disable auto color scaling
        showscale=False,
        name=f"Inline {idx_i}",
        hovertemplate=f"I={idx_i}<br>J=%{{y}}<br>{k_label}=%{{z:.4f}}{k_unit}<br>Value=%{{surfacecolor:.3f}}<extra></extra>",
        lighting=dict(
            ambient=0.8, diffuse=0.8, specular=0.1, roughness=0.9, fresnel=0.1
        ),
        lightposition=dict(x=0, y=0, z=1000),
    )
    traces.append(trace_inline)

    # === Crossline slice (constant J = idx_j) ===
    i_range = np.arange(ni)
    I_cross, K_cross = np.meshgrid(i_range, k_range)
    J_cross = np.full_like(I_cross, idx_j, dtype=float)
    cross_data = cube[:, idx_j, :].T  # Shape: (nk, ni)

    trace_cross = go.Surface(
        x=I_cross,
        y=J_cross,
        z=K_cross,
        surfacecolor=cross_data,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        cauto=False,  # Disable auto color scaling
        showscale=False,
        name=f"Crossline {idx_j}",
        hovertemplate=f"I=%{{x}}<br>J={idx_j}<br>{k_label}=%{{z:.4f}}{k_unit}<br>Value=%{{surfacecolor:.3f}}<extra></extra>",
        lighting=dict(
            ambient=0.8, diffuse=0.8, specular=0.1, roughness=0.9, fresnel=0.1
        ),
        lightposition=dict(x=0, y=0, z=1000),
    )
    traces.append(trace_cross)

    # === Z-slice (constant K = idx_k) ===
    I_z, J_z = np.meshgrid(i_range, j_range)
    K_z = np.full_like(I_z, idx_k * k_scale, dtype=float)
    z_data = cube[:, :, idx_k].T  # Shape: (nj, ni)

    k_value = idx_k * k_scale

    # Configure colorbar
    colorbar_config = None
    if show_colorbar:
        if is_categorical:
            # For categorical data (facies), use discrete ticks
            colorbar_config = dict(
                title=dict(text=colorbar_title, side="right", font=dict(size=14)),
                len=0.45,
                x=colorbar_x,
                y=0.5,
                thickness=10,
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Facies 0", "Facies 1", "Facies 2", "Facies 3"],
                tickfont=dict(size=12),
                xpad=5,
            )
        else:
            # For continuous data (seismic), use standard colorbar
            colorbar_config = dict(
                title=dict(text=colorbar_title, side="right", font=dict(size=14)),
                len=0.45,
                x=colorbar_x,
                y=0.5,
                thickness=10,
                tickfont=dict(size=12),
                xpad=5,
            )

    trace_z = go.Surface(
        x=I_z,
        y=J_z,
        z=K_z,
        surfacecolor=z_data,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        cauto=False,  # Disable auto color scaling
        showscale=show_colorbar,
        name=f"{k_label} slice",
        hovertemplate=f"I=%{{x}}<br>J=%{{y}}<br>{k_label}={k_value:.4f}{k_unit}<br>Value=%{{surfacecolor:.3f}}<extra></extra>",
        colorbar=colorbar_config,
        lighting=dict(
            ambient=0.8, diffuse=0.8, specular=0.1, roughness=0.9, fresnel=0.1
        ),
        lightposition=dict(x=0, y=0, z=1000),
    )
    traces.append(trace_z)

    return traces


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D seismic visualization"
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
        help="Use single-angle EI (legacy) instead of multi-angle depth-domain impedance (default)",
    )
    args = parser.parse_args()

    # Invert the no-multiangle flag to get use_multiangle
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

    print(f"\n{'='*70}")
    print(f"3D Interactive Visualization - {args.domain.upper()} DOMAIN")
    if args.use_multiangle:
        print(f"Using MULTI-ANGLE depth-domain elastic impedance (optimal)")
    else:
        print(f"Using SINGLE-ANGLE elastic impedance (legacy)")
    print(f"{'='*70}\n")

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

    # Find EI files - use depth domain single-angle files
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

    # Prioritize depth domain for depth, time domain for time
    if args.domain == "depth" and ei_depth_files:
        ei_files = ei_depth_files
        ei_data_key = "ei_optimal"  # Multi-angle optimal stack
        ei_is_depth_domain = True
    elif args.domain == "time" and ei_time_files:
        ei_files = ei_time_files
        ei_data_key = "ei_seismic"
        ei_is_depth_domain = False
    elif ei_depth_files:
        ei_files = ei_depth_files
        ei_data_key = "ei_optimal"  # Multi-angle optimal stack
        ei_is_depth_domain = True
    else:
        ei_files = ei_time_files
        ei_data_key = "ei_seismic"
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
    print(f"  EI: {os.path.basename(ei_fn)}")

    # Load data based on selected domain
    if args.domain == "time":
        # Load time-domain seismograms for AVO and AI
        # Note: AVO depth cache doesn't have full_stack, use angle_0
        avo_cache = np.load(avo_fn, mmap_mode="r")
        if "full_stack" in avo_cache:
            avo = avo_cache["full_stack"]
        else:
            avo = avo_cache["angle_0"]  # Use near angle if no full_stack

        ai_cache = np.load(ai_fn, mmap_mode="r")
        if "seismogram_ai" in ai_cache:
            ai = ai_cache["seismogram_ai"]
        else:
            ai = ai_cache["impedance_ai"]  # Depth domain fallback

        print(f"Loaded shapes:")
        print(f"  AVO: {avo.shape} (time domain)")
        print(f"  AI: {ai.shape} (time domain)")
    else:
        # Load depth-domain impedances for AVO and AI
        avo_cache = np.load(avo_fn, mmap_mode="r")
        if "impedance_depth" in avo_cache:
            avo = avo_cache["impedance_depth"]
        else:
            avo = avo_cache["angle_0"]

        ai_cache = np.load(ai_fn, mmap_mode="r")
        ai = ai_cache["impedance_ai"]

        print(f"Loaded shapes:")
        print(f"  AVO: {avo.shape} (depth domain impedance)")
        print(f"  AI: {ai.shape} (depth domain impedance)")

    # Load EI
    ei_cache = np.load(ei_fn, mmap_mode="r")
    ei_raw = ei_cache[ei_data_key]

    print(
        f"  EI: {ei_raw.shape} ({'depth domain impedance' if ei_is_depth_domain else 'time domain seismogram'})"
    )

    # Handle EI conversion based on domain
    if args.domain == "time" and ei_is_depth_domain:
        print(f"\nMulti-angle mode: Converting depth-domain impedance to seismogram...")

        # Load velocity model and facies for conversion
        DATA_PATH = "."
        FILE_MAP = {"vp": "P-wave Velocity", "facies": "Facies"}
        props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
        # Convert velocity from km/s to m/s (Stanford VI-E data is in km/s)
        props_depth["vp"] = props_depth["vp"] * 1000.0
        vp_depth = props_depth["vp"]

        # Get target time samples from AVO
        target_nt = avo.shape[2]

        # Convert depth impedance to time impedance
        print(
            f"Converting {ei_raw.shape[0]}x{ei_raw.shape[1]}x{ei_raw.shape[2]} (depth) → {ei_raw.shape[0]}x{ei_raw.shape[1]}x{target_nt} (time)..."
        )
        ei_impedance_time = convert_depth_to_time_with_target(
            ei_raw, vp_depth, DZ, DT, target_nt
        )

        # Convert time-domain impedance to seismogram
        ei = impedance_to_seismogram(ei_impedance_time, DT, f_peak=30)
        print(f"\nFinal EI seismogram shape: {ei.shape}")
    elif args.domain == "depth":
        # In depth domain, use impedance directly (no conversion needed)
        ei = ei_raw
        print(f"\nUsing EI depth-domain impedance directly: {ei.shape}")
    else:
        ei = ei_raw

    # Resample EI to match AVO/AI grid if needed
    if ei.shape != avo.shape:
        print(f"\n⚠ EI shape mismatch - resampling EI to match AVO/AI...")

        ni, nj, nt_avo = avo.shape
        _, _, nt_ei = ei.shape

        # Create time axes
        time_axis_ei = np.arange(nt_ei) * DT
        time_axis_avo = np.arange(nt_avo) * DT

        # Resample each trace
        ei_resampled = np.zeros((ni, nj, nt_avo), dtype=ei.dtype)
        for i in range(ni):
            if i % 30 == 0:
                print(f"  Progress: {i}/{ni} ({100*i/ni:.0f}%)")
            for j in range(nj):
                interp_func = interp1d(
                    time_axis_ei,
                    ei[i, j, :],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                ei_resampled[i, j, :] = interp_func(time_axis_avo)

        ei = ei_resampled
        print(f"✓ EI resampled to: {ei.shape}")

    # Load facies and velocity if not already loaded
    if "props_depth" not in locals():
        # Need to load now
        print("Loading facies and velocity...")
        DATA_PATH = "."
        FILE_MAP = {"vp": "P-wave Velocity", "facies": "Facies"}
        props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
        # Convert velocity from km/s to m/s (Stanford VI-E data is in km/s)
        props_depth["vp"] = props_depth["vp"] * 1000.0
    else:
        print("Using previously loaded facies and velocity...")

    facies_depth = props_depth["facies"]
    vp = props_depth["vp"]  # Already converted to m/s

    # Process data based on selected domain
    if args.domain == "depth":
        print("\n=== Using DEPTH domain (converting impedances to seismograms) ===")

        # Convert depth-domain impedances to seismograms for visualization
        # This is necessary because cache files store impedances, but we want to display seismograms
        # Use DZ / 2000.0 as approximate dt for depth domain (same as plot_2d_slices.py)
        print("Converting impedances to seismograms...")
        dt_approx = DZ / 2000.0  # Approximate time sampling for depth domain

        avo_display = impedance_to_seismogram(avo, dt_approx, f_peak=30)
        ai_display = impedance_to_seismogram(ai, dt_approx, f_peak=30)
        ei_display = impedance_to_seismogram(ei, dt_approx, f_peak=30)

        # Use depth-domain facies directly
        facies_display = facies_depth

        # Set depth domain parameters
        k_scale = DZ  # 1 meter per sample
        k_label = "Depth"
        k_unit = "m"

        # Calculate slice indices (center of volume)
        ni, nj, nk = avo_display.shape
        slice_indices = (ni // 2, nj // 2, nk // 2)

        print(f"Depth domain seismogram shape: {avo_display.shape}")
        print(f"\nData ranges for plotting (converted to seismograms):")
        print(f"  AVO: [{avo_display.min():.6f}, {avo_display.max():.6f}]")
        print(f"  AI: [{ai_display.min():.6f}, {ai_display.max():.6f}]")
        print(f"  EI: [{ei_display.min():.6f}, {ei_display.max():.6f}]")

    else:  # args.domain == "time"
        print("\n=== Using TIME domain (seismograms) ===")
        # Use time-domain seismograms directly
        avo_display = avo
        ai_display = ai
        ei_display = ei

        # Convert facies from depth to time domain
        print("Converting facies from depth to time...")
        vp = props_depth["vp"]

        # Create TWT grid
        nk_depth = facies_depth.shape[2]
        depth_samples = np.arange(nk_depth) * DZ

        # Initialize time-domain facies
        ni, nj, nk_time = avo.shape
        facies_display = np.zeros((ni, nj, nk_time), dtype=facies_depth.dtype)

        # Convert each trace
        for i in range(ni):
            for j in range(nj):
                # Calculate TWT from velocity
                vp_trace = vp[i, j, :]
                dt_samples = 2 * DZ / vp_trace  # Two-way time per sample
                twt = np.cumsum(dt_samples)
                twt = np.insert(twt, 0, 0)  # Add zero at start

                # Interpolate facies to time domain (nearest neighbor)
                time_samples = np.arange(nk_time) * DT
                facies_interp = interp1d(
                    twt,
                    np.append(facies_depth[i, j, :], facies_depth[i, j, -1]),
                    kind="nearest",
                    bounds_error=False,
                    fill_value=facies_depth[i, j, -1],
                )
                facies_display[i, j, :] = facies_interp(time_samples)

        # Set time domain parameters
        k_scale = DT
        k_label = "Time"
        k_unit = "s"

        # Calculate slice indices (center of volume)
        ni, nj, nk = avo_display.shape
        slice_indices = (ni // 2, nj // 2, nk // 2)

        print(f"Facies converted to time domain: {facies_display.shape}")

    # Debug: Check amplitude ranges before plotting
    print(f"\nData ranges for plotting ({args.domain} domain):")
    print(f"  AVO: [{avo_display.min():.6f}, {avo_display.max():.6f}]")
    print(f"  AI: [{ai_display.min():.6f}, {ai_display.max():.6f}]")
    print(f"  EI: [{ei_display.min():.6f}, {ei_display.max():.6f}]")

    # Create subplots with 2x3 layout (tight spacing)
    print("Creating interactive 3D visualization...")

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
            [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
        ],
        subplot_titles=(
            "Full-Stack Seismogram",
            "AI Synthetic Seismogram",
            "EI Synthetic Seismogram",
            "Full-Stack vs AI Difference",
            "Full-Stack vs EI Difference",
            "Facies",
        ),
        vertical_spacing=0.02,
        horizontal_spacing=0.01,
    )

    # Subplot 1: Full-Stack Seismogram (top-left)
    print("Adding Full-Stack traces...")
    avo_traces = create_3d_volume_plotly(
        avo_display,
        slice_indices,
        "Full-Stack Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        is_impedance=False,  # Always seismogram data
        # colorscale is auto-generated inside the function
        colorbar_x=0.32,  # Position for left column
        colorbar_title="Seismic Amplitude",
    )
    for trace in avo_traces:
        fig.add_trace(trace, row=1, col=1)

    # Subplot 2: AI Synthetic Seismogram (top-center) - hide colorbar
    print("Adding AI Synthetic traces...")
    ai_traces = create_3d_volume_plotly(
        ai_display,
        slice_indices,
        "AI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        is_impedance=False,  # Always seismogram data
        # colorscale is auto-generated inside the function
        show_colorbar=False,  # Hide - same scale as Full-Stack
    )
    for trace in ai_traces:
        fig.add_trace(trace, row=1, col=2)

    # Subplot 3: EI Synthetic Seismogram (top-right) - hide colorbar
    print("Adding EI Synthetic traces...")
    ei_traces = create_3d_volume_plotly(
        ei_display,
        slice_indices,
        "EI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        is_impedance=False,  # Always seismogram data
        # colorscale is auto-generated inside the function
        show_colorbar=False,  # Hide - same scale as Full-Stack
    )
    for trace in ei_traces:
        fig.add_trace(trace, row=1, col=3)

    # Subplot 4: Difference Full-Stack vs AI Synthetic (bottom-left) - hide colorbar
    print("Adding Full-Stack vs AI Synthetic difference traces...")
    diff_avo_ai = avo_display - ai_display
    diff_avo_ai_traces = create_3d_volume_plotly(
        diff_avo_ai,
        slice_indices,
        "Full-Stack vs AI Difference",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        is_impedance=False,  # Difference can be positive or negative
        # colorscale is auto-generated inside the function
        show_colorbar=False,  # Hide - same scale as Full-Stack
    )
    for trace in diff_avo_ai_traces:
        fig.add_trace(trace, row=2, col=1)

    # Subplot 5: Difference Full-Stack vs EI Synthetic (bottom-center) - hide colorbar
    print("Adding Full-Stack vs EI Synthetic difference traces...")
    diff_avo_ei = avo_display - ei_display
    diff_avo_ei_traces = create_3d_volume_plotly(
        diff_avo_ei,
        slice_indices,
        "Full-Stack vs EI Difference",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        is_impedance=False,  # Difference can be positive or negative
        # colorscale is auto-generated inside the function
        show_colorbar=False,  # Hide - same scale as Full-Stack
    )
    for trace in diff_avo_ei_traces:
        fig.add_trace(trace, row=2, col=2)

    # Subplot 6: Facies (bottom-right)
    print("Adding facies traces...")
    facies_traces = create_3d_volume_plotly(
        facies_display,
        slice_indices,
        "Facies",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        colorscale="Viridis",
        is_categorical=True,
        colorbar_x=0.99,
        colorbar_title="Facies Type",
    )
    for trace in facies_traces:
        fig.add_trace(trace, row=2, col=3)

    # Update layout for 2x3 grid visualization
    domain_label = "Time Domain" if args.domain == "time" else "Depth Domain"
    fig.update_layout(
        title=dict(
            text=f"Interactive 3D Seismic Volumes: Full-Stack vs AI Synthetic vs EI Synthetic ({domain_label})",
            font=dict(size=15, family="Arial, sans-serif", color="#333"),
            x=0.5,
            xanchor="center",
            y=0.995,
            yanchor="top",
        ),
        height=800,
        width=1600,  # Compact width to fit without scrolling
        showlegend=False,
        margin=dict(l=2, r=2, t=35, b=2),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Update subplot title font size and position
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14, family="Arial, sans-serif", color="#444")
        annotation["y"] = annotation["y"] - 0.015  # Move titles very close to plots

    # Update all 3D scene properties with appropriate z-axis label
    z_axis_title = f"{k_label} ({k_unit})"
    scene_dict = dict(
        xaxis=dict(title="Inline (I)", showbackground=True),
        yaxis=dict(title="Crossline (J)", showbackground=True),
        zaxis=dict(title=z_axis_title, autorange="reversed", showbackground=True),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3),
            center=dict(x=0, y=0, z=0),
        ),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1.33, z=0.5),
    )

    # Apply same scene settings to all subplots (2x3 grid)
    fig.update_scenes(scene_dict, row=1, col=1)
    fig.update_scenes(scene_dict, row=1, col=2)
    fig.update_scenes(scene_dict, row=1, col=3)
    fig.update_scenes(scene_dict, row=2, col=1)
    fig.update_scenes(scene_dict, row=2, col=2)
    fig.update_scenes(scene_dict, row=2, col=3)

    # Update scene domains for 2x3 grid layout with no horizontal spacing
    # Each column gets exactly 1/3 of width, each row gets 1/2 of height
    # No horizontal margins - plots are adjacent
    margin_x = 0.0  # No horizontal margin - plots touch
    margin_y = 0.01  # Minimal vertical margin between rows
    col_width = 1.0 / 3  # Exact third of width
    row_height = (1.0 - margin_y) / 2

    fig.update_layout(
        # Row 1 (top): AVO, AI, EI - exact thirds with no horizontal gaps
        scene=dict(domain=dict(x=[0.0, col_width], y=[0.5 + margin_y, 1.0])),
        scene2=dict(
            domain=dict(
                x=[col_width, 2 * col_width],
                y=[0.5 + margin_y, 1.0],
            )
        ),
        scene3=dict(domain=dict(x=[2 * col_width, 1.0], y=[0.5 + margin_y, 1.0])),
        # Row 2 (bottom): Differences and Facies - exact thirds with no horizontal gaps
        scene4=dict(domain=dict(x=[0.0, col_width], y=[0.0, 0.5])),
        scene5=dict(domain=dict(x=[col_width, 2 * col_width], y=[0.0, 0.5])),
        scene6=dict(domain=dict(x=[2 * col_width, 1.0], y=[0.0, 0.5])),
    )

    # Add domain suffix to filenames
    domain_suffix = "_depth" if args.domain == "depth" else "_time"

    # Save as HTML for full interactivity
    html_fn = os.path.join(cache_dir, f"seismic_viewer{domain_suffix}.html")
    fig.write_html(html_fn)
    print(f"\n✓ Saved interactive HTML to {html_fn}")

    # Also save as static PNG (optional, requires kaleido package)
    png_fn = os.path.join(cache_dir, f"seismic_viewer{domain_suffix}_preview.png")
    try:
        fig.write_image(png_fn, width=1600, height=800, scale=2)
        print(f"✓ Saved static PNG to {png_fn}")
    except (ValueError, ImportError) as e:
        print(f"ℹ Static PNG export skipped (install kaleido for this feature)")

    print(f"\nOpen {html_fn} in your web browser for full interactivity!")
    print("  - Rotate: Click and drag")
    print("  - Zoom: Scroll or pinch")
    print("  - Pan: Right-click and drag")
    print("  - Hover: See data values")


if __name__ == "__main__":
    main()
