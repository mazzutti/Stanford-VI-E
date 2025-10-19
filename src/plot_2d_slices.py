"""Small helper to load cached AVO and AI results and write a lightweight PNG.

This avoids re-running the heavy modeling and keeps memory usage low.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts and suppress font warnings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

# Enable anti-aliasing for smoother plots
plt.rcParams["image.interpolation"] = "bilinear"
plt.rcParams["image.resample"] = True
plt.rcParams["image.composite_image"] = True

from . import visualization
from . import data_loader
from . import reflectivity as refl_module
from . import wavelets


def convert_time_to_depth(seismogram_time, vp_depth, dz, dt, is_categorical=False):
    """
    Convert a time-domain seismogram back to depth domain.

    Args:
        seismogram_time (np.ndarray): 3D seismogram in time domain (ni, nj, nt).
        vp_depth (np.ndarray): 3D P-wave velocity in depth domain (ni, nj, nz).
        dz (float): Vertical spacing in depth domain (meters).
        dt (float): Time sampling interval (seconds).
        is_categorical (bool): If True, use nearest-neighbor for categorical data.

    Returns:
        np.ndarray: 3D seismogram in depth domain (ni, nj, nz).
    """
    data_type = "categorical data" if is_categorical else "seismogram"
    print(f"Converting {data_type} from time to depth domain...")
    ni, nj, nt = seismogram_time.shape
    _, _, nz = vp_depth.shape

    # Create depth axis
    depth_axis = np.arange(nz) * dz

    # Create time axis
    time_axis = np.arange(nt) * dt

    # Initialize output
    seismogram_depth = np.zeros((ni, nj, nz), dtype=seismogram_time.dtype)

    # Convert depth to TWT for each trace
    for i in range(ni):
        for j in range(nj):
            # Calculate TWT for this trace
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt_trace = 2 * one_way_time

            # Interpolate seismogram from time to depth
            seism_trace = seismogram_time[i, j, :]
            if is_categorical:
                # Use nearest-neighbor for categorical data (facies)
                interp_func = interp1d(
                    time_axis,
                    seism_trace,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=0.0,
                )
            else:
                # Use linear interpolation for continuous data (seismic)
                interp_func = interp1d(
                    time_axis,
                    seism_trace,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
            seismogram_depth[i, j, :] = interp_func(twt_trace)

    return seismogram_depth


def convert_depth_to_time(data_depth, vp_depth, dz, dt, is_categorical=False):
    """
    Convert depth-domain data to time domain using velocity model.

    Args:
        data_depth (np.ndarray): 3D data cube in depth domain (ni, nj, nz).
        vp_depth (np.ndarray): 3D P-wave velocity in depth domain (ni, nj, nz).
        dz (float): Vertical spacing in depth domain (meters).
        dt (float): Time sampling interval (seconds).
        is_categorical (bool): If True, use nearest-neighbor for categorical data.

    Returns:
        np.ndarray: 3D data cube in time domain.
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
            twt_trace = np.concatenate([[0], twt_trace])  # Add time=0 at surface
            depth_trace = np.concatenate([[0], depth_axis + dz])

            data_trace = data_depth[i, j, :]
            data_trace_padded = np.concatenate([[data_trace[0]], data_trace])

            # Interpolate from depth to time
            if is_categorical:
                # Use nearest-neighbor for categorical data
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=0.0,
                )
            else:
                # Use linear for continuous data
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
            data_time[i, j, :] = interp_func(time_axis)

    return data_time


def convert_depth_to_time_with_target(
    data_depth, vp_depth, dz, dt, target_nt, is_categorical=False
):
    """
    Convert depth-domain data to time domain with a specific target time axis length.

    Args:
        data_depth (np.ndarray): 3D data cube in depth domain (ni, nj, nz).
        vp_depth (np.ndarray): 3D P-wave velocity in depth domain (ni, nj, nz) in m/s.
        dz (float): Vertical spacing in depth domain (meters).
        dt (float): Time sampling interval (seconds).
        target_nt (int): Target number of time samples.
        is_categorical (bool): If True, use nearest-neighbor for categorical data.

    Returns:
        np.ndarray: 3D data cube in time domain (ni, nj, target_nt).
    """
    data_type = "categorical data" if is_categorical else "data"
    print(
        f"Converting {data_type} from depth to time domain (target nt={target_nt})..."
    )
    ni, nj, nz = data_depth.shape

    # Create output array with target time samples
    data_time = np.zeros((ni, nj, target_nt), dtype=data_depth.dtype)

    # Create depth axis
    depth_axis = np.arange(nz) * dz

    # Create target time axis
    time_axis = np.arange(target_nt) * dt

    # Convert depth to time for each trace
    for i in range(ni):
        for j in range(nj):
            # Calculate TWT for this trace
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
                # Use nearest-neighbor for categorical data (facies)
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=0.0,
                )
            else:
                # Use linear for continuous data
                interp_func = interp1d(
                    twt_trace,
                    data_trace_padded,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
            data_time[i, j, :] = interp_func(time_axis)

    return data_time


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

    print(f"Converting impedance to seismogram (f_peak={f_peak} Hz)...")

    # Compute reflectivity
    refl = refl_module.reflectivity_from_ai(impedance)
    print(f"  Reflectivity range: [{refl.min():.6f}, {refl.max():.6f}]")

    # Generate wavelet
    wavelet = wavelets.ricker_wavelet(f_peak=f_peak, dt=dt)
    print(f"  Wavelet: {len(wavelet)} samples at {f_peak} Hz")

    # Convolve each trace with wavelet
    ni, nj, nk = impedance.shape
    seismogram = np.zeros_like(impedance)

    for i in range(ni):
        if i % 30 == 0:
            print(f"    Progress: {i}/{ni} ({i*100//ni}%)")
        for j in range(nj):
            trace = fftconvolve(refl[i, j, :], wavelet, mode="same")
            seismogram[i, j, :] = trace

    print(f"  Seismogram range: [{seismogram.min():.6f}, {seismogram.max():.6f}]")
    return seismogram


def plot_with_units(
    ax,
    cube,
    slice_idx,
    slice_orientation,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
    cmap="seismic",
    is_categorical=False,
):
    """
    Plot 2D slices with proper axis scaling and units.

    Args:
        ax: Matplotlib axis object
        cube: 3D data cube
        slice_idx: Integer index for slicing
        slice_orientation: 'inline' (I), 'crossline' (J), or 'timeslice'/'depthslice' (K)
        title: Plot title
        k_scale: Scaling factor for k-axis (e.g., DT for time, DZ for depth)
        k_label: Label for k-axis (e.g., "TWT", "Depth")
        k_unit: Unit string (e.g., "s", "m")
        cmap: Colormap to use (default: "seismic")
        is_categorical: If True, use discrete colormap for categorical data
    """
    ni, nj, nk = cube.shape

    # Extract the appropriate slice based on orientation
    if slice_orientation == "inline":
        slice_data = cube[slice_idx, :, :]  # [J, K]
        xlabel = "Crossline (J)"
        ylabel_base = k_label
        extent = [0, nj - 1, (nk - 1) * k_scale, 0]  # J, K
        title_with_slice = f"{title}\n(Inline I={slice_idx})"
    elif slice_orientation == "crossline":
        slice_data = cube[:, slice_idx, :]  # [I, K]
        xlabel = "Inline (I)"
        ylabel_base = k_label
        extent = [0, ni - 1, (nk - 1) * k_scale, 0]  # I, K
        title_with_slice = f"{title}\n(Crossline J={slice_idx})"
    elif slice_orientation in ["timeslice", "depthslice"]:
        slice_data = cube[:, :, slice_idx]  # [I, J]
        xlabel = "Crossline (J)"
        ylabel_base = "Inline (I)"
        extent = [0, nj - 1, ni - 1, 0]  # J, I
        slice_label = (
            f"{k_label}={slice_idx * k_scale:.3f}{k_unit}"
            if k_unit
            else f"{k_label}={slice_idx}"
        )
        title_with_slice = f"{title}\n({slice_label})"
        k_unit = ""  # Don't add units to ylabel for horizontal slices
    else:
        raise ValueError(f"Unknown slice orientation: {slice_orientation}")

    # Determine vmin/vmax
    if is_categorical:
        vmin = 0
        vmax = 3  # Fixed to 4 facies (0, 1, 2, 3)
        # Create discrete colormap with exactly 4 colors
        from matplotlib.colors import ListedColormap

        colors = plt.cm.tab10(np.linspace(0, 0.4, 4))  # Get first 4 colors from tab10
        cmap_discrete = ListedColormap(colors)
    else:
        # Use seismic colormap convention
        # Use 99.5 percentile to better capture the full dynamic range
        # while still clipping extreme outliers
        p_i = np.percentile(np.abs(slice_data), 99.5)
        vmax = float(p_i)
        vmin = -vmax
        if vmax == vmin:
            vmax = vmin + 1.0

    ax.clear()
    # For categorical data, use nearest neighbor to preserve discrete values
    # For seismic, use smooth interpolation for better visual quality
    interp_method = "nearest" if is_categorical else "bilinear"

    # For seismic displays, aspect="auto" is standard practice
    # This stretches the image to fill the subplot, which is what we want
    # The "elongation" you see is normal - it makes features more visible
    im = ax.imshow(
        slice_data.T,
        aspect="auto",
        cmap=cmap_discrete if is_categorical else cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",  # Time/depth increases downward (or north->south for horizontal slices)
        extent=extent,
        interpolation=interp_method,
        interpolation_stage="rgba",  # Apply interpolation after colormapping for smoother results
    )

    ax.set_title(title_with_slice, fontsize=10)
    ax.set_xlabel(xlabel)

    # Update axis label with units
    if k_unit:
        ax.set_ylabel(f"{ylabel_base} ({k_unit})")
    else:
        ax.set_ylabel(ylabel_base)

    # Add colorbar for categorical data with exactly 4 discrete colors
    if is_categorical:
        cbar = plt.colorbar(
            im, ax=ax, ticks=[0, 1, 2, 3], boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5]
        )
        cbar.set_label("Facies")


def plot_alignment_metric(
    ax,
    alignment_data,
    slice_idx,
    slice_orientation,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
):
    """
    Plot boundary alignment metric with hot colormap.

    Args:
        ax: Matplotlib axis object
        alignment_data: 2D alignment metric array
        slice_idx: Integer index for slicing
        slice_orientation: 'inline' (I), 'crossline' (J), or 'timeslice'/'depthslice' (K)
        title: Plot title
        k_scale: Scaling factor for k-axis
        k_label: Label for k-axis
        k_unit: Unit string
    """
    nj, nk = alignment_data.shape

    # Determine extent and labels based on orientation
    if slice_orientation == "inline":
        xlabel = "Crossline (J)"
        ylabel_base = k_label
        extent = [0, nj - 1, (nk - 1) * k_scale, 0]
        title_with_slice = f"{title}\n(Inline I={slice_idx})"
    elif slice_orientation == "crossline":
        xlabel = "Inline (I)"
        ylabel_base = k_label
        extent = [0, nj - 1, (nk - 1) * k_scale, 0]
        title_with_slice = f"{title}\n(Crossline J={slice_idx})"
    else:  # timeslice or depthslice
        xlabel = "Inline (I)"
        ylabel_base = "Crossline (J)"
        extent = [0, nj - 1, nk - 1, 0]
        title_with_slice = f"{title}\n({k_label} K={slice_idx})"

    im = ax.imshow(
        alignment_data.T,
        aspect="auto",
        cmap="hot",  # Black (weak) to white (strong alignment)
        vmin=0,
        vmax=1,
        origin="upper",
        extent=extent,
        interpolation="bilinear",
    )

    ax.set_title(title_with_slice, fontsize=10)
    ax.set_xlabel(xlabel)

    if k_unit:
        ax.set_ylabel(f"{ylabel_base} ({k_unit})")
    else:
        ax.set_ylabel(ylabel_base)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Alignment\nStrength", fontsize=8)


def detect_facies_boundaries(facies, sigma=1.0):
    """
    Detect boundaries between facies using edge detection.

    Args:
        facies: 2D array of facies values (categorical)
        sigma: Gaussian smoothing parameter

    Returns:
        2D binary array where 1 = boundary, 0 = no boundary
    """
    from scipy.ndimage import gaussian_filter, sobel

    # Smooth slightly to reduce noise
    facies_smooth = gaussian_filter(facies.astype(float), sigma=sigma)

    # Compute gradients
    grad_x = sobel(facies_smooth, axis=0)
    grad_y = sobel(facies_smooth, axis=1)

    # Compute magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold to get binary boundaries
    # Any gradient > 0.1 indicates a facies change
    boundaries = (grad_magnitude > 0.1).astype(float)

    return boundaries


def compute_boundary_alignment(seismic, facies, sigma=1.5):
    """
    Compute how well seismic reflections align with facies boundaries.
    Returns seismic gradient strength at facies boundary locations.

    Higher values (brighter) = stronger seismic response at geological interfaces
    Lower values (darker) = weaker response or no boundary present

    Args:
        seismic: 2D seismic data array
        facies: 2D facies array (categorical)
        sigma: Gaussian smoothing parameter

    Returns:
        2D alignment metric array
    """
    from scipy.ndimage import gaussian_filter

    # Detect facies boundaries
    facies_boundaries = detect_facies_boundaries(facies, sigma=1.0)

    # Compute seismic gradients (reflectivity/edges)
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
        description="Generate 2D slice visualizations comparing AVO, AI, and EI methods"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["time", "depth"],
        default="depth",
        help="Domain to visualize: 'time' or 'depth' (default: depth)",
    )
    parser.add_argument(
        "--no-multiangle",
        action="store_true",
        help="Use single-angle EI seismogram instead of multi-angle impedance (default: use multi-angle)",
    )
    args = parser.parse_args()

    # Multi-angle is ON by default (matches modeling.py behavior)
    args.use_multiangle = not args.no_multiangle

    # Configuration parameters matching main workflow
    DATA_PATH = "."
    FILE_MAP = {
        "vp": "P-wave Velocity",
        "facies": "Facies",
    }
    GRID_SHAPE = (150, 200, 200)  # (nx, ny, nz)
    # Stanford VI-E paper specifies: 1m vertical cell size, 200m total thickness
    DZ = 1.0  # Vertical spacing in meters (200 samples × 1m = 200m total depth)
    DT = 0.001  # Time sampling interval in seconds (1 ms)

    cache_dir = ".cache"
    # Note: Cache file names will change with different DT values
    # For DT=0.1ms, hash will be different than DT=1ms
    # Load TIME DOMAIN seismograms for AVO and AI (with _time_ infix)
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

    avo_fn = os.path.join(cache_dir, sorted(avo_files)[-1])  # Use most recent
    ai_fn = os.path.join(cache_dir, sorted(ai_files)[-1])
    ei_fn = os.path.join(cache_dir, sorted(ei_files)[-1])

    print(f"Using cache files:")
    print(f"  AVO: {os.path.basename(avo_fn)}")
    print(f"  AI: {os.path.basename(ai_fn)}")
    print(f"  EI: {os.path.basename(ei_fn)} ({ei_type_str})")

    # Load data - handle both depth and time caches
    avo_cache = np.load(avo_fn, mmap_mode="r")
    if "full_stack" in avo_cache:
        avo = avo_cache["full_stack"]
    elif "angle_0" in avo_cache:
        avo = avo_cache["angle_0"]
    else:
        avo = avo_cache["impedance_depth"]

    ai_cache = np.load(ai_fn, mmap_mode="r")
    if "seismogram_ai" in ai_cache:
        ai = ai_cache["seismogram_ai"]
    else:
        ai = ai_cache["impedance_ai"]

    ei_cache = np.load(ei_fn, mmap_mode="r")
    ei = ei_cache[ei_data_key]

    print(f"Loaded data shapes:")
    print(f"  AVO: {avo.shape}")
    print(f"  AI: {ai.shape}")
    if ei_is_depth_domain:
        print(f"  EI: {ei.shape} (depth-domain impedance)")
    else:
        print(f"  EI: {ei.shape} (time-domain seismogram)")

    # Note: Rock physics attributes (Lambda-Rho, Fluid Factor, etc.) are plotted separately
    # Use: python -m src.plot_rock_physics_attributes

    # Calculate expected samples based on DT
    # Actual measured max TWT from velocity model: 143.388 ms (not 147.1 ms)
    # However, the actual array has nt samples from dynamic calculation
    nt_actual = avo.shape[2]
    max_twt_actual = (nt_actual - 1) * DT
    print(
        f"Actual time samples: {nt_actual}, max TWT: {max_twt_actual*1000:.3f} ms (DT={DT*1000:.1f}ms)"
    )

    # Use DT directly (velocity units now corrected in data generation)
    DT_CORRECTED = DT
    nyquist_freq = 1.0 / (2.0 * DT_CORRECTED)
    print(
        f"Time sampling: {DT_CORRECTED*1000:.1f} ms (Nyquist frequency: {nyquist_freq:.0f} Hz)"
    )

    # Load velocity model and facies for time-to-depth conversion
    print("Loading velocity model and facies for depth conversion...")
    props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
    vp_depth = props_depth["vp"]
    facies_depth = props_depth["facies"]

    # Convert velocity from km/s to m/s (Stanford VI-E data is in km/s)
    print(
        f"Vp range before conversion: {np.min(vp_depth):.2f} - {np.max(vp_depth):.2f} km/s"
    )
    vp_depth = vp_depth * 1000.0  # Convert km/s to m/s
    print(
        f"Vp range after conversion: {np.min(vp_depth):.0f} - {np.max(vp_depth):.0f} m/s"
    )

    # Get target time axis length from AVO/AI
    target_nt = avo.shape[2]

    # Handle EI based on its domain
    if ei_is_depth_domain:
        # Multi-angle: EI is in depth domain (impedance), need to:
        # 1. Convert impedance to seismogram (reflectivity + wavelet)
        # 2. Keep depth version as-is
        print("Multi-angle mode: Converting depth-domain impedance to seismogram...")

        # Convert to time domain first (for proper time axis)
        ei_impedance_time = convert_depth_to_time_with_target(
            ei, vp_depth, DZ, DT_CORRECTED, target_nt, is_categorical=False
        )
        print(f"  EI impedance (time) shape: {ei_impedance_time.shape}")
        print(
            f"  EI impedance range: [{ei_impedance_time.min():.2f}, {ei_impedance_time.max():.2f}]"
        )

        # Generate seismogram from impedance (reflectivity + convolution)
        ei_time = impedance_to_seismogram(ei_impedance_time, DT_CORRECTED, f_peak=30)

        # For depth domain, also convert to seismogram
        ei_impedance_depth = ei  # Original impedance in depth
        ei_depth = impedance_to_seismogram(
            ei_impedance_depth, DZ / 2000.0, f_peak=30
        )  # Approximate dt for depth

        print(f"  EI seismogram (time) shape: {ei_time.shape}")
        print(f"  EI seismogram (depth) shape: {ei_depth.shape}")
    else:
        # Single-angle: EI is in time domain (already a seismogram), need to convert to depth
        print("Single-angle mode: Converting EI from time to depth domain...")
        ei_time = ei  # Already in time domain (seismogram)
        ei_depth = convert_time_to_depth(ei, vp_depth, DZ, DT_CORRECTED)
        print(f"  EI seismogram (time) shape: {ei_time.shape}")
        print(f"  EI seismogram (depth) shape: {ei_depth.shape}")

    # Convert AVO and AI seismograms from time to depth
    avo_depth = convert_time_to_depth(avo, vp_depth, DZ, DT_CORRECTED)
    ai_depth = convert_time_to_depth(ai, vp_depth, DZ, DT_CORRECTED)
    print("Converted AVO and AI seismograms from time to depth domain")

    # pick center slices
    ni, nj, nk = avo.shape
    idx_i_time = ni // 2
    idx_j_time = nj // 2
    idx_k_time = nk // 2

    ni_d, nj_d, nk_d = avo_depth.shape
    idx_i_depth = ni_d // 2
    idx_j_depth = nj_d // 2
    idx_k_depth = nk_d // 2

    # Select domain based on user input
    if args.domain == "time":
        domain_label = "Time Domain"
        seismogram_avo = avo
        seismogram_ai = ai
        seismogram_ei = ei_time  # Use time-domain EI
        facies_data = convert_depth_to_time_with_target(
            facies_depth, vp_depth, DZ, DT_CORRECTED, target_nt, is_categorical=True
        )
        idx_i = idx_i_time
        idx_j = idx_j_time
        idx_k = idx_k_time
        k_scale = DT_CORRECTED
        k_label = "Time"
        k_unit = "s"
        slice3_orientation = "timeslice"
    else:  # depth
        domain_label = "Depth Domain"
        seismogram_avo = avo_depth
        seismogram_ai = ai_depth
        seismogram_ei = ei_depth  # Use depth-domain EI
        facies_data = facies_depth
        idx_i = idx_i_depth
        idx_j = idx_j_depth
        idx_k = idx_k_depth
        k_scale = DZ
        k_label = "Depth"
        k_unit = "m"
        slice3_orientation = "depthslice"

    # Compute boundary alignment metrics for each slice orientation
    print("\nComputing boundary alignment metrics...")

    # Inline slices (I-slices)
    avo_inline = seismogram_avo[idx_i, :, :]
    ai_inline = seismogram_ai[idx_i, :, :]
    ei_inline = seismogram_ei[idx_i, :, :]
    facies_inline = facies_data[idx_i, :, :]

    avo_align_inline = compute_boundary_alignment(avo_inline, facies_inline)
    ai_align_inline = compute_boundary_alignment(ai_inline, facies_inline)
    ei_align_inline = compute_boundary_alignment(ei_inline, facies_inline)

    # Crossline slices (J-slices)
    avo_crossline = seismogram_avo[:, idx_j, :]
    ai_crossline = seismogram_ai[:, idx_j, :]
    ei_crossline = seismogram_ei[:, idx_j, :]
    facies_crossline = facies_data[:, idx_j, :]

    avo_align_crossline = compute_boundary_alignment(avo_crossline, facies_crossline)
    ai_align_crossline = compute_boundary_alignment(ai_crossline, facies_crossline)
    ei_align_crossline = compute_boundary_alignment(ei_crossline, facies_crossline)

    # Time/Depth slices (K-slices)
    avo_kslice = seismogram_avo[:, :, idx_k]
    ai_kslice = seismogram_ai[:, :, idx_k]
    ei_kslice = seismogram_ei[:, :, idx_k]
    facies_kslice = facies_data[:, :, idx_k]

    avo_align_kslice = compute_boundary_alignment(avo_kslice, facies_kslice)
    ai_align_kslice = compute_boundary_alignment(ai_kslice, facies_kslice)
    ei_align_kslice = compute_boundary_alignment(ei_kslice, facies_kslice)

    # Create subplot layout
    # Columns: AVO, AI, EI, AVO Align, AI Align, EI Align, Facies
    ncols = 7
    fig = plt.figure(figsize=(5 * ncols, 12))
    ei_mode_str = " (Multi-angle optimal)" if args.use_multiangle else " (Single-angle)"
    domain_suffix = " (Depth Domain)" if args.domain == "depth" else " (Time Domain)"
    fig.suptitle(
        f"Seismic-Facies Boundary Alignment Analysis{domain_suffix}{ei_mode_str}\n(Brighter = Better Alignment with Geological Boundaries)",
        fontsize=16,
        y=0.995,
    )

    # ===== Row 1: Inline Slices =====
    col = 1
    ax1 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax1,
        seismogram_avo,
        idx_i,
        "inline",
        "Full-Stack Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax2 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax2,
        seismogram_ai,
        idx_i,
        "inline",
        "AI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax3 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax3,
        seismogram_ei,
        idx_i,
        "inline",
        "EI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax4 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax4,
        avo_align_inline,
        idx_i,
        "inline",
        "Full-Stack Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax5 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax5,
        ai_align_inline,
        idx_i,
        "inline",
        "AI Synthetic Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax6 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax6,
        ei_align_inline,
        idx_i,
        "inline",
        "EI Synthetic Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax7_facies = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax7_facies,
        facies_data,
        idx_i,
        "inline",
        "Facies",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="tab10",
        is_categorical=True,
    )

    # ===== Row 2: Crossline Slices =====
    col = 1 + ncols
    ax8 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax8,
        seismogram_avo,
        idx_j,
        "crossline",
        "Full-Stack Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax9 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax9,
        seismogram_ai,
        idx_j,
        "crossline",
        "AI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax10 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax10,
        seismogram_ei,
        idx_j,
        "crossline",
        "EI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax11 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax11,
        avo_align_crossline,
        idx_j,
        "crossline",
        "Full-Stack Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax12 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax12,
        ai_align_crossline,
        idx_j,
        "crossline",
        "AI Synthetic Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax13 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax13,
        ei_align_crossline,
        idx_j,
        "crossline",
        "EI Synthetic Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax14_facies = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax14_facies,
        facies_data,
        idx_j,
        "crossline",
        "Facies",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="tab10",
        is_categorical=True,
    )

    # ===== Row 3: Time/Depth Slices =====
    col = 1 + 2 * ncols
    ax15 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax15,
        seismogram_avo,
        idx_k,
        slice3_orientation,
        "Full-Stack Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax16 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax16,
        seismogram_ai,
        idx_k,
        slice3_orientation,
        "AI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax17 = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax17,
        seismogram_ei,
        idx_k,
        slice3_orientation,
        "EI Synthetic Seismogram",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax18 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax18,
        avo_align_kslice,
        idx_k,
        slice3_orientation,
        "Full-Stack Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax19 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax19,
        ai_align_kslice,
        idx_k,
        slice3_orientation,
        "AI Synthetic Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax20 = fig.add_subplot(3, ncols, col)
    plot_alignment_metric(
        ax20,
        ei_align_kslice,
        idx_k,
        slice3_orientation,
        "EI Synthetic Alignment",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
    )
    col += 1

    ax21_facies = fig.add_subplot(3, ncols, col)
    plot_with_units(
        ax21_facies,
        facies_data,
        idx_k,
        slice3_orientation,
        "Facies",
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap="tab10",
        is_categorical=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(cache_dir, exist_ok=True)

    # Save with descriptive filename including domain
    domain_suffix = "_depth" if args.domain == "depth" else "_time"
    outfn = os.path.join(cache_dir, f"seismic_comparison{domain_suffix}.png")
    # Save at higher DPI for better quality and enable anti-aliasing
    plt.savefig(outfn, dpi=300, facecolor="white", edgecolor="none")
    print(f"Saved cached visualization to {outfn} at 300 DPI")


if __name__ == "__main__":
    main()
