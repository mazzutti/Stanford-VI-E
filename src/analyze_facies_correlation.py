"""Quantitative analysis of seismic-facies correlation.

This script performs statistical analysis to measure:
1. How well seismic amplitudes correlate with facies boundaries
2. Reflection strength at facies interfaces
3. Comparative performance of AVO vs AI techniques
4. Facies discrimination capability

Usage:
    python -m src.analyze_facies_correlation                        # Default: depth domain, multi-angle EI
    python -m src.analyze_facies_correlation --no-multiangle        # Use single-angle EI seismogram
    python -m src.analyze_facies_correlation --domain time          # Time domain (implies --no-multiangle)
    python -m src.analyze_facies_correlation --domain depth         # Explicit depth domain with multi-angle
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
import pandas as pd
import logging

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

from . import data_loader


def convert_depth_to_time(data_depth, vp_depth, dz, dt, is_categorical=False):
    """Convert depth-domain data to time domain using velocity model."""
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
            vp_trace = vp_depth[i, j, :]
            slowness = 1.0 / vp_trace
            one_way_time = np.cumsum(slowness * dz)
            twt_trace = 2 * one_way_time
            twt_trace = np.concatenate([[0], twt_trace])
            depth_trace = np.concatenate([[0], depth_axis + dz])

            data_trace = data_depth[i, j, :]
            data_trace_padded = np.concatenate([[data_trace[0]], data_trace])

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


def convert_time_to_depth(seismogram_time, vp_depth, dz, dt):
    """
    Convert a time-domain seismogram to depth domain.

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
            interp_func = interp1d(
                time_axis,
                seism_trace,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            seismogram_depth[i, j, :] = interp_func(twt_trace)

    return seismogram_depth


def impedance_to_seismogram_depth(impedance, dz, f_peak=30):
    """
    Convert depth-domain impedance to seismogram by computing reflectivity
    and convolving with wavelet.

    Args:
        impedance: 3D impedance cube in depth (ni, nj, nk)
        dz: Depth sampling interval (meters)
        f_peak: Peak frequency for Ricker wavelet (Hz)

    Returns:
        seismogram: 3D seismogram cube in depth (ni, nj, nk)
    """
    from scipy.signal import fftconvolve
    from . import wavelets, reflectivity as refl_module

    print(f"  Converting impedance to seismogram (f_peak={f_peak} Hz, depth domain)...")

    # Compute reflectivity in depth domain
    refl = refl_module.reflectivity_from_ai(impedance)

    # Clip extreme reflectivity values
    refl = np.clip(refl, -0.99, 0.99)

    print(f"    Reflectivity range: [{refl.min():.6f}, {refl.max():.6f}]")

    # Generate wavelet - use equivalent time sampling for depth domain
    # dt_equiv ~ dz / average_velocity (rough approximation)
    dt_equiv = dz / 2500.0  # Assuming average velocity ~2500 m/s
    wavelet = wavelets.ricker_wavelet(f_peak=f_peak, dt=dt_equiv)
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


def detect_facies_boundaries(facies_cube):
    """
    Detect facies boundaries in 3D using edge detection.

    Returns:
        Binary 3D array where True indicates a boundary location
    """
    print("Detecting facies boundaries in 3D...")
    ni, nj, nk = facies_cube.shape
    boundaries = np.zeros_like(facies_cube, dtype=bool)

    # Detect boundaries in each 2D slice
    for i in range(ni):
        slice_2d = facies_cube[i, :, :]
        smoothed = gaussian_filter(slice_2d.astype(float), sigma=0.5)
        grad_j = sobel(smoothed, axis=0)
        grad_k = sobel(smoothed, axis=1)
        gradient_magnitude = np.sqrt(grad_j**2 + grad_k**2)
        boundaries[i, :, :] = gradient_magnitude > 0.1

    return boundaries


def extract_boundary_amplitudes(seismic_cube, boundaries, window=2):
    """
    Extract seismic amplitudes at and near facies boundaries.

    Args:
        seismic_cube: 3D seismic amplitude array
        boundaries: 3D binary array of boundary locations
        window: Number of samples to include on each side of boundary

    Returns:
        Dictionary with amplitudes at boundaries and away from boundaries
    """
    print("Extracting amplitudes at facies boundaries...")

    # Ensure shapes match
    ni_s, nj_s, nk_s = seismic_cube.shape
    ni_b, nj_b, nk_b = boundaries.shape

    ni = min(ni_s, ni_b)
    nj = min(nj_s, nj_b)
    nk = min(nk_s, nk_b)

    seismic_aligned = seismic_cube[:ni, :nj, :nk]
    boundaries_aligned = boundaries[:ni, :nj, :nk]

    # Dilate boundaries to create a window
    from scipy.ndimage import binary_dilation

    boundary_zone = binary_dilation(boundaries_aligned, iterations=window)

    # Extract amplitudes
    at_boundaries = seismic_aligned[boundary_zone]
    away_from_boundaries = seismic_aligned[~boundary_zone]

    return {
        "at_boundaries": at_boundaries,
        "away_from_boundaries": away_from_boundaries,
        "boundary_mask": boundary_zone,
    }


def calculate_gradient_correlation(seismic_cube, facies_cube):
    """
    Calculate correlation between seismic amplitude gradients and facies boundaries.

    High correlation indicates seismic reflections align with geological interfaces.
    """
    print("Calculating gradient correlation...")

    # Ensure both cubes have the same shape (crop to minimum)
    ni_s, nj_s, nk_s = seismic_cube.shape
    ni_f, nj_f, nk_f = facies_cube.shape

    ni = min(ni_s, ni_f)
    nj = min(nj_s, nj_f)
    nk = min(nk_s, nk_f)

    seismic_aligned = seismic_cube[:ni, :nj, :nk]
    facies_aligned = facies_cube[:ni, :nj, :nk]

    print(
        f"  Aligned shapes: seismic={seismic_aligned.shape}, facies={facies_aligned.shape}"
    )

    # Calculate seismic vertical gradient (time/depth derivative)
    seismic_grad = np.gradient(seismic_aligned, axis=2)
    seismic_grad_abs = np.abs(seismic_grad)

    # Detect facies boundaries
    boundaries = detect_facies_boundaries(facies_aligned)

    # Calculate correlation between absolute seismic gradient and boundaries
    # Flatten for correlation calculation
    seismic_grad_flat = seismic_grad_abs.flatten()
    boundaries_flat = boundaries.flatten().astype(float)

    # Remove NaN/Inf values
    valid_mask = np.isfinite(seismic_grad_flat) & np.isfinite(boundaries_flat)
    seismic_grad_valid = seismic_grad_flat[valid_mask]
    boundaries_valid = boundaries_flat[valid_mask]

    # Calculate correlations
    pearson_corr, pearson_pval = pearsonr(seismic_grad_valid, boundaries_valid)
    spearman_corr, spearman_pval = spearmanr(seismic_grad_valid, boundaries_valid)

    return {
        "pearson_correlation": pearson_corr,
        "pearson_pvalue": pearson_pval,
        "spearman_correlation": spearman_corr,
        "spearman_pvalue": spearman_pval,
        "seismic_gradient": seismic_grad_abs,
        "boundaries": boundaries,
    }


def analyze_interface_reflections(seismic_cube, facies_cube):
    """
    Analyze reflection strength at specific facies interfaces.

    Returns statistics for each type of facies transition.
    """
    print("Analyzing reflection strength at interfaces...")

    # Ensure shapes match
    ni_s, nj_s, nk_s = seismic_cube.shape
    ni_f, nj_f, nk_f = facies_cube.shape

    ni = min(ni_s, ni_f)
    nj = min(nj_s, nj_f)
    nk = min(nk_s, nk_f)

    seismic_aligned = seismic_cube[:ni, :nj, :nk]
    facies_aligned = facies_cube[
        :ni, :nj, :nk
    ]  # Initialize storage for interface statistics
    interface_stats = {
        "0->1": [],
        "1->0": [],
        "1->2": [],
        "2->1": [],
        "2->3": [],
        "3->2": [],
        "0->2": [],
        "2->0": [],
        "0->3": [],
        "3->0": [],
        "1->3": [],
        "3->1": [],
    }

    # Analyze along vertical direction (k-axis, time/depth)
    for i in range(ni):
        for j in range(nj):
            facies_trace = facies_aligned[i, j, :]
            seismic_trace = seismic_aligned[i, j, :]

            # Find facies transitions
            for k in range(1, nk):
                if facies_trace[k] != facies_trace[k - 1]:
                    # Facies transition detected
                    facies_from = int(facies_trace[k - 1])
                    facies_to = int(facies_trace[k])

                    # Get seismic amplitude at interface (average around interface)
                    window = slice(max(0, k - 2), min(nk, k + 3))
                    interface_amp = np.abs(seismic_trace[window]).mean()

                    # Store by transition type
                    key = f"{facies_from}->{facies_to}"
                    if key in interface_stats:
                        interface_stats[key].append(interface_amp)

    # Calculate statistics for each transition type
    summary = {}
    for key, values in interface_stats.items():
        if len(values) > 0:
            summary[key] = {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "min": np.min(values),
                "max": np.max(values),
            }
        else:
            summary[key] = None

    return summary, interface_stats


def calculate_facies_discrimination(seismic_cube, facies_cube):
    """
    Measure how well seismic amplitudes discriminate between facies types.

    Uses statistical separation metrics.
    """
    print("Calculating facies discrimination capability...")

    # Ensure shapes match
    ni_s, nj_s, nk_s = seismic_cube.shape
    ni_f, nj_f, nk_f = facies_cube.shape

    ni = min(ni_s, ni_f)
    nj = min(nj_s, nj_f)
    nk = min(nk_s, nk_f)

    seismic_aligned = seismic_cube[:ni, :nj, :nk]
    facies_aligned = facies_cube[:ni, :nj, :nk]

    # Extract amplitudes for each facies
    facies_amplitudes = {}
    for facies_val in range(4):
        mask = facies_aligned == facies_val
        if np.any(mask):
            facies_amplitudes[facies_val] = seismic_aligned[
                mask
            ]  # Calculate statistics per facies
    facies_stats = {}
    for facies_val, amps in facies_amplitudes.items():
        facies_stats[facies_val] = {
            "count": len(amps),
            "mean": np.mean(amps),
            "std": np.std(amps),
            "median": np.median(amps),
            "q25": np.percentile(amps, 25),
            "q75": np.percentile(amps, 75),
        }

    # Calculate separation between facies (using means)
    separation_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j and i in facies_stats and j in facies_stats:
                # Cohen's d effect size
                mean_diff = abs(facies_stats[i]["mean"] - facies_stats[j]["mean"])
                pooled_std = np.sqrt(
                    (facies_stats[i]["std"] ** 2 + facies_stats[j]["std"] ** 2) / 2
                )
                separation_matrix[i, j] = mean_diff / (pooled_std + 1e-10)

    return facies_stats, separation_matrix, facies_amplitudes


def compare_techniques(avo_stats, ai_stats, metric_name):
    """Compare AVO vs AI performance on a given metric."""
    comparison = {}

    if metric_name == "gradient_correlation":
        comparison["AVO"] = {
            "Pearson": avo_stats["pearson_correlation"],
            "Spearman": avo_stats["spearman_correlation"],
        }
        comparison["AI"] = {
            "Pearson": ai_stats["pearson_correlation"],
            "Spearman": ai_stats["spearman_correlation"],
        }
        comparison["Winner"] = (
            "AVO"
            if avo_stats["pearson_correlation"] > ai_stats["pearson_correlation"]
            else "AI"
        )
        comparison["Difference"] = abs(
            avo_stats["pearson_correlation"] - ai_stats["pearson_correlation"]
        )

    return comparison


def create_summary_plots(
    avo_results, ai_results, ei_results, cache_dir, domain="depth"
):
    """Create comprehensive visualization of analysis results."""

    fig = plt.figure(figsize=(24, 18))
    domain_label = "Depth Domain" if domain == "depth" else "Time Domain"
    fig.suptitle(
        f"Quantitative Seismic-Facies Correlation Analysis: AVO vs AI vs EI ({domain_label})",
        fontsize=16,
        y=0.995,
    )

    # 1. Amplitude distributions at vs away from boundaries
    ax1 = plt.subplot(4, 4, 1)
    ax1.hist(
        avo_results["boundary_amps"]["at_boundaries"],
        bins=50,
        alpha=0.7,
        label="At Boundaries",
        density=True,
        color="red",
    )
    ax1.hist(
        avo_results["boundary_amps"]["away_from_boundaries"],
        bins=50,
        alpha=0.7,
        label="Away from Boundaries",
        density=True,
        color="blue",
    )
    ax1.set_xlabel("AVO Amplitude")
    ax1.set_ylabel("Density")
    ax1.set_title("AVO: Amplitude Distribution")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(4, 4, 2)
    ax2.hist(
        ai_results["boundary_amps"]["at_boundaries"],
        bins=50,
        alpha=0.7,
        label="At Boundaries",
        density=True,
        color="red",
    )
    ax2.hist(
        ai_results["boundary_amps"]["away_from_boundaries"],
        bins=50,
        alpha=0.7,
        label="Away from Boundaries",
        density=True,
        color="blue",
    )
    ax2.set_xlabel("AI Amplitude")
    ax2.set_ylabel("Density")
    ax2.set_title("AI: Amplitude Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3_ei = plt.subplot(4, 4, 3)
    ax3_ei.hist(
        ei_results["boundary_amps"]["at_boundaries"],
        bins=50,
        alpha=0.7,
        label="At Boundaries",
        density=True,
        color="red",
    )
    ax3_ei.hist(
        ei_results["boundary_amps"]["away_from_boundaries"],
        bins=50,
        alpha=0.7,
        label="Away from Boundaries",
        density=True,
        color="blue",
    )
    ax3_ei.set_xlabel("EI Amplitude")
    ax3_ei.set_ylabel("Density")
    ax3_ei.set_title("EI: Amplitude Distribution")
    ax3_ei.legend(fontsize=8)
    ax3_ei.grid(True, alpha=0.3)

    # 2. Comparison bar chart (gradient correlation)
    ax4 = plt.subplot(4, 4, 4)
    methods = ["AVO", "AI", "EI"]
    pearson_values = [
        avo_results["gradient_correlation"]["pearson_correlation"],
        ai_results["gradient_correlation"]["pearson_correlation"],
        ei_results["gradient_correlation"]["pearson_correlation"],
    ]
    colors_comp = ["steelblue", "coral", "mediumseagreen"]
    ax4.bar(methods, pearson_values, color=colors_comp, alpha=0.7)
    ax4.set_ylabel("Pearson Correlation")
    ax4.set_title("Gradient-Boundary Correlation Comparison")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_ylim([0, max(pearson_values) * 1.2])

    # 3. Reflection strength at different interface types
    ax5 = plt.subplot(4, 4, 5)
    interface_types = []
    avo_means = []
    avo_stds = []
    for key, stats in avo_results["interface_stats_summary"].items():
        if stats is not None and stats["count"] > 10:
            interface_types.append(key)
            avo_means.append(stats["mean"])
            avo_stds.append(stats["std"])

    x_pos = np.arange(len(interface_types))
    ax5.bar(x_pos, avo_means, yerr=avo_stds, alpha=0.7, color="steelblue", capsize=5)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(interface_types, rotation=45, ha="right", fontsize=8)
    ax5.set_ylabel("Mean Amplitude")
    ax5.set_title("AVO: Reflection Strength at Interfaces")
    ax5.grid(True, alpha=0.3, axis="y")

    ax6 = plt.subplot(4, 4, 6)
    interface_types_ai = []
    ai_means = []
    ai_stds = []
    for key, stats in ai_results["interface_stats_summary"].items():
        if stats is not None and stats["count"] > 10:
            interface_types_ai.append(key)
            ai_means.append(stats["mean"])
            ai_stds.append(stats["std"])

    x_pos_ai = np.arange(len(interface_types_ai))
    ax6.bar(x_pos_ai, ai_means, yerr=ai_stds, alpha=0.7, color="coral", capsize=5)
    ax6.set_xticks(x_pos_ai)
    ax6.set_xticklabels(interface_types_ai, rotation=45, ha="right", fontsize=8)
    ax6.set_ylabel("Mean Amplitude")
    ax6.set_title("AI: Reflection Strength at Interfaces")
    ax6.grid(True, alpha=0.3, axis="y")

    ax7_ei = plt.subplot(4, 4, 7)
    interface_types_ei = []
    ei_means = []
    ei_stds = []
    for key, stats in ei_results["interface_stats_summary"].items():
        if stats is not None and stats["count"] > 10:
            interface_types_ei.append(key)
            ei_means.append(stats["mean"])
            ei_stds.append(stats["std"])

    x_pos_ei = np.arange(len(interface_types_ei))
    ax7_ei.bar(
        x_pos_ei, ei_means, yerr=ei_stds, alpha=0.7, color="mediumseagreen", capsize=5
    )
    ax7_ei.set_xticks(x_pos_ei)
    ax7_ei.set_xticklabels(interface_types_ei, rotation=45, ha="right", fontsize=8)
    ax7_ei.set_ylabel("Mean Amplitude")
    ax7_ei.set_title("EI: Reflection Strength at Interfaces")
    ax7_ei.grid(True, alpha=0.3, axis="y")

    # Position 8: Facies Separation Comparison
    ax8 = plt.subplot(4, 4, 8)
    methods = ["AVO", "AI", "EI"]
    sep_values = [
        np.mean(avo_results["separation_matrix"][avo_results["separation_matrix"] > 0]),
        np.mean(ai_results["separation_matrix"][ai_results["separation_matrix"] > 0]),
        np.mean(ei_results["separation_matrix"][ei_results["separation_matrix"] > 0]),
    ]
    colors_sep = ["steelblue", "coral", "mediumseagreen"]
    bars = ax8.bar(methods, sep_values, color=colors_sep, alpha=0.7)
    ax8.set_ylabel("Cohen's d (Effect Size)")
    ax8.set_title("Facies Separation Comparison")
    ax8.grid(True, alpha=0.3, axis="y")
    ax8.set_ylim([0, max(sep_values) * 1.2])
    # Add value labels on bars
    for bar, val in zip(bars, sep_values):
        height = bar.get_height()
        ax8.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Facies discrimination - amplitude by facies type
    ax9 = plt.subplot(4, 4, 9)
    facies_labels = []
    avo_facies_data = []
    for facies_val in range(4):
        if facies_val in avo_results["facies_amplitudes"]:
            facies_labels.append(f"Facies {facies_val}")
            # Sample for plotting (too many points otherwise)
            data = avo_results["facies_amplitudes"][facies_val]
            sampled = data[:: max(1, len(data) // 1000)]
            avo_facies_data.append(sampled)

    bp = ax9.boxplot(avo_facies_data, labels=facies_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 0.4, 4))):
        patch.set_facecolor(color)
    ax9.set_ylabel("AVO Amplitude")
    ax9.set_title("AVO: Amplitude by Facies Type")
    ax9.grid(True, alpha=0.3, axis="y")

    ax10 = plt.subplot(4, 4, 10)
    facies_labels_ai = []
    ai_facies_data = []
    for facies_val in range(4):
        if facies_val in ai_results["facies_amplitudes"]:
            facies_labels_ai.append(f"Facies {facies_val}")
            data = ai_results["facies_amplitudes"][facies_val]
            sampled = data[:: max(1, len(data) // 1000)]
            ai_facies_data.append(sampled)

    bp = ax10.boxplot(ai_facies_data, labels=facies_labels_ai, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 0.4, 4))):
        patch.set_facecolor(color)
    ax10.set_ylabel("AI Amplitude")
    ax10.set_title("AI: Amplitude by Facies Type")
    ax10.grid(True, alpha=0.3, axis="y")

    ax11_ei = plt.subplot(4, 4, 11)
    facies_labels_ei = []
    ei_facies_data = []
    for facies_val in range(4):
        if facies_val in ei_results["facies_amplitudes"]:
            facies_labels_ei.append(f"Facies {facies_val}")
            data = ei_results["facies_amplitudes"][facies_val]
            sampled = data[:: max(1, len(data) // 1000)]
            ei_facies_data.append(sampled)

    bp = ax11_ei.boxplot(ei_facies_data, labels=facies_labels_ei, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 0.4, 4))):
        patch.set_facecolor(color)
    ax11_ei.set_ylabel("EI Amplitude")
    ax11_ei.set_title("EI: Amplitude by Facies Type")
    ax11_ei.grid(True, alpha=0.3, axis="y")

    # Position 12: Boundary Amplitude Comparison
    ax12 = plt.subplot(4, 4, 12)

    boundary_means = [
        np.mean(np.abs(avo_results["boundary_amps"]["at_boundaries"])),
        np.mean(np.abs(ai_results["boundary_amps"]["at_boundaries"])),
        np.mean(np.abs(ei_results["boundary_amps"]["at_boundaries"])),
    ]
    away_means = [
        np.mean(np.abs(avo_results["boundary_amps"]["away_from_boundaries"])),
        np.mean(np.abs(ai_results["boundary_amps"]["away_from_boundaries"])),
        np.mean(np.abs(ei_results["boundary_amps"]["away_from_boundaries"])),
    ]

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax12.bar(
        x - width / 2,
        boundary_means,
        width,
        label="At Boundaries",
        color=["steelblue", "coral", "mediumseagreen"],
        alpha=0.8,
    )
    bars2 = ax12.bar(
        x + width / 2,
        away_means,
        width,
        label="Away from Boundaries",
        color=["lightsteelblue", "lightcoral", "lightgreen"],
        alpha=0.8,
    )

    ax12.set_ylabel("Mean |Amplitude|")
    ax12.set_title("Boundary vs Background Amplitude")
    ax12.set_xticks(x)
    ax12.set_xticklabels(methods)
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3, axis="y")

    # 5. Facies separation matrix (Cohen's d)
    ax13 = plt.subplot(4, 4, 13)
    im = ax13.imshow(
        avo_results["separation_matrix"], cmap="YlOrRd", aspect="auto", vmin=0, vmax=3
    )
    ax13.set_xticks([0, 1, 2, 3])
    ax13.set_yticks([0, 1, 2, 3])
    ax13.set_xticklabels(["F0", "F1", "F2", "F3"])
    ax13.set_yticklabels(["F0", "F1", "F2", "F3"])
    ax13.set_xlabel("Facies")
    ax13.set_ylabel("Facies")
    ax13.set_title("AVO: Facies Separation (Cohen's d)")
    # Add text annotations
    for i in range(4):
        for j in range(4):
            if i != j:
                text = ax13.text(
                    j,
                    i,
                    f'{avo_results["separation_matrix"][i, j]:.2f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    plt.colorbar(im, ax=ax13, label="Effect Size")

    ax14 = plt.subplot(4, 4, 14)
    im = ax14.imshow(
        ai_results["separation_matrix"], cmap="YlOrRd", aspect="auto", vmin=0, vmax=3
    )
    ax14.set_xticks([0, 1, 2, 3])
    ax14.set_yticks([0, 1, 2, 3])
    ax14.set_xticklabels(["F0", "F1", "F2", "F3"])
    ax14.set_yticklabels(["F0", "F1", "F2", "F3"])
    ax14.set_xlabel("Facies")
    ax14.set_ylabel("Facies")
    ax14.set_title("AI: Facies Separation (Cohen's d)")
    for i in range(4):
        for j in range(4):
            if i != j:
                text = ax14.text(
                    j,
                    i,
                    f'{ai_results["separation_matrix"][i, j]:.2f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    plt.colorbar(im, ax=ax14, label="Effect Size")

    ax15_ei = plt.subplot(4, 4, 15)
    im = ax15_ei.imshow(
        ei_results["separation_matrix"], cmap="YlOrRd", aspect="auto", vmin=0, vmax=3
    )
    ax15_ei.set_xticks([0, 1, 2, 3])
    ax15_ei.set_yticks([0, 1, 2, 3])
    ax15_ei.set_xticklabels(["F0", "F1", "F2", "F3"])
    ax15_ei.set_yticklabels(["F0", "F1", "F2", "F3"])
    ax15_ei.set_xlabel("Facies")
    ax15_ei.set_ylabel("Facies")
    ax15_ei.set_title("EI: Facies Separation (Cohen's d)")
    for i in range(4):
        for j in range(4):
            if i != j:
                text = ax15_ei.text(
                    j,
                    i,
                    f'{ei_results["separation_matrix"][i, j]:.2f}',
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    plt.colorbar(im, ax=ax15_ei, label="Effect Size")

    # 6. Summary statistics table
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis("off")

    # Create summary table with EI
    avo_pearson = avo_results["gradient_correlation"]["pearson_correlation"]
    ai_pearson = ai_results["gradient_correlation"]["pearson_correlation"]
    ei_pearson = ei_results["gradient_correlation"]["pearson_correlation"]

    avo_boundary = np.mean(avo_results["boundary_amps"]["at_boundaries"])
    ai_boundary = np.mean(ai_results["boundary_amps"]["at_boundaries"])
    ei_boundary = np.mean(ei_results["boundary_amps"]["at_boundaries"])

    avo_sep = np.mean(
        avo_results["separation_matrix"][avo_results["separation_matrix"] > 0]
    )
    ai_sep = np.mean(
        ai_results["separation_matrix"][ai_results["separation_matrix"] > 0]
    )
    ei_sep = np.mean(
        ei_results["separation_matrix"][ei_results["separation_matrix"] > 0]
    )

    summary_data = [
        ["Metric", "AVO", "AI", "EI", "Best"],
        [
            "Pearson r",
            f"{avo_pearson:.4f}",
            f"{ai_pearson:.4f}",
            f"{ei_pearson:.4f}",
            (
                "EI"
                if ei_pearson == max(avo_pearson, ai_pearson, ei_pearson)
                else (
                    "AI"
                    if ai_pearson == max(avo_pearson, ai_pearson, ei_pearson)
                    else "AVO"
                )
            ),
        ],
        [
            "Boundary Amp",
            f"{avo_boundary:.4f}",
            f"{ai_boundary:.4f}",
            f"{ei_boundary:.4f}",
            (
                "EI"
                if ei_boundary == max(avo_boundary, ai_boundary, ei_boundary)
                else (
                    "AI"
                    if ai_boundary == max(avo_boundary, ai_boundary, ei_boundary)
                    else "AVO"
                )
            ),
        ],
        [
            "Avg Sep (d)",
            f"{avo_sep:.3f}",
            f"{ai_sep:.3f}",
            f"{ei_sep:.3f}",
            (
                "EI"
                if ei_sep == max(avo_sep, ai_sep, ei_sep)
                else "AI" if ai_sep == max(avo_sep, ai_sep, ei_sep) else "AVO"
            ),
        ],
    ]

    table = ax16.table(
        cellText=summary_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax16.set_title("Performance Summary", fontsize=10, pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    outfn = os.path.join(cache_dir, f"facies_analysis_{domain}.png")
    plt.savefig(
        outfn, dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight"
    )
    print(f"\n✓ Saved quantitative analysis to {outfn}")

    return outfn


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Quantitative seismic-facies correlation analysis"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["depth", "time"],
        default="depth",
        help="Domain for analysis: 'depth' (default) or 'time'",
    )
    parser.add_argument(
        "--no-multiangle",
        action="store_true",
        help="Use single-angle EI seismogram instead of multi-angle impedance (default: use multi-angle)",
    )
    args = parser.parse_args()

    # Multi-angle is ON by default (matches modeling.py behavior)
    args.use_multiangle = not args.no_multiangle

    # Time domain analysis cannot use multi-angle (only has seismograms)
    if args.domain == "time" and args.use_multiangle:
        print("Note: Time domain analysis requires seismograms, disabling multi-angle")
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

    # Find cache files based on domain (exclude inversion results)
    domain_suffix = f"{args.domain}_"  # "depth_" or "time_"

    avo_files = [
        f
        for f in os.listdir(cache_dir)
        if f.startswith(f"avo_{domain_suffix}")
        and f.endswith(".npz")
        and "results" not in f
    ]
    ai_files = [
        f
        for f in os.listdir(cache_dir)
        if f.startswith(f"ai_{domain_suffix}") and f.endswith(".npz")
    ]

    # Find EI files based on multi-angle vs single-angle mode
    if args.use_multiangle:
        # Multi-angle mode: search based on domain
        if args.domain == "depth":
            # Look for EI depth cache
            ei_depth_files = [
                f
                for f in os.listdir(cache_dir)
                if f.startswith("ei_depth_") and f.endswith(".npz")
            ]
            if ei_depth_files:
                ei_files = ei_depth_files
                ei_data_key = "ei_product"  # variance-weighted optimal
                ei_type_str = (
                    "multi-angle depth-domain impedance (variance-weighted optimal)"
                )
                ei_is_depth_domain = True
            else:
                # No multi-angle file, need to generate it first
                ei_files = []
                ei_data_key = None
                ei_type_str = "MULTI-ANGLE DEPTH FILE NOT FOUND (run modeling.py first)"
                ei_is_depth_domain = False
        else:  # time domain
            # Look for time-domain EI files (generated from multi-angle in modeling.py)
            ei_time_files = [
                f
                for f in os.listdir(cache_dir)
                if f.startswith("ei_time_") and f.endswith(".npz")
            ]
            if ei_time_files:
                # Check if the file is from multi-angle by checking its config
                latest_ei = os.path.join(cache_dir, sorted(ei_time_files)[-1])
                try:
                    ei_test = np.load(latest_ei, allow_pickle=True)
                    config = ei_test.get("config", None)
                    if config is not None and hasattr(config, "item"):
                        config_dict = config.item()
                        if (
                            "source" in config_dict
                            and "multi-angle" in config_dict["source"]
                        ):
                            ei_type_str = "multi-angle time-domain seismogram (from optimal stack)"
                        else:
                            ei_type_str = "time-domain seismogram"
                    else:
                        ei_type_str = "time-domain seismogram"
                    ei_test.close()
                except:
                    ei_type_str = "time-domain seismogram"

                ei_files = ei_time_files
                ei_data_key = "ei_seismic"
                ei_is_depth_domain = False
            else:
                ei_files = []
                ei_data_key = None
                ei_type_str = "EI TIME FILE NOT FOUND (run modeling.py first)"
                ei_is_depth_domain = False
    else:
        # Single-angle mode: prioritize depth domain files
        ei_depth_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("ei_depth_")
            and f.endswith(".npz")
            and "multiangle" not in f
        ]
        ei_time_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("ei_time_") and f.endswith(".npz")
        ]

        # Use depth domain EI if available, otherwise time domain
        if args.domain == "depth" and ei_depth_files:
            ei_files = ei_depth_files
            ei_data_key = "ei_product"  # Try multi-angle first
            ei_type_str = "depth-domain impedance"
            ei_is_depth_domain = True
        elif ei_time_files:
            ei_files = ei_time_files
            ei_data_key = "ei_seismic"
            ei_type_str = "time-domain seismogram"
            ei_is_depth_domain = False
        else:
            ei_files = []
            ei_data_key = None
            ei_type_str = "NOT FOUND"
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

    # Load data from cache files
    avo_cache = np.load(avo_fn)
    ai_cache = np.load(ai_fn)
    ei_cache = np.load(ei_fn)

    # Extract data with correct keys based on domain
    # AVO: use angle stack for depth, full stack for time
    if "impedance_depth" in avo_cache:
        avo = avo_cache["impedance_depth"]  # Depth: angle stack average
    else:
        avo = avo_cache["full_stack"]  # Time: seismogram

    # AI: different keys for depth vs time
    if "impedance_ai" in ai_cache:
        ai = ai_cache["impedance_ai"]  # Depth: impedance
    else:
        ai = ai_cache["seismogram_ai"]  # Time: seismogram

    # EI: Prioritize weighted product synergy approach
    # NOTE: ei_product is in km/s units (gives better Cohen's d discrimination)
    # We convert to m/s-equivalent impedance for amplitude comparison with AVO/AI
    if "ei_product" in ei_cache:
        ei_km = ei_cache["ei_product"]  # km/s units

        # Convert to m/s-equivalent impedance using Connolly physics
        # The conversion factor depends on the actual rock properties and angles used
        # From our earlier calculation: ~491× theoretical, but let's use ei_optimal as reference
        if "ei_optimal" in ei_cache:
            ei_optimal = ei_cache["ei_optimal"]
            # Calculate empirical conversion factor from the data itself
            conversion_factor = np.mean(ei_optimal) / np.mean(ei_km)
            ei = ei_km * conversion_factor
            print(
                f"  Using WEIGHTED PRODUCT EI (converted to m/s, factor: {conversion_factor:.1f}×)"
            )
            ei_source = "Weighted Product (km/s converted to m/s)"
        else:
            # Fallback: use theoretical conversion
            ei = ei_km * 491.0
            print(f"  Using WEIGHTED PRODUCT EI (theoretical conversion ~491×)")
            ei_source = "Weighted Product (theoretical conversion)"
    elif "ei_optimal" in ei_cache:
        ei = ei_cache["ei_optimal"]
        print(f"  Using OPTIMAL EI (variance-weighted, m/s units)")
        ei_source = "Variance-weighted optimal (m/s units)"
    else:
        ei = ei_cache[ei_data_key]
        ei_source = "standard multi-angle optimal"

    data_type = "impedances" if args.domain == "depth" else "seismograms"
    print(f"Loaded {data_type} ({args.domain} domain):")
    print(f"  AVO: {avo.shape}")
    print(f"  AI: {ai.shape}")
    print(f"  EI: {ei.shape} ({ei_source})")

    # Load velocity model and facies (depth domain)
    print("\nLoading velocity model and facies...")
    props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)
    vp_depth = props_depth["vp"]
    facies_depth = props_depth["facies"]

    # Convert velocity from km/s to m/s
    print("Converting velocity units from km/s to m/s...")
    vp_depth = vp_depth * 1000.0

    # NOTE: For analysis, we work with impedances directly (not seismograms)
    # This preserves the superior facies discrimination of impedance-based analysis
    # (Cohen's d ~14 for impedances vs ~0.5 for seismograms)

    # Process data based on selected domain
    if args.domain == "depth":
        print(f"\n{'='*70}")
        print("QUANTITATIVE ANALYSIS - DEPTH DOMAIN")
        print(f"{'='*70}")

        # Use depth-domain impedances directly (NO conversion to seismogram)
        print("\nUsing depth-domain impedances for analysis")
        print("(Impedances provide superior facies discrimination)")
        avo_display = avo
        ai_display = ai

        # EI may already be in depth domain (if using multi-angle)
        if ei_is_depth_domain:
            print("  EI: Already in depth domain (impedance)")
            ei_display = ei
        else:
            print("  EI: Converting from time-domain seismogram to depth")
            ei_display = convert_time_to_depth(ei, vp_depth, DZ, DT)

        # Use depth-domain facies directly
        facies_display = facies_depth

        print(f"Impedance shapes (depth domain):")
        print(f"  AVO: {avo_display.shape}")
        print(f"  AI: {ai_display.shape}")
        print(f"  EI: {ei_display.shape}")

    else:  # args.domain == "time"
        print(f"\n{'='*70}")
        print("QUANTITATIVE ANALYSIS - TIME DOMAIN")
        print(f"{'='*70}")

        # Use time-domain data (impedances, will show as large values)
        avo_display = avo
        ai_display = ai
        ei_display = ei

        # Convert facies from depth to time domain
        print("\nConverting facies from depth to time domain...")
        facies_display = convert_depth_to_time(
            facies_depth, vp_depth, DZ, DT, is_categorical=True
        )

        print(f"Facies converted to time domain: {facies_display.shape}")

    print(f"\n{'='*70}")
    print("AVO vs AI vs EI Seismic-Facies Correlation")
    print(f"{'='*70}\n")

    # ===== ANALYZE AVO =====
    print("=" * 70)
    print("ANALYZING AVO TECHNIQUE")
    print("=" * 70)

    avo_gradient_corr = calculate_gradient_correlation(avo_display, facies_display)
    print(f"\n1. Gradient-Boundary Correlation (AVO):")
    print(
        f"   Pearson r = {avo_gradient_corr['pearson_correlation']:.4f} (p={avo_gradient_corr['pearson_pvalue']:.2e})"
    )
    print(
        f"   Spearman ρ = {avo_gradient_corr['spearman_correlation']:.4f} (p={avo_gradient_corr['spearman_pvalue']:.2e})"
    )

    avo_boundary_amps = extract_boundary_amplitudes(
        avo_display, avo_gradient_corr["boundaries"]
    )
    print(f"\n2. Amplitude Statistics (AVO):")
    print(
        f"   At boundaries: mean={np.mean(avo_boundary_amps['at_boundaries']):.4f}, "
        f"std={np.std(avo_boundary_amps['at_boundaries']):.4f}"
    )
    print(
        f"   Away from boundaries: mean={np.mean(avo_boundary_amps['away_from_boundaries']):.4f}, "
        f"std={np.std(avo_boundary_amps['away_from_boundaries']):.4f}"
    )

    avo_interface_summary, avo_interface_raw = analyze_interface_reflections(
        avo_display, facies_display
    )
    print(f"\n3. Interface Reflection Strength (AVO):")
    for key, stats in avo_interface_summary.items():
        if stats is not None and stats["count"] > 10:
            print(
                f"   {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['count']}"
            )

    avo_facies_stats, avo_separation, avo_facies_amps = calculate_facies_discrimination(
        avo_display, facies_display
    )
    print(f"\n4. Facies Discrimination (AVO):")
    for facies_val, stats in avo_facies_stats.items():
        print(
            f"   Facies {facies_val}: mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )
    print(
        f"   Average separation (Cohen's d): {np.mean(avo_separation[avo_separation > 0]):.3f}"
    )

    # ===== ANALYZE AI =====
    print(f"\n{'='*70}")
    print("ANALYZING AI TECHNIQUE")
    print("=" * 70)

    ai_gradient_corr = calculate_gradient_correlation(ai_display, facies_display)
    print(f"\n1. Gradient-Boundary Correlation (AI):")
    print(
        f"   Pearson r = {ai_gradient_corr['pearson_correlation']:.4f} (p={ai_gradient_corr['pearson_pvalue']:.2e})"
    )
    print(
        f"   Spearman ρ = {ai_gradient_corr['spearman_correlation']:.4f} (p={ai_gradient_corr['spearman_pvalue']:.2e})"
    )

    ai_boundary_amps = extract_boundary_amplitudes(
        ai_display, ai_gradient_corr["boundaries"]
    )
    print(f"\n2. Amplitude Statistics (AI):")
    print(
        f"   At boundaries: mean={np.mean(ai_boundary_amps['at_boundaries']):.4f}, "
        f"std={np.std(ai_boundary_amps['at_boundaries']):.4f}"
    )
    print(
        f"   Away from boundaries: mean={np.mean(ai_boundary_amps['away_from_boundaries']):.4f}, "
        f"std={np.std(ai_boundary_amps['away_from_boundaries']):.4f}"
    )

    ai_interface_summary, ai_interface_raw = analyze_interface_reflections(
        ai_display, facies_display
    )
    print(f"\n3. Interface Reflection Strength (AI):")
    for key, stats in ai_interface_summary.items():
        if stats is not None and stats["count"] > 10:
            print(
                f"   {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['count']}"
            )

    ai_facies_stats, ai_separation, ai_facies_amps = calculate_facies_discrimination(
        ai_display, facies_display
    )
    print(f"\n4. Facies Discrimination (AI):")
    for facies_val, stats in ai_facies_stats.items():
        print(
            f"   Facies {facies_val}: mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )
    print(
        f"   Average separation (Cohen's d): {np.mean(ai_separation[ai_separation > 0]):.3f}"
    )

    # ===== ANALYZE EI =====
    print(f"\n{'='*70}")
    print("ANALYZING EI TECHNIQUE")
    print("=" * 70)

    ei_gradient_corr = calculate_gradient_correlation(ei_display, facies_display)
    print(f"\n1. Gradient-Boundary Correlation (EI):")
    print(
        f"   Pearson r = {ei_gradient_corr['pearson_correlation']:.4f} (p={ei_gradient_corr['pearson_pvalue']:.2e})"
    )
    print(
        f"   Spearman ρ = {ei_gradient_corr['spearman_correlation']:.4f} (p={ei_gradient_corr['spearman_pvalue']:.2e})"
    )

    ei_boundary_amps = extract_boundary_amplitudes(
        ei_display, ei_gradient_corr["boundaries"]
    )
    print(f"\n2. Amplitude Statistics (EI):")
    print(
        f"   At boundaries: mean={np.mean(ei_boundary_amps['at_boundaries']):.4f}, "
        f"std={np.std(ei_boundary_amps['at_boundaries']):.4f}"
    )
    print(
        f"   Away from boundaries: mean={np.mean(ei_boundary_amps['away_from_boundaries']):.4f}, "
        f"std={np.std(ei_boundary_amps['away_from_boundaries']):.4f}"
    )

    ei_interface_summary, ei_interface_raw = analyze_interface_reflections(
        ei_display, facies_display
    )
    print(f"\n3. Interface Reflection Strength (EI):")
    for key, stats in ei_interface_summary.items():
        if stats is not None and stats["count"] > 10:
            print(
                f"   {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['count']}"
            )

    ei_facies_stats, ei_separation, ei_facies_amps = calculate_facies_discrimination(
        ei_display, facies_display
    )
    print(f"\n4. Facies Discrimination (EI):")
    for facies_val, stats in ei_facies_stats.items():
        print(
            f"   Facies {facies_val}: mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )
    print(
        f"   Average separation (Cohen's d): {np.mean(ei_separation[ei_separation > 0]):.3f}"
    )

    # ===== COMPARISON =====
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS: AVO vs AI vs EI")
    print("=" * 70)

    print(f"\n1. Gradient Correlation (Higher is better):")
    avo_pearson = avo_gradient_corr["pearson_correlation"]
    ai_pearson = ai_gradient_corr["pearson_correlation"]
    ei_pearson = ei_gradient_corr["pearson_correlation"]
    max_pearson = max(avo_pearson, ai_pearson, ei_pearson)
    if max_pearson == avo_pearson:
        winner = "AVO"
    elif max_pearson == ai_pearson:
        winner = "AI"
    else:
        winner = "EI"
    print(f"   Winner: {winner}")
    print(f"   AVO: {avo_pearson:.4f}")
    print(f"   AI: {ai_pearson:.4f}")
    print(f"   EI: {ei_pearson:.4f}")

    print(f"\n2. Amplitude at Boundaries (Higher is better for imaging):")
    avo_boundary_mean = np.mean(np.abs(avo_boundary_amps["at_boundaries"]))
    ai_boundary_mean = np.mean(np.abs(ai_boundary_amps["at_boundaries"]))
    ei_boundary_mean = np.mean(np.abs(ei_boundary_amps["at_boundaries"]))
    max_boundary = max(avo_boundary_mean, ai_boundary_mean, ei_boundary_mean)
    if max_boundary == avo_boundary_mean:
        winner = "AVO"
    elif max_boundary == ai_boundary_mean:
        winner = "AI"
    else:
        winner = "EI"
    print(f"   Winner: {winner}")
    print(f"   AVO: {avo_boundary_mean:.4f}")
    print(f"   AI: {ai_boundary_mean:.4f}")
    print(f"   EI: {ei_boundary_mean:.4f}")

    print(f"\n3. Facies Separation (Higher is better for discrimination):")
    avo_sep_mean = np.mean(avo_separation[avo_separation > 0])
    ai_sep_mean = np.mean(ai_separation[ai_separation > 0])
    ei_sep_mean = np.mean(ei_separation[ei_separation > 0])
    max_sep = max(avo_sep_mean, ai_sep_mean, ei_sep_mean)
    if max_sep == avo_sep_mean:
        winner = "AVO"
    elif max_sep == ai_sep_mean:
        winner = "AI"
    else:
        winner = "EI"
    print(f"   Winner: {winner}")
    print(f"   AVO: {avo_sep_mean:.3f}")
    print(f"   AI: {ai_sep_mean:.3f}")
    print(f"   EI: {ei_sep_mean:.3f}")

    print(f"\n4. FLUID DISCRIMINATION (F0-F3: Brine vs Oil):")
    avo_f0_f3 = avo_separation[0, 3]
    ai_f0_f3 = ai_separation[0, 3]
    ei_f0_f3 = ei_separation[0, 3]
    max_f0_f3 = max(avo_f0_f3, ai_f0_f3, ei_f0_f3)
    if max_f0_f3 == avo_f0_f3:
        winner_fluid = "AVO"
    elif max_f0_f3 == ai_f0_f3:
        winner_fluid = "AI"
    else:
        winner_fluid = "EI"
    print(f"   Winner: {winner_fluid}")
    print(f"   AVO: {avo_f0_f3:.3f}")
    print(f"   AI:  {ai_f0_f3:.3f}")
    print(f"   EI:  {ei_f0_f3:.3f}")

    print(f"\n5. LITHOLOGY DISCRIMINATION (F0-F1: Brine vs Shale):")
    avo_f0_f1 = avo_separation[0, 1]
    ai_f0_f1 = ai_separation[0, 1]
    ei_f0_f1 = ei_separation[0, 1]
    max_f0_f1 = max(avo_f0_f1, ai_f0_f1, ei_f0_f1)
    if max_f0_f1 == avo_f0_f1:
        winner_litho = "AVO"
    elif max_f0_f1 == ai_f0_f1:
        winner_litho = "AI"
    else:
        winner_litho = "EI"
    print(f"   Winner: {winner_litho}")
    print(f"   AVO: {avo_f0_f1:.3f}")
    print(f"   AI:  {ai_f0_f1:.3f}")
    print(f"   EI:  {ei_f0_f1:.3f}")

    # Package results
    avo_results = {
        "gradient_correlation": avo_gradient_corr,
        "boundary_amps": avo_boundary_amps,
        "interface_stats_summary": avo_interface_summary,
        "interface_stats_raw": avo_interface_raw,
        "facies_stats": avo_facies_stats,
        "separation_matrix": avo_separation,
        "facies_amplitudes": avo_facies_amps,
    }

    ai_results = {
        "gradient_correlation": ai_gradient_corr,
        "boundary_amps": ai_boundary_amps,
        "interface_stats_summary": ai_interface_summary,
        "interface_stats_raw": ai_interface_raw,
        "facies_stats": ai_facies_stats,
        "separation_matrix": ai_separation,
        "facies_amplitudes": ai_facies_amps,
    }

    ei_results = {
        "gradient_correlation": ei_gradient_corr,
        "boundary_amps": ei_boundary_amps,
        "interface_stats_summary": ei_interface_summary,
        "interface_stats_raw": ei_interface_raw,
        "facies_stats": ei_facies_stats,
        "separation_matrix": ei_separation,
        "facies_amplitudes": ei_facies_amps,
    }

    # Create visualizations
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATION...")
    print("=" * 70)
    create_summary_plots(
        avo_results, ai_results, ei_results, cache_dir, domain=args.domain
    )

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE:")
    print("=" * 70)
    print(
        "• Pearson/Spearman r: Measures how well seismic gradients align with boundaries"
    )
    print("  - Values closer to 1.0 indicate better alignment")
    print("  - Positive correlation means reflections occur at facies changes")
    print(
        "\n• Amplitude at Boundaries: Higher values = stronger reflections at interfaces"
    )
    print("  - Indicates how well the technique images geological contacts")
    print(
        "\n• Cohen's d (Separation): Measures how distinct facies are in seismic amplitude"
    )
    print("  - d > 0.8 = Large effect (good separation)")
    print("  - d = 0.5-0.8 = Medium effect")
    print("  - d < 0.5 = Small effect (poor separation)")
    print(
        "\n• Interface Reflection Strength: Shows which facies transitions are strongest"
    )
    print("  - Useful for identifying which boundaries are best imaged")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
