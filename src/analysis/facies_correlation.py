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

import numpy as np
import os
import logging
import matplotlib.pyplot as plt

from src.processing.velocity import VelocityModel
from scipy.ndimage import sobel, gaussian_filter
from scipy.stats import pearsonr, spearmanr
from src.io import data_loader
from src.io.grid import GridSpec

logger = logging.getLogger(__name__)

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

# Public API for the analysis module
__all__ = [
    "convert_time_to_depth",
    "impedance_to_seismogram_depth",
    "detect_facies_boundaries",
    "extract_boundary_amplitudes",
    "calculate_gradient_correlation",
    "analyze_interface_reflections",
    "calculate_facies_discrimination",
    "compare_techniques",
    "create_summary_plots",
    "analyze_facies_correlation",
    "main",
]


# Thin object-oriented facade for the facies-correlation utilities
class FaciesCorrelationAnalyzer:
    def convert_time_to_depth(self, seismogram_time, vp_depth, grid_spec: "GridSpec"):
        return _impl_convert_time_to_depth(seismogram_time, vp_depth, grid_spec)

    def impedance_to_seismogram_depth(
        self, impedance, grid_spec: "GridSpec", f_peak=30
    ):
        return _impl_impedance_to_seismogram_depth(impedance, grid_spec, f_peak=f_peak)

    def detect_facies_boundaries(self, facies_cube):
        return _impl_detect_facies_boundaries(facies_cube)

    def extract_boundary_amplitudes(self, seismic_cube, boundaries, window=2):
        return _impl_extract_boundary_amplitudes(
            seismic_cube, boundaries, window=window
        )

    def calculate_gradient_correlation(self, seismic_cube, facies_cube):
        return _impl_calculate_gradient_correlation(seismic_cube, facies_cube)

    def analyze_interface_reflections(self, seismic_cube, facies_cube):
        return _impl_analyze_interface_reflections(seismic_cube, facies_cube)

    def calculate_facies_discrimination(self, seismic_cube, facies_cube):
        return _impl_calculate_facies_discrimination(seismic_cube, facies_cube)

    def compare_techniques(self, avo_stats, ai_stats, metric_name):
        return _impl_compare_techniques(avo_stats, ai_stats, metric_name)

    def create_summary_plots(
        self, avo_results, ai_results, ei_results, cache_dir, domain="depth"
    ):
        return _impl_create_summary_plots(
            avo_results, ai_results, ei_results, cache_dir, domain=domain
        )


from src.utils.facades import LazyObjectProxy


# Module-level singleton (lazy proxy)
facies_correlation_analyzer = LazyObjectProxy(lambda: FaciesCorrelationAnalyzer())


__all__.extend(["FaciesCorrelationAnalyzer", "facies_correlation_analyzer"])


def get_facies_correlation_analyzer(
    instance: FaciesCorrelationAnalyzer | None = None,
) -> "FaciesCorrelationAnalyzer":
    """Return provided FaciesCorrelationAnalyzer or module-level lazy singleton."""
    return instance if instance is not None else facies_correlation_analyzer


__all__.append("get_facies_correlation_analyzer")


# The previous `convert_depth_to_time` thin wrapper was removed in favor of
# calling `DepthTimeResampler` directly. Callers should construct a resampler
# and use `depth_to_time_cube` so there's a single canonical implementation.


def convert_time_to_depth(seismogram_time, vp_depth, grid_spec: GridSpec):
    """
    Convert a time-domain seismogram to depth domain.

    Args:
        seismogram_time: 3D seismogram in time domain (ni, nj, nt)
        vp_depth: 3D P-wave velocity in depth domain (ni, nj, nz)
        grid_spec: GridSpec containing vertical spacing (`dz`) and time
            sampling interval (`dt`) used for conversion.
    # Note: callers should obtain a ResamplePlan from the shared cache
    # via `get_resample_plan_cache().get_plan(...)` when needed.
    Returns:
        3D seismogram in depth domain (ni, nj, nz)
    """
    logger.info("Converting seismogram from time to depth domain...")

    from src.processing.resampler import resampler_factory

    resampler = resampler_factory.get_resampler(grid_spec)
    from src.processing.resample_cache import get_resample_plan_cache

    plan = get_resample_plan_cache().get_plan(grid_spec, vp_depth)
    return resampler.time_to_depth_cube(seismogram_time, vp_depth, plan=plan)


def impedance_to_seismogram_depth(impedance, grid_spec: GridSpec, f_peak=30):
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
    from src.processing.seismic_operator import SeismicOperator

    logger.info(
        "Converting impedance to seismogram (f_peak=%s Hz, depth domain)...", f_peak
    )

    seismogram = SeismicOperator.impedance_to_seismogram_depth(
        impedance, grid_spec.dz, f_peak=f_peak
    )

    logger.info("Seismogram range: [%.6f, %.6f]", seismogram.min(), seismogram.max())
    return seismogram


def detect_facies_boundaries(facies_cube):
    """
    Detect facies boundaries in 3D using edge detection.

    Returns:
        Binary 3D array where True indicates a boundary location
    """
    logger.info("Detecting facies boundaries in 3D...")
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
    logger.info("Extracting amplitudes at facies boundaries...")

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
    logger.info("Calculating gradient correlation...")

    # Ensure both cubes have the same shape (crop to minimum)
    ni_s, nj_s, nk_s = seismic_cube.shape
    ni_f, nj_f, nk_f = facies_cube.shape

    ni = min(ni_s, ni_f)
    nj = min(nj_s, nj_f)
    nk = min(nk_s, nk_f)

    seismic_aligned = seismic_cube[:ni, :nj, :nk]
    facies_aligned = facies_cube[:ni, :nj, :nk]

    logger.debug(
        "  Aligned shapes: seismic=%s, facies=%s",
        seismic_aligned.shape,
        facies_aligned.shape,
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
    logger.info("Analyzing reflection strength at interfaces...")

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
    logger.info("Calculating facies discrimination capability...")

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
    ax12.bar(
        x - width / 2,
        boundary_means,
        width,
        label="At Boundaries",
        color=["steelblue", "coral", "mediumseagreen"],
        alpha=0.8,
    )
    ax12.bar(
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
                ax13.text(
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
                ax14.text(
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
                ax15_ei.text(
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
    logger.info("✓ Saved quantitative analysis to %s", outfn)

    return outfn


def analyze_facies_correlation():
    """CLI-style entry moved from src.__main__.py.

    This function parses common plotting CLI args via start_plot_main,
    loads the appropriate AVO/AI/EI cache files, converts domains as
    necessary, and then runs the analysis helpers defined in this
    module. Returns the path to the generated summary PNG.
    """
    from src.__main__ import ParserFactory
    from src.plotting.helpers.plot import plot_helper

    # os not required at module level; keep os usage local where necessary
    import numpy as np

    args, DATA_PATH, FILE_MAP, grid_spec, compute_boundary_alignment = (
        ParserFactory.start_plot_main(
            description="Quantitative analysis of seismic-facies correlation"
        )
    )

    # Multi-angle default mirrors modeling behavior
    args.use_multiangle = not getattr(args, "no_multiangle", False)

    if args.domain == "time" and args.use_multiangle:
        logger.warning(
            "Note: Time domain analysis requires seismograms, disabling multi-angle"
        )
        args.use_multiangle = False

    cache_dir = args.cache_dir

    # retrieve cache list but we don't use the returned value here

    avo_fn, ai_fn, ei_fn, ei_data_key, ei_type_str, ei_is_depth_domain = (
        plot_helper.select_cache_files(cache_dir, args.domain)
    )

    assert avo_fn is not None, f"No AVO cache file found in {cache_dir}"
    assert ai_fn is not None, f"No AI cache file found in {cache_dir}"
    assert ei_fn is not None, f"No EI cache file found in {cache_dir}"

    logger.info("Loading cache files:")
    logger.info("  AVO: %s", os.path.basename(avo_fn))
    logger.info("  AI: %s", os.path.basename(ai_fn))
    logger.info("  EI: %s (%s)", os.path.basename(ei_fn), ei_type_str)

    avo_cache = np.load(avo_fn)
    ai_cache = np.load(ai_fn)
    ei_cache = np.load(ei_fn)

    if "impedance_depth" in avo_cache:
        avo = avo_cache["impedance_depth"]
    else:
        avo = avo_cache.get("full_stack")

    if "impedance_ai" in ai_cache:
        ai = ai_cache["impedance_ai"]
    else:
        ai = ai_cache.get("seismogram_ai")

    if "ei_product" in ei_cache:
        ei_km = ei_cache["ei_product"]
        if "ei_optimal" in ei_cache:
            ei_optimal = ei_cache["ei_optimal"]
            conversion_factor = np.mean(ei_optimal) / np.mean(ei_km)
            ei = ei_km * conversion_factor
            ei_source = "Weighted Product (converted)"
        else:
            ei = ei_km * 491.0
            ei_source = "Weighted Product (theoretical)"
    elif "ei_optimal" in ei_cache:
        ei = ei_cache["ei_optimal"]
        ei_source = "Variance-weighted optimal"
    else:
        ei = ei_cache[ei_data_key]
        ei_source = "standard multi-angle optimal"

    data_type = "impedances" if args.domain == "depth" else "seismograms"
    logger.info("Loaded %s (%s domain):", data_type, args.domain)
    logger.info("  AVO: %s", getattr(avo, "shape", None))
    logger.info("  AI: %s", getattr(ai, "shape", None))
    logger.info("  EI: %s (%s)", getattr(ei, "shape", None), ei_source)

    # Load velocity model and facies (depth domain)
    dm = data_loader.DatasetManager.from_stanfordsix(DATA_PATH, FILE_MAP, grid_spec)
    props_depth = dm.data
    facies_depth = props_depth["facies"]

    # Build a VelocityModel which handles unit conversion and validation
    vm = VelocityModel.from_dataset(dm, vp_key="vp")
    vp_depth = vm.vp

    if args.domain == "depth":
        logger.info("%s", "\n" + "=" * 70)
        logger.info("QUANTITATIVE ANALYSIS - DEPTH DOMAIN")
        logger.info("%s", "=" * 70)

        avo_display = avo
        ai_display = ai

        if ei_is_depth_domain:
            ei_display = ei
        else:
            # Use the centralized ResamplerFactory (resampler_factory) to
            # obtain a cached DepthTimeResampler for the GridSpec and use the
            # shared ResamplePlan cache to avoid recomputation.
            from src.processing.resampler import resampler_factory
            from src.processing.resample_cache import get_resample_plan_cache

            resampler = resampler_factory.get_resampler(grid_spec)
            plan = get_resample_plan_cache().get_plan(
                grid_spec, vp_depth, target_dt=grid_spec.dt
            )
            ei_display = resampler.time_to_depth_cube(ei, vp_depth, plan=plan)

        facies_display = facies_depth
    else:
        logger.info("%s", "\n" + "=" * 70)
        logger.info("QUANTITATIVE ANALYSIS - TIME DOMAIN")
        logger.info("%s", "=" * 70)

        avo_display = avo
        ai_display = ai
        ei_display = ei

        # Resample facies from depth to time using the VelocityModel convenience
        # which uses DepthTimeResampler under the hood. Preserve categorical
        # handling so integer facies labels are not interpolated incorrectly.
        facies_display, _dt = vm.resample_to_time(
            facies_depth, is_categorical=True, target_dt=grid_spec.dt
        )

    logger.info("\nAVO vs AI vs EI Seismic-Facies Correlation\n")

    # AVO analysis
    avo_gradient_corr = calculate_gradient_correlation(avo_display, facies_display)
    avo_boundary_amps = extract_boundary_amplitudes(
        avo_display, avo_gradient_corr["boundaries"]
    )
    avo_interface_summary, avo_interface_raw = analyze_interface_reflections(
        avo_display, facies_display
    )
    avo_facies_stats, avo_separation, avo_facies_amps = calculate_facies_discrimination(
        avo_display, facies_display
    )

    # AI analysis
    ai_gradient_corr = calculate_gradient_correlation(ai_display, facies_display)
    ai_boundary_amps = extract_boundary_amplitudes(
        ai_display, ai_gradient_corr["boundaries"]
    )
    ai_interface_summary, ai_interface_raw = analyze_interface_reflections(
        ai_display, facies_display
    )
    ai_facies_stats, ai_separation, ai_facies_amps = calculate_facies_discrimination(
        ai_display, facies_display
    )

    # EI analysis
    ei_gradient_corr = calculate_gradient_correlation(ei_display, facies_display)
    ei_boundary_amps = extract_boundary_amplitudes(
        ei_display, ei_gradient_corr["boundaries"]
    )
    ei_interface_summary, ei_interface_raw = analyze_interface_reflections(
        ei_display, facies_display
    )
    ei_facies_stats, ei_separation, ei_facies_amps = calculate_facies_discrimination(
        ei_display, facies_display
    )

    # Create summary plots
    outfn = create_summary_plots(
        {
            "boundary_amps": avo_boundary_amps,
            "gradient_correlation": avo_gradient_corr,
            "separation_matrix": avo_separation,
            "facies_amplitudes": avo_facies_amps,
            "interface_stats_summary": avo_interface_summary,
        },
        {
            "boundary_amps": ai_boundary_amps,
            "gradient_correlation": ai_gradient_corr,
            "separation_matrix": ai_separation,
            "facies_amplitudes": ai_facies_amps,
            "interface_stats_summary": ai_interface_summary,
        },
        {
            "boundary_amps": ei_boundary_amps,
            "gradient_correlation": ei_gradient_corr,
            "separation_matrix": ei_separation,
            "facies_amplitudes": ei_facies_amps,
            "interface_stats_summary": ei_interface_summary,
        },
        cache_dir,
        domain=args.domain,
    )

    return outfn


# --- Implementation aliases for OO facade (keep these after functions are defined)
_impl_convert_time_to_depth = convert_time_to_depth
_impl_impedance_to_seismogram_depth = impedance_to_seismogram_depth
_impl_detect_facies_boundaries = detect_facies_boundaries
_impl_extract_boundary_amplitudes = extract_boundary_amplitudes
_impl_calculate_gradient_correlation = calculate_gradient_correlation
_impl_analyze_interface_reflections = analyze_interface_reflections
_impl_calculate_facies_discrimination = calculate_facies_discrimination
_impl_compare_techniques = compare_techniques
_impl_create_summary_plots = create_summary_plots
_impl_analyze_facies_correlation = analyze_facies_correlation


def main(
    *,
    cache_dir: str = ".cache",
    domain: str = "depth",
    no_multiangle: bool = False,
    verbose: bool = False,
):
    """Programmatic entrypoint for facies correlation analysis.

    Accepts formal parameters (no argv). Mirrors the behavior of
    `analyze_facies_correlation()` but takes explicit keyword args so
    callers like `src.__main__` can delegate programmatically.
    """
    from src.plotting.helpers.plot import plot_helper, default_plot_config
    import os
    import numpy as np
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    plot_cfg = default_plot_config()
    gs = plot_cfg.grid_spec
    DATA_PATH, FILE_MAP, grid_spec = plot_cfg.data_path, plot_cfg.file_map, gs

    # Multi-angle default mirrors modeling behavior
    use_multiangle = not no_multiangle
    if domain == "time" and use_multiangle:
        # Time domain analysis requires seismograms, disable multi-angle
        use_multiangle = False

    # retrieve cache list
    avo_fn, ai_fn, ei_fn, ei_data_key, ei_type_str, ei_is_depth_domain = (
        plot_helper.select_cache_files(cache_dir, domain)
    )

    assert avo_fn is not None, f"No AVO cache file found in {cache_dir}"
    assert ai_fn is not None, f"No AI cache file found in {cache_dir}"
    assert ei_fn is not None, f"No EI cache file found in {cache_dir}"

    logger.info("Loading cache files:")
    logger.info("  AVO: %s", os.path.basename(avo_fn))
    logger.info("  AI: %s", os.path.basename(ai_fn))
    logger.info("  EI: %s (%s)", os.path.basename(ei_fn), ei_type_str)

    avo_cache = np.load(avo_fn)
    ai_cache = np.load(ai_fn)
    ei_cache = np.load(ei_fn)

    if "impedance_depth" in avo_cache:
        avo = avo_cache["impedance_depth"]
    else:
        avo = avo_cache.get("full_stack")

    if "impedance_ai" in ai_cache:
        ai = ai_cache["impedance_ai"]
    else:
        ai = ai_cache.get("seismogram_ai")

    if "ei_product" in ei_cache:
        ei_km = ei_cache["ei_product"]
        if "ei_optimal" in ei_cache:
            ei_optimal = ei_cache["ei_optimal"]
            conversion_factor = np.mean(ei_optimal) / np.mean(ei_km)
            ei = ei_km * conversion_factor
            ei_source = "Weighted Product (converted)"
        else:
            ei = ei_km * 491.0
            ei_source = "Weighted Product (theoretical)"
    elif "ei_optimal" in ei_cache:
        ei = ei_cache["ei_optimal"]
        ei_source = "Variance-weighted optimal"
    else:
        ei = ei_cache[ei_data_key]
        ei_source = "standard multi-angle optimal"

    data_type = "impedances" if domain == "depth" else "seismograms"
    logger.info("Loaded %s (%s domain):", data_type, domain)
    logger.info("  AVO: %s", getattr(avo, "shape", None))
    logger.info("  AI: %s", getattr(ai, "shape", None))
    logger.info("  EI: %s (%s)", getattr(ei, "shape", None), ei_source)

    # Load velocity model and facies (depth domain)
    # `grid_spec` is provided by the plotting config from start_plot_main
    dm = data_loader.DatasetManager.from_stanfordsix(DATA_PATH, FILE_MAP, grid_spec)
    props_depth = dm.data
    facies_depth = props_depth["facies"]

    # Use VelocityModel to handle unit conversion and validation
    vm = VelocityModel.from_dataset(dm, vp_key="vp")
    vp_depth = vm.vp

    if domain == "depth":
        avo_display = avo
        ai_display = ai

        if ei_is_depth_domain:
            ei_display = ei
        else:
            # Resample EI from time to depth using the shared resampler and plan cache
            from src.processing.resampler import resampler_factory
            from src.processing.resample_cache import get_resample_plan_cache

            resampler = resampler_factory.get_resampler(grid_spec)
            plan = get_resample_plan_cache().get_plan(
                grid_spec, vp_depth, target_dt=grid_spec.dt
            )
            ei_display = resampler.time_to_depth_cube(ei, vp_depth, plan=plan)

        facies_display = facies_depth
    else:
        avo_display = avo
        ai_display = ai
        ei_display = ei

        # Resample facies from depth to time using the shared ResamplerFactory
        from src.processing.resampler import resampler_factory
        from src.processing.resample_cache import get_resample_plan_cache

        resampler = resampler_factory.get_resampler(grid_spec)
        plan = get_resample_plan_cache().get_plan(
            grid_spec, vp_depth, target_dt=grid_spec.dt
        )
        facies_display, _dt = resampler.depth_to_time_cube(
            facies_depth, vp_depth, target_dt=grid_spec.dt, plan=plan
        )

    # AVO analysis
    avo_gradient_corr = calculate_gradient_correlation(avo_display, facies_display)
    avo_boundary_amps = extract_boundary_amplitudes(
        avo_display, avo_gradient_corr["boundaries"]
    )
    avo_interface_summary, avo_interface_raw = analyze_interface_reflections(
        avo_display, facies_display
    )
    avo_facies_stats, avo_separation, avo_facies_amps = calculate_facies_discrimination(
        avo_display, facies_display
    )

    # AI analysis
    ai_gradient_corr = calculate_gradient_correlation(ai_display, facies_display)
    ai_boundary_amps = extract_boundary_amplitudes(
        ai_display, ai_gradient_corr["boundaries"]
    )
    ai_interface_summary, ai_interface_raw = analyze_interface_reflections(
        ai_display, facies_display
    )
    ai_facies_stats, ai_separation, ai_facies_amps = calculate_facies_discrimination(
        ai_display, facies_display
    )

    # EI analysis
    ei_gradient_corr = calculate_gradient_correlation(ei_display, facies_display)
    ei_boundary_amps = extract_boundary_amplitudes(
        ei_display, ei_gradient_corr["boundaries"]
    )
    ei_interface_summary, ei_interface_raw = analyze_interface_reflections(
        ei_display, facies_display
    )
    ei_facies_stats, ei_separation, ei_facies_amps = calculate_facies_discrimination(
        ei_display, facies_display
    )

    # Create summary plots
    outfn = create_summary_plots(
        {
            "boundary_amps": avo_boundary_amps,
            "gradient_correlation": avo_gradient_corr,
            "separation_matrix": avo_separation,
            "facies_amplitudes": avo_facies_amps,
            "interface_stats_summary": avo_interface_summary,
        },
        {
            "boundary_amps": ai_boundary_amps,
            "gradient_correlation": ai_gradient_corr,
            "separation_matrix": ai_separation,
            "facies_amplitudes": ai_facies_amps,
            "interface_stats_summary": ai_interface_summary,
        },
        {
            "boundary_amps": ei_boundary_amps,
            "gradient_correlation": ei_gradient_corr,
            "separation_matrix": ei_separation,
            "facies_amplitudes": ei_facies_amps,
            "interface_stats_summary": ei_interface_summary,
        },
        cache_dir,
        domain=domain,
    )

    return outfn
