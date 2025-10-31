"""Quantitative analysis of seismic-facies correlation.

This script performs statistical analysis to measure:
1. How well seismic amplitudes correlate with facies boundaries
2. Reflection strength at facies interfaces
3. Comparative performance of AVO techniques
4. Facies discrimination capability

Usage:
    python -m src.analyze_facies_correlation
        # Default: depth domain (AVO analysis)
    python -m src.analyze_facies_correlation --domain time
        # Time domain (AVO analysis uses seismograms)
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
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)


# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

# Public API for the analysis module
__all__ = [
    "convert_time_to_depth",
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

    def compare_techniques(self, avo_stats, metric_name):
        return _impl_compare_techniques(avo_stats, metric_name)

    def create_summary_plots(self, avo_results, cache_dir, domain="depth"):
        return _impl_create_summary_plots(avo_results, cache_dir, domain=domain)


# Module-level singleton (lazy proxy)
facies_correlation_analyzer = LazyObjectProxy(lambda: FaciesCorrelationAnalyzer())


__all__.extend(["FaciesCorrelationAnalyzer", "facies_correlation_analyzer"])


def get_facies_correlation_analyzer(
    instance: FaciesCorrelationAnalyzer | None = None,
) -> "FaciesCorrelationAnalyzer":
    """Return provided FaciesCorrelationAnalyzer or module-level lazy singleton."""
    return _impl_get_facies_correlation_analyzer(instance)


__all__.append("get_facies_correlation_analyzer")


def _impl_get_facies_correlation_analyzer(
    instance: FaciesCorrelationAnalyzer | None = None,
) -> FaciesCorrelationAnalyzer:
    """Canonical implementation for obtaining the FaciesCorrelationAnalyzer.

    Returns the provided instance when not None, otherwise returns the
    module-level `facies_correlation_analyzer` lazy proxy. Using a single
    `_impl_*` entrypoint simplifies dependency injection and testing.
    """
    return instance if instance is not None else facies_correlation_analyzer


# Prefer using `DepthTimeResampler` directly for depth/time conversions.
# Callers can construct a resampler and use `depth_to_time_cube` for a
# single canonical implementation.


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


def compare_techniques(avo_stats, metric_name):
    """Return a concise AVO-only comparison for the requested metric.

    This repository has been adjusted to focus on AVO results. The
    function returns AVO-centric numbers and a short summary for a given
    metric.
    """

    if metric_name == "gradient_correlation":
        return {
            "AVO": {
                "Pearson": avo_stats.get("pearson_correlation"),
                "Spearman": avo_stats.get("spearman_correlation"),
            },
            "Winner": "AVO",
            "Difference": 0.0,
        }

    # Fallback: return the raw AVO stats under a single key
    return {"AVO": avo_stats}


def create_summary_plots(avo_results, cache_dir, domain="depth"):
    """Create visualization of AVO-only analysis results.

    Parameters
    - avo_results: dict with keys used by the plotting code (boundary_amps,
      interface_stats_summary, facies_amplitudes, separation_matrix)
    - cache_dir: unused here; present for caller convenience
    - domain: 'depth' or 'time'
    """

    fig = plt.figure(figsize=(18, 12))
    domain_label = "Depth Domain" if domain == "depth" else "Time Domain"
    fig.suptitle(
        f"Quantitative Seismic-Facies Correlation Analysis: AVO Only ({domain_label})",
        fontsize=16,
        y=0.995,
    )

    # 1. AVO amplitude distribution (at boundaries vs away)
    ax1 = plt.subplot(2, 3, 1)
    at_bounds = avo_results.get("boundary_amps", {}).get("at_boundaries", np.array([]))
    away = avo_results.get("boundary_amps", {}).get(
        "away_from_boundaries", np.array([])
    )
    if at_bounds.size:
        ax1.hist(
            at_bounds,
            bins=50,
            alpha=0.7,
            label="At Boundaries",
            density=True,
            color="red",
        )
    if away.size:
        ax1.hist(
            away,
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

    # 2. Reflection strength at different interface types (AVO)
    ax2 = plt.subplot(2, 3, 2)
    interface_types = []
    avo_means = []
    avo_stds = []
    for key, stats in (avo_results.get("interface_stats_summary") or {}).items():
        if stats is not None and stats.get("count", 0) > 10:
            interface_types.append(key)
            avo_means.append(stats.get("mean", 0.0))
            avo_stds.append(stats.get("std", 0.0))

    x_pos = np.arange(len(interface_types))
    if len(x_pos):
        ax2.bar(
            x_pos, avo_means, yerr=avo_stds, alpha=0.7, color="steelblue", capsize=5
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(interface_types, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean Amplitude")
    ax2.set_title("AVO: Reflection Strength at Interfaces")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Facies discrimination - amplitude by facies type (AVO)
    ax3 = plt.subplot(2, 3, 3)
    facies_labels = []
    avo_facies_data = []
    for facies_val in range(4):
        facies_data = (avo_results.get("facies_amplitudes") or {}).get(facies_val)
        if facies_data is not None:
            facies_labels.append(f"Facies {facies_val}")
            sampled = facies_data[:: max(1, len(facies_data) // 1000)]
            avo_facies_data.append(sampled)

    if avo_facies_data:
        bp = ax3.boxplot(avo_facies_data, labels=facies_labels, patch_artist=True)
        for patch, color in zip(
            bp["boxes"], plt.cm.tab10(np.linspace(0, 0.4, len(bp["boxes"])))
        ):
            patch.set_facecolor(color)
    ax3.set_ylabel("AVO Amplitude")
    ax3.set_title("AVO: Amplitude by Facies Type")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Boundary amplitude comparison (AVO only: at vs away)
    ax4 = plt.subplot(2, 3, 4)
    boundary_mean = np.nan
    away_mean = np.nan
    if at_bounds.size:
        boundary_mean = np.mean(np.abs(at_bounds))
    if away.size:
        away_mean = np.mean(np.abs(away))

    labels = ["At Boundaries", "Away from Boundaries"]
    values = [
        boundary_mean if not np.isnan(boundary_mean) else 0.0,
        away_mean if not np.isnan(away_mean) else 0.0,
    ]
    ax4.bar(labels, values, color=["steelblue", "lightsteelblue"], alpha=0.8)
    ax4.set_ylabel("Mean |Amplitude|")
    ax4.set_title("Boundary vs Background Amplitude (AVO)")
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Facies separation matrix (Cohen's d) for AVO
    ax5 = plt.subplot(2, 3, 5)
    sep = avo_results.get("separation_matrix")
    if sep is not None:
        ax5.imshow(sep, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
        ax5.set_xticks([0, 1, 2, 3])
        ax5.set_yticks([0, 1, 2, 3])
        ax5.set_xticklabels(["F0", "F1", "F2", "F3"])
        ax5.set_yticklabels(["F0", "F1", "F2", "F3"])
        ax5.set_xlabel("Facies")
        ax5.set_ylabel("Facies")
        ax5.set_title("AVO: Facies Separation (Cohen's d)")
        # Add text annotations when matrix is small
        if hasattr(sep, "shape") and sep.shape == (4, 4):
            for i in range(4):
                for j in range(4):
                    ax5.text(
                        j, i, f"{sep[i, j]:.2f}", ha="center", va="center", fontsize=8
                    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def analyze_facies_correlation():
    """CLI-style entry moved from src.__main__.py.

    This function parses common plotting CLI args via start_plot_main,
    loads the appropriate AVO cache file, converts domains as
    necessary, and then runs the analysis helpers defined in this
    module. Returns the path to the generated summary PNG.
    """
    from src.__main__ import ParserFactory

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

    # retrieve AVO cache file
    from src.plotting.helpers.plot import select_cache_files

    avo_fn = select_cache_files(cache_dir, args.domain)[0]

    assert avo_fn is not None, f"No AVO cache file found in {cache_dir}"

    logger.info("Loading cache file:")
    logger.info("  AVO: %s", os.path.basename(avo_fn))

    avo_cache = np.load(avo_fn)

    avo = avo_cache.get("full_stack")
    data_type = "seismograms" if args.domain == "time" else "seismograms"
    logger.info("Loaded %s (%s domain):", data_type, args.domain)
    logger.info("  AVO: %s", getattr(avo, "shape", None))

    # Load velocity model and facies (depth domain)
    dm = data_loader.DatasetManager.from_stanfordsix(DATA_PATH, FILE_MAP, grid_spec)
    props_depth = dm.data
    facies_depth = props_depth["facies"]

    # Build a VelocityModel which handles unit conversion and validation
    vm = VelocityModel.from_dataset(dm, vp_key="vp")

    if args.domain == "depth":
        logger.info("%s", "\n" + "=" * 70)
        logger.info("QUANTITATIVE ANALYSIS - DEPTH DOMAIN")
        logger.info("%s", "=" * 70)
        avo_display = avo
        facies_display = facies_depth
    else:
        logger.info("%s", "\n" + "=" * 70)
        logger.info("QUANTITATIVE ANALYSIS - TIME DOMAIN")
        logger.info("%s", "=" * 70)
        avo_display = avo

        # Resample facies from depth to time using the VelocityModel convenience
        # which uses DepthTimeResampler under the hood. Preserve categorical
        # handling so integer facies labels are not interpolated incorrectly.
        facies_display, _dt = vm.resample_to_time(
            facies_depth, is_categorical=True, target_dt=grid_spec.dt
        )

    logger.info("\nAVO Seismic-Facies Correlation\n")

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
    # Create summary plots (AVO-only)
    outfn = create_summary_plots(
        {
            "boundary_amps": avo_boundary_amps,
            "gradient_correlation": avo_gradient_corr,
            "separation_matrix": avo_separation,
            "facies_amplitudes": avo_facies_amps,
            "interface_stats_summary": avo_interface_summary,
        },
        cache_dir,
        domain=args.domain,
    )

    return outfn


# --- Implementation aliases for OO facade (keep these after functions are defined)
_impl_convert_time_to_depth = convert_time_to_depth
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
    from src.plotting.helpers.plot import default_plot_config
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

    # retrieve AVO cache
    from src.plotting.helpers.plot import select_cache_files

    avo_fn = select_cache_files(cache_dir, domain)

    assert avo_fn is not None, f"No AVO cache file found in {cache_dir}"

    logger.info("Loading cache file:")
    logger.info("  AVO: %s", os.path.basename(avo_fn))

    avo_cache = np.load(avo_fn)

    avo = avo_cache.get("full_stack")
    data_type = "seismograms" if domain == "time" else "seismograms"
    logger.info("Loaded %s (%s domain):", data_type, domain)
    logger.info("  AVO: %s", getattr(avo, "shape", None))

    # Load velocity model and facies (depth domain)
    # `grid_spec` is provided by the plotting config from start_plot_main
    dm = data_loader.DatasetManager.from_stanfordsix(DATA_PATH, FILE_MAP, grid_spec)
    props_depth = dm.data
    facies_depth = props_depth["facies"]

    # Use VelocityModel to handle unit conversion and validation
    vm = VelocityModel.from_dataset(dm, vp_key="vp")

    if domain == "depth":
        logger.info("%s", "\n" + "=" * 70)
        logger.info("QUANTITATIVE ANALYSIS - DEPTH DOMAIN")
        logger.info("%s", "=" * 70)

        avo_display = avo
        facies_display = facies_depth
    else:
        logger.info("%s", "\n" + "=" * 70)
        logger.info("QUANTITATIVE ANALYSIS - TIME DOMAIN")
        logger.info("%s", "=" * 70)

        avo_display = avo

        # Resample facies from depth to time using the VelocityModel convenience
        # which uses DepthTimeResampler under the hood. Preserve categorical
        # handling so integer facies labels are not interpolated incorrectly.
        facies_display, _dt = vm.resample_to_time(
            facies_depth, is_categorical=True, target_dt=grid_spec.dt
        )

    logger.info("\nAVO Seismic-Facies Correlation\n")

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

    # Create summary plots (AVO-only)
    outfn = create_summary_plots(
        {
            "boundary_amps": avo_boundary_amps,
            "gradient_correlation": avo_gradient_corr,
            "separation_matrix": avo_separation,
            "facies_amplitudes": avo_facies_amps,
            "interface_stats_summary": avo_interface_summary,
        },
        cache_dir,
        domain=domain,
    )

    return outfn
