"""Core modeling routines.

This module contains the heavy compute functions (AVO/EI computations,
convolutions, and noise models). These implementations provide canonical
modeling helpers used by higher-level tooling and scripts.
"""

import hashlib
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from tqdm.auto import tqdm
import sys
import matplotlib.pyplot as plt
import logging
from src.utils.quantity import Quantity
from typing import Sequence, Optional
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)

# Ensure we warn at most once per process when normalization fallback is used
_ei_normalization_warning_emitted = False

__all__ = [
    "create_avo_synthetics",
    "run_convolution_3d",
    "apply_angle_quality_weighting",
    "ModelingEngine",
    "modeling_engine",
]


def get_modeling_engine(config: dict | None = None, *, use_gpu: bool = True):
    """Return the module-level `modeling_engine` when config is None.

    If `config` is provided, create and return a fresh `ModelingEngine`
    instance configured according to the dict (supports 'use_gpu' key).
    This helper centralizes the default access pattern while allowing callers
    to obtain a configured instance when needed.
    """
    return _impl_get_modeling_engine(config, use_gpu=use_gpu)


def _impl_get_modeling_engine(config: dict | None = None, *, use_gpu: bool = True):
    if config is None:
        return modeling_engine
    me = ModelingEngine()
    # apply simple well-known config keys
    if "use_gpu" in config:
        try:
            me.use_gpu = bool(config["use_gpu"])
        except Exception:
            pass
    else:
        me.use_gpu = bool(use_gpu)
    return me


__all__.append("get_modeling_engine")


# ============================================================================
# AVO MODELING IMPROVEMENTS (Based on Inversion Study - Oct 2025)
# ============================================================================

ANGLE_QUALITY_WEIGHTS = {
    0: 0.90,
    5: 0.95,
    10: 0.98,
    15: 1.00,
    30: 0.70,
    45: 0.40,
}

ANGLE_NOISE_SIGMA = {
    0: 0.011,
    5: 0.007,
    10: 0.004,
    15: 0.002,
    30: 0.033,
    45: 0.023,
}


def get_angle_weight(angle_deg):
    angles_sorted = sorted(ANGLE_QUALITY_WEIGHTS.keys())

    if angle_deg <= angles_sorted[0]:
        return ANGLE_QUALITY_WEIGHTS[angles_sorted[0]]
    if angle_deg >= angles_sorted[-1]:
        return ANGLE_QUALITY_WEIGHTS[angles_sorted[-1]]

    for i in range(len(angles_sorted) - 1):
        a1, a2 = angles_sorted[i], angles_sorted[i + 1]
        if a1 <= angle_deg <= a2:
            w1, w2 = ANGLE_QUALITY_WEIGHTS[a1], ANGLE_QUALITY_WEIGHTS[a2]
            t = (angle_deg - a1) / (a2 - a1)
            return w1 + t * (w2 - w1)

    return 1.0


def get_noise_level(angle_deg):
    angles_sorted = sorted(ANGLE_NOISE_SIGMA.keys())

    if angle_deg <= angles_sorted[0]:
        return ANGLE_NOISE_SIGMA[angles_sorted[0]]
    if angle_deg >= angles_sorted[-1]:
        return ANGLE_NOISE_SIGMA[angles_sorted[-1]]

    for i in range(len(angles_sorted) - 1):
        a1, a2 = angles_sorted[i], angles_sorted[i + 1]
        if a1 <= angle_deg <= a2:
            s1, s2 = ANGLE_NOISE_SIGMA[a1], ANGLE_NOISE_SIGMA[a2]
            t = (angle_deg - a1) / (a2 - a1)
            return s1 + t * (s2 - s1)

    return 0.01


def add_realistic_noise(seismic, angle_deg, snr_db=20, seed=None):
    if seed is not None:
        np.random.seed(seed)

    sigma_systematic = get_noise_level(angle_deg)
    signal_power = np.var(seismic)
    target_snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / target_snr_linear
    noise_random = np.random.randn(*seismic.shape) * np.sqrt(noise_power)
    noise_systematic = np.random.randn(*seismic.shape) * sigma_systematic
    total_noise = noise_random + noise_systematic
    return seismic + total_noise.astype(seismic.dtype)


def _impl_apply_angle_quality_weighting(angle_stacks, angles, normalize=True):
    if len(angle_stacks) != len(angles):
        raise ValueError("Number of angle stacks must match number of angles")

    weights = np.array([get_angle_weight(a) for a in angles])

    if normalize:
        weights = weights / weights.sum()

    weighted_stack = np.zeros_like(angle_stacks[0])
    for stack, weight in zip(angle_stacks, weights):
        weighted_stack += stack * weight

    return weighted_stack


def apply_angle_quality_weighting(angle_stacks, angles, normalize=True):
    """Module-level wrapper kept for backward compatibility.

    Prefer using `modeling_engine.apply_angle_quality_weighting` for OOP
    usage. This thin wrapper delegates to the canonical implementation.
    """
    return _impl_apply_angle_quality_weighting(
        angle_stacks, angles, normalize=normalize
    )


# ============================================================================
# END OF IMPROVEMENTS
# ============================================================================


def run_convolution_3d(rc_cube, wavelet, use_gpu=True):
    return _impl_run_convolution_3d(rc_cube, wavelet, use_gpu=use_gpu)


def _impl_create_avo_synthetics(
    props_time,
    angles,
    wavelet,
    use_quality_weighting=False,
    add_noise=False,
    snr_db=20,
    noise_seed=None,
):
    # Support Quantity inputs for vp/vs/rho by unwrapping to numeric arrays
    vp_val = props_time["vp"]
    vs_val = props_time["vs"]
    rho_val = props_time["rho"]
    vp = vp_val.array if isinstance(vp_val, Quantity) else np.asarray(vp_val)
    vs = vs_val.array if isinstance(vs_val, Quantity) else np.asarray(vs_val)
    rho = rho_val.array if isinstance(rho_val, Quantity) else np.asarray(rho_val)

    ni, nj, nk = vp.shape
    angle_stacks = []
    full_stack = np.zeros((ni, nj, nk), dtype=np.float32)
    n_angles = len(angles)
    bar = tqdm(
        total=len(angles),
        desc="Processing Angles",
        leave=True,
        dynamic_ncols=True,
        file=sys.stderr,
    )
    debug_mode = sys.gettrace() is not None
    block_i = 10
    for idx, angle in enumerate(angles):
        bar.update(1)
        bar.refresh()
        try:
            sys.stderr.flush()
        except Exception:
            pass

        angle_stack_full = np.zeros((ni, nj, nk), dtype=np.float32)

        for i0 in range(0, ni, block_i):
            i1 = min(ni, i0 + block_i)
            vp_block = vp[i0:i1]
            vs_block = vs[i0:i1]
            rho_block = rho[i0:i1]

            vp1b, vp2b = vp_block[..., :-1], vp_block[..., 1:]
            vs1b, vs2b = vs_block[..., :-1], vs_block[..., 1:]
            rho1b, rho2b = rho_block[..., :-1], rho_block[..., 1:]

            from src.signal.reflectivity import zoeppritz_solver

            rc_values = zoeppritz_solver.solve(
                vp1b, vs1b, rho1b, vp2b, vs2b, rho2b, angle
            )
            rc_real = np.real(rc_values).astype(np.float32)
            rc_pad = np.zeros((i1 - i0, nj, nk), dtype=np.float32)
            rc_pad[..., 1:] = rc_real
            angle_block = modeling_engine.run_convolution_3d(rc_pad, wavelet)
            angle_stack_full[i0:i1] = angle_block
            full_stack[i0:i1] += angle_block / float(n_angles)

        if add_noise:
            angle_stack_full = add_realistic_noise(
                angle_stack_full, angle, snr_db=snr_db, seed=noise_seed
            )

        angle_stacks.append(angle_stack_full)

        if debug_mode:
            logger.debug("[DEBUG] Angle %d/%d completed", idx + 1, n_angles)

    bar.close()

    if use_quality_weighting:
        full_stack = modeling_engine.apply_angle_quality_weighting(angle_stacks, angles)

    return angle_stacks, full_stack


def _impl_run_convolution_3d(rc_cube, wavelet, use_gpu=True):
    def convolve_trace(trace):
        return convolve(trace, wavelet, mode="same", method="fft")

    return np.apply_along_axis(convolve_trace, axis=-1, arr=rc_cube)


class ModelingEngine:
    """Object-oriented facade for core modeling routines.

    This class provides a minimal, stable API that wraps the existing
    module-level functions. It is intentionally a thin facade so callers
    can adopt an OOP access pattern without changing behaviour.

    Methods mirror the top-level functions in this module:
      - create_avo_synthetics(props_time, angles, wavelet, ...)
      - run_convolution_3d(rc_cube, wavelet, ...)
      - apply_angle_quality_weighting(angle_stacks, angles, ...)
      - ei_to_seismogram(ei_volume, time_axis, wavelet, ...)
      - add_ei_noise(...)

    The instance holds no mutable shared state by default.
    """

    def __init__(self):
        # Placeholder for future configuration (e.g., GPU toggle)
        self.use_gpu = True

    def create_avo_synthetics(self, *args, **kwargs):
        return _impl_create_avo_synthetics(*args, **kwargs)

    def run_convolution_3d(self, *args, **kwargs):
        return _impl_run_convolution_3d(*args, **kwargs)

    def apply_angle_quality_weighting(self, *args, **kwargs):
        return _impl_apply_angle_quality_weighting(*args, **kwargs)

    def ei_to_seismogram(self, *args, **kwargs):
        return _impl_ei_to_seismogram(*args, **kwargs)

    def add_ei_noise(self, *args, **kwargs):
        return _impl_add_ei_noise(*args, **kwargs)

    # Additional thin wrappers to expose more of the module API via the
    # ModelingEngine facade. This allows callers to use the
    # object-oriented proxy `modeling_engine` without changing behaviour.
    def run_multiangle_analysis(self, *args, **kwargs):
        return run_multiangle_analysis(*args, **kwargs)

    def compute_ei_weighted_product(self, *args, **kwargs):
        return compute_ei_weighted_product(*args, **kwargs)

    def cached_avo(self, *args, **kwargs):
        # Import locally to avoid circular imports
        from src.modeling import model_cache

        return model_cache.cached_avo(*args, **kwargs)

    def cached_ai_seismogram(self, *args, **kwargs):
        from src.modeling import model_cache

        return model_cache.cached_ai_seismogram(*args, **kwargs)

    def cached_avo_depth(self, *args, **kwargs):
        from src.modeling import model_cache

        return model_cache.cached_avo_depth(*args, **kwargs)

    def cached_ai_depth(self, *args, **kwargs):
        from src.modeling import model_cache

        return model_cache.cached_ai_depth(*args, **kwargs)

    def compute_ei_angle(self, *args, **kwargs):
        # Accept older keyword 'angle' for backwards compatibility and
        # forward to the canonical implementation which expects 'angle_deg'.
        if "angle" in kwargs and "angle_deg" not in kwargs:
            kwargs["angle_deg"] = kwargs.pop("angle")
        return compute_ei_angle(*args, **kwargs)

    def compute_ei_multiangle(self, *args, **kwargs):
        # Accept older keyword 'angles' and map to 'angles_deg'
        if "angles" in kwargs and "angles_deg" not in kwargs:
            kwargs["angles_deg"] = kwargs.pop("angles")
        return compute_ei_multiangle(*args, **kwargs)


# Module-level singleton used by callers that prefer an object instance.
modeling_engine = LazyObjectProxy(lambda: ModelingEngine())


def create_avo_synthetics(
    props_time,
    angles,
    wavelet,
    use_quality_weighting=False,
    add_noise=False,
    snr_db=20,
    noise_seed=None,
):
    return _impl_create_avo_synthetics(
        props_time,
        angles,
        wavelet,
        use_quality_weighting=use_quality_weighting,
        add_noise=add_noise,
        snr_db=snr_db,
        noise_seed=noise_seed,
    )


def _hash_for_cache(arrays, extras=None):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    if extras:
        for e in extras:
            if isinstance(e, (list, tuple)):
                h.update(str(list(e)).encode())
            elif isinstance(e, np.ndarray):
                h.update(e.tobytes())
            else:
                h.update(str(e).encode())
    return h.hexdigest()[:20]


def ei_to_seismogram(ei_volume, time_axis, wavelet, show_progress=True):
    return _impl_ei_to_seismogram(
        ei_volume, time_axis, wavelet, show_progress=show_progress
    )


def _impl_ei_to_seismogram(ei_volume, time_axis, wavelet, show_progress=True):
    """Canonical implementation: convert EI volume to seismogram by
    convolving along the time axis with the provided wavelet.

    This keeps the implementation local and avoids recursive delegation to
    the top-level wrapper. The function intentionally performs a minimal
    transformation: ensure numeric arrays and call the optimized 3D
    convolution implementation used elsewhere in this module.
    """
    import numpy as _np

    # Ensure array-like inputs are converted to numpy arrays.
    ei_arr = _np.asarray(ei_volume)

    # Delegate to the canonical 3D convolution implementation which
    # applies the wavelet along the last axis (time/depth).
    return _impl_run_convolution_3d(ei_arr, wavelet)


def create_optimal_ei_stack(ei_results, optimization="variance"):
    from src.processing.ei import ei_processor

    ei_volumes = ei_results["ei_volumes"]
    angles = ei_results["angles"]

    weights = ei_processor.compute_ei_weights(ei_volumes, angles, method=optimization)
    ei_stack, weights = ei_processor.finalize_weighted_stack(ei_volumes, weights)
    return ei_stack, weights


def analyze_facies_correlation_depth(ei_volume, facies):
    from src.processing.ei import (
        analyze_facies_correlation_depth as shared_analyze,
    )

    return shared_analyze(ei_volume, facies)


def run_multiangle_analysis(props_depth, angles_deg: Optional[Sequence[int]] = None):
    from src.processing.ei import ei_processor

    if angles_deg is None:
        angles_deg = [0, 5, 10, 15, 20, 25]

    return ei_processor.compute_and_save_multiangle_ei(
        props_depth,
        angles_deg=angles_deg,
        ei_pc3_fluid=None,
        optimization="cohens_d",
        top_n_angles=4,
        cache_dir=".cache",
        show_progress=True,
        compute_ei_multiangle=compute_ei_multiangle,
        create_optimal_ei_stack=create_optimal_ei_stack,
        analyze_facies_correlation_depth=analyze_facies_correlation_depth,
    )


EI_FREQUENCY_NOISE_SIGMA = {
    20: 0.008,
    25: 0.010,
    30: 0.012,
    35: 0.015,
    40: 0.018,
    50: 0.025,
    60: 0.035,
    70: 0.045,
    80: 0.060,
    95: 0.080,
    100: 0.100,
}

ROCK_PHYSICS_UNCERTAINTY = 0.06


def get_ei_noise_sigma(frequency_hz):
    freq_sorted = sorted(EI_FREQUENCY_NOISE_SIGMA.keys())
    if frequency_hz <= freq_sorted[0]:
        return EI_FREQUENCY_NOISE_SIGMA[freq_sorted[0]]
    if frequency_hz >= freq_sorted[-1]:
        return EI_FREQUENCY_NOISE_SIGMA[freq_sorted[-1]]
    for i in range(len(freq_sorted) - 1):
        f1, f2 = freq_sorted[i], freq_sorted[i + 1]
        if f1 <= frequency_hz <= f2:
            s1 = EI_FREQUENCY_NOISE_SIGMA[f1]
            s2 = EI_FREQUENCY_NOISE_SIGMA[f2]
            t = (frequency_hz - f1) / (f2 - f1)
            return s1 + t * (s2 - s1)
    return 0.012


def add_ei_noise(
    ei_seismic,
    frequency_hz,
    snr_db=None,
    include_rock_physics_error=True,
    spatial_correlation_length=3,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    sigma_freq = get_ei_noise_sigma(frequency_hz)
    signal_power = np.var(ei_seismic)

    if snr_db is not None:
        target_snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / target_snr_linear
        sigma_random = np.sqrt(noise_power)
    else:
        sigma_random = sigma_freq

    random_noise = np.random.normal(0, sigma_random, ei_seismic.shape)

    if include_rock_physics_error:
        uncorrelated_rp = np.random.normal(
            0, ROCK_PHYSICS_UNCERTAINTY * np.std(ei_seismic), ei_seismic.shape
        )
        rock_physics_noise = gaussian_filter(
            uncorrelated_rp, sigma=[spatial_correlation_length] * 3, mode="wrap"
        )
    else:
        rock_physics_noise = 0

    total_noise = random_noise + rock_physics_noise
    noisy_seismic = ei_seismic + total_noise
    return noisy_seismic


def _impl_add_ei_noise(
    ei_seismic,
    frequency_hz,
    snr_db=None,
    include_rock_physics_error=True,
    spatial_correlation_length=3,
    seed=None,
):
    return add_ei_noise(
        ei_seismic,
        frequency_hz,
        snr_db=snr_db,
        include_rock_physics_error=include_rock_physics_error,
        spatial_correlation_length=spatial_correlation_length,
        seed=seed,
    )


def compare_noise_levels(
    clean_seismic, noisy_seismic, facies, title="EI Noise Analysis"
):
    clean_flat = clean_seismic.flatten()
    noisy_flat = noisy_seismic.flatten()
    noise = (noisy_seismic - clean_seismic).flatten()
    facies_flat = facies.flatten()
    r_clean, _ = pearsonr(np.abs(clean_flat), facies_flat)
    r_noisy, _ = pearsonr(np.abs(noisy_flat), facies_flat)
    signal_power = np.var(clean_flat)
    noise_power = np.var(noise)
    snr_db = 10 * np.log10(signal_power / noise_power)
    correlation_loss = (r_clean - r_noisy) / r_clean * 100
    stats = {
        "snr_db": snr_db,
        "correlation_clean": r_clean,
        "correlation_noisy": r_noisy,
        "correlation_loss_pct": correlation_loss,
        "signal_std": np.std(clean_flat),
        "noise_std": np.std(noise),
    }
    return stats


def visualize_noise_impact(
    clean_seismic,
    noisy_seismic,
    facies,
    inline_idx=75,
    output_file=".cache/ei_noise_analysis.png",
):
    noise = noisy_seismic - clean_seismic
    from src.plotting.helpers.plot import plot_helper

    fig, axes = plot_helper.create_figure_grid(figsize=(18, 10), nrows=2, ncols=3)
    vmin, vmax = np.percentile(clean_seismic, [2, 98])
    im1 = axes[0, 0].imshow(
        clean_seismic[inline_idx, :, :].T,
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im1, ax=axes[0, 0], label="Amplitude")
    im2 = axes[0, 1].imshow(
        noisy_seismic[inline_idx, :, :].T,
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im2, ax=axes[0, 1], label="Amplitude")
    noise_vmax = np.percentile(np.abs(noise), 99)
    im3 = axes[0, 2].imshow(
        noise[inline_idx, :, :].T,
        aspect="auto",
        cmap="seismic",
        vmin=-noise_vmax,
        vmax=noise_vmax,
    )
    plt.colorbar(im3, ax=axes[0, 2], label="Amplitude")
    axes[1, 0].hist(
        clean_seismic.flatten(), bins=100, alpha=0.5, label="Clean", density=True
    )
    axes[1, 0].hist(
        noisy_seismic.flatten(), bins=100, alpha=0.5, label="Noisy", density=True
    )
    step = 10
    clean_sub = clean_seismic[::step, ::step, ::step].flatten()
    noisy_sub = noisy_seismic[::step, ::step, ::step].flatten()
    facies_sub = facies[::step, ::step, ::step].flatten()
    scatter = axes[1, 1].scatter(
        clean_sub, noisy_sub, c=facies_sub, cmap="tab10", alpha=0.3, s=1
    )
    plt.colorbar(scatter, ax=axes[1, 1], label="Facies")
    facies_ids = np.unique(facies)
    noise_by_facies = []
    facies_labels = []
    for fid in facies_ids:
        mask = facies == fid
        facies_noise = noise[mask]
        noise_by_facies.append(facies_noise)
        facies_labels.append(f"Facies {int(fid)}")
    bp = axes[1, 2].boxplot(noise_by_facies, labels=facies_labels, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(facies_ids)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    return fig


def frequency_noise_analysis(
    frequencies, output_file=".cache/ei_frequency_noise_curve.png"
):
    noise_sigmas = [get_ei_noise_sigma(f) for f in frequencies]
    snr_dbs = [10 * np.log10(1 / (sigma**2)) for sigma in noise_sigmas]
    from src.plotting.helpers.plot import plot_helper

    fig, axes = plot_helper.create_figure_grid(figsize=(14, 5), nrows=1, ncols=2)
    axes[0].plot(frequencies, noise_sigmas, "o-", linewidth=2, markersize=8)
    axes[1].plot(frequencies, snr_dbs, "o-", linewidth=2, markersize=8, color="green")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    return fig


def compute_ei_angle(vp, vs, rho, angle_deg):
    # Accept Quantity wrappers for vp/vs/rho and coerce to SI numeric arrays
    if isinstance(vp, Quantity):
        vp_arr = vp.to("m/s", copy=True).array
    else:
        vp_arr = np.asarray(vp)
    if isinstance(vs, Quantity):
        vs_arr = vs.to("m/s", copy=True).array
    else:
        vs_arr = np.asarray(vs)
    if isinstance(rho, Quantity):
        rho_arr = rho.to("kg/m3", copy=True).array
    else:
        rho_arr = np.asarray(rho)

    # Numerically stable computation using log-space to avoid overflow/underflow
    theta_rad = np.deg2rad(float(angle_deg))
    sin_theta = np.sin(theta_rad)
    tan_theta = np.tan(theta_rad)
    sin2_theta = sin_theta**2
    tan2_theta = tan_theta**2

    # Ensure inputs are finite and positive where physically required.
    # Use a small floor to avoid log(0) or division-by-zero.
    small_floor = 1e-12
    vp_safe = np.asarray(vp_arr, dtype=float)
    vs_safe = np.asarray(vs_arr, dtype=float)
    rho_safe = np.asarray(rho_arr, dtype=float)

    # Replace NaNs/infs with zeros (we'll floor later) but log and warn if present
    if (
        not np.all(np.isfinite(vp_safe))
        or not np.all(np.isfinite(vs_safe))
        or not np.all(np.isfinite(rho_safe))
    ):
        logger.warning(
            "compute_ei_angle: non-finite vp/vs/rho detected; treating as zeros"
        )
        vp_safe = np.nan_to_num(vp_safe, nan=0.0, posinf=0.0, neginf=0.0)
        vs_safe = np.nan_to_num(vs_safe, nan=0.0, posinf=0.0, neginf=0.0)
        rho_safe = np.nan_to_num(rho_safe, nan=0.0, posinf=0.0, neginf=0.0)

    # Negative or zero physical properties are invalid for log-based formula
    if np.any(vp_safe <= 0) or np.any(vs_safe <= 0) or np.any(rho_safe <= 0):
        logger.debug(
            "compute_ei_angle: non-positive vp/vs/rho values encountered; "
            "applying floor=%g",
            small_floor,
        )
    vp_floor = np.maximum(vp_safe, small_floor)
    vs_floor = np.maximum(vs_safe, small_floor)
    rho_floor = np.maximum(rho_safe, small_floor)

    # Stable K computation
    K = (vs_floor / vp_floor) ** 2

    a = 1.0 + tan2_theta
    b = -8.0 * K * sin2_theta
    c = 1.0 - 4.0 * K * sin2_theta

    # Compute in log-space: log(ei) = a*log(vp) + b*log(vs) + c*log(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vp = np.log(vp_floor)
        log_vs = np.log(vs_floor)
        log_rho = np.log(rho_floor)

    # Broadcast a/b/c if needed (b depends on K array)
    log_ei = a * log_vp + b * log_vs + c * log_rho

    # Clip log_ei to avoid overflow in exp
    max_log = 700.0
    log_ei = np.clip(log_ei, -max_log, max_log)

    ei = np.exp(log_ei)
    ei = np.nan_to_num(ei, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return ei


def compute_ei_multiangle(vp, vs, rho, angles_deg, show_progress=True):
    # Accept Quantity wrappers and coerce to numeric arrays for computation
    if isinstance(vp, Quantity):
        vp_arr = vp.to("m/s", copy=True).array
    else:
        vp_arr = np.asarray(vp)
    if isinstance(vs, Quantity):
        vs_arr = vs.to("m/s", copy=True).array
    else:
        vs_arr = np.asarray(vs)
    if isinstance(rho, Quantity):
        rho_arr = rho.to("kg/m3", copy=True).array
    else:
        rho_arr = np.asarray(rho)

    angles_array = np.array(angles_deg, dtype=float)
    if angles_array.size == 0:
        raise ValueError("compute_ei_multiangle: empty angles list")
    ei_volumes = []
    iterator = tqdm(angles_array, desc="Computing EI angles", disable=not show_progress)
    for angle in iterator:
        ei_angle = compute_ei_angle(vp_arr, vs_arr, rho_arr, angle)
        ei_volumes.append(ei_angle)
    weights = np.cos(np.deg2rad(angles_array))
    wsum = weights.sum()
    if not np.isfinite(wsum) or wsum == 0:
        logger.warning(
            "compute_ei_multiangle: angle weights sum to zero/invalid; "
            "falling back to uniform weights"
        )
        weights = np.ones_like(weights) / float(len(weights))
    else:
        weights = weights / wsum
    ei_stack = np.zeros_like(ei_volumes[0])
    for ei_vol, weight in zip(ei_volumes, weights):
        ei_stack += weight * ei_vol
    ei_near = ei_volumes[0]
    ei_far = ei_volumes[-1]
    ei_gradient = ei_far - ei_near
    from src.processing.ei import ei_processor

    extra = {
        "vp_range": [float(vp_arr.min()), float(vp_arr.max())],
        "vs_range": [float(vs_arr.min()), float(vs_arr.max())],
        "rho_range": [float(rho_arr.min()), float(rho_arr.max())],
    }
    # Wrap EI outputs in Quantity with a descriptive unit label so callers
    # can carry unit metadata. The EI metric is effectively unitless but we
    # label it 'EI' for clarity.
    try:
        from src.utils.quantity import Quantity as _Quantity

        ei_volumes_q = [_Quantity(v, "EI") for v in ei_volumes]
        ei_stack_q = _Quantity(ei_stack, "EI")
        ei_gradient_q = _Quantity(ei_gradient, "EI")
    except Exception:
        # If Quantity cannot be imported for any reason, fall back to raw arrays
        ei_volumes_q = ei_volumes
        ei_stack_q = ei_stack
        ei_gradient_q = ei_gradient

    results = ei_processor.package_ei_results(
        ei_volumes=ei_volumes_q,
        angles_array=angles_array,
        ei_stack=ei_stack_q,
        ei_gradient=ei_gradient_q,
        weights=weights,
        extra_config=extra,
    )
    return results


def compute_ei_weighted_product(
    vp,
    vs,
    rho,
    litho_angles=None,
    fluid_angles=None,
    litho_weight=0.7,
    fluid_weight=0.3,
    show_progress=True,
):
    if litho_angles is None:
        litho_angles = [15, 10, 20, 25]
    if fluid_angles is None:
        fluid_angles = [30, 35, 25, 40]
    ei_litho_result = compute_ei_multiangle(
        vp, vs, rho, litho_angles, show_progress=show_progress
    )
    ei_litho = ei_litho_result["ei_stack"]
    ei_fluid_result = compute_ei_multiangle(
        vp, vs, rho, fluid_angles, show_progress=show_progress
    )
    ei_fluid = ei_fluid_result["ei_stack"]

    # Unwrap Quantity if present
    if isinstance(ei_litho, Quantity):
        ei_litho_arr = ei_litho.array
    else:
        ei_litho_arr = np.asarray(ei_litho)
    if isinstance(ei_fluid, Quantity):
        ei_fluid_arr = ei_fluid.array
    else:
        ei_fluid_arr = np.asarray(ei_fluid)

    # Robust normalization: avoid division by zero or NaN means.
    import logging

    _logger = logging.getLogger(__name__)

    def _safe_mean_abs(x: np.ndarray) -> float:
        # prefer nanmean to ignore NaNs; fall back to a tiny epsilon if result
        # is zero or non-finite to avoid invalid divisions downstream.
        m = np.nanmean(np.abs(x))
        if not np.isfinite(m) or m == 0:
            # use a small epsilon relative to the dynamic range when possible
            try:
                max_abs = np.nanmax(np.abs(x))
            except Exception:
                max_abs = 0.0
            eps = max_abs * 1e-12 if max_abs > 0 else 1e-12
            # Emit warning at most once per process to avoid noisy logs
            global _ei_normalization_warning_emitted
            if not _ei_normalization_warning_emitted:
                _logger.warning(
                    "EI normalization: input has zero/invalid mean; "
                    "falling back to epsilon=%g",
                    eps,
                )
                _ei_normalization_warning_emitted = True
            return float(eps)
        return float(m)

    litho_mean = _safe_mean_abs(ei_litho_arr)
    fluid_mean = _safe_mean_abs(ei_fluid_arr)

    ei_litho_norm = ei_litho_arr / litho_mean
    ei_fluid_norm = ei_fluid_arr / fluid_mean
    ei_litho_abs = np.abs(ei_litho_norm)
    ei_fluid_abs = np.abs(ei_fluid_norm)
    ei_product_norm = (ei_litho_abs**litho_weight) * (ei_fluid_abs**fluid_weight)
    ei_product_norm = np.sign(ei_litho_norm) * ei_product_norm
    # Use the same safe means for scaling (these correspond to the means of
    # the absolute EI arrays used during normalization).
    scale_factor = (litho_mean**litho_weight) * (fluid_mean**fluid_weight)
    ei_product = ei_product_norm * scale_factor

    # Wrap outputs as Quantity('EI') where appropriate
    try:
        from src.utils.quantity import Quantity as _Quantity

        ei_product_q = _Quantity(ei_product, "EI")
        ei_litho_q = _Quantity(ei_litho_arr, "EI")
        ei_fluid_q = _Quantity(ei_fluid_arr, "EI")
    except Exception:
        ei_product_q = ei_product
        ei_litho_q = ei_litho_arr
        ei_fluid_q = ei_fluid_arr

    results = {
        "ei_product": ei_product_q,
        "ei_litho": ei_litho_q,
        "ei_fluid": ei_fluid_q,
        "litho_angles": litho_angles,
        "fluid_angles": fluid_angles,
        "config": {
            "litho_angles": litho_angles,
            "fluid_angles": fluid_angles,
            "litho_weight": litho_weight,
            "fluid_weight": fluid_weight,
            "method": "Weighted Product EI",
        },
    }
    return results


__all__ = [
    "create_avo_synthetics",
    "ei_to_seismogram",
    "compute_ei_angle",
    "compute_ei_multiangle",
    "create_optimal_ei_stack",
    "compute_ei_weighted_product",
    "run_multiangle_analysis",
    "add_ei_noise",
    "visualize_noise_impact",
    "frequency_noise_analysis",
    "run_convolution_3d",
    "apply_angle_quality_weighting",
    "ModelingEngine",
    "modeling_engine",
]
