"""Core modeling routines.

This module contains the heavy compute functions for AVO (convolutions and
noise models). These implementations provide canonical modeling helpers
used by higher-level tooling and scripts.
"""

import hashlib
import numpy as np
from scipy.signal import convolve
from tqdm.auto import tqdm
import sys
import logging
from src.utils.quantity import Quantity
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)

__all__ = [
    "create_avo_synthetics",
    "run_convolution_3d",
    "apply_angle_quality_weighting",
    "ModelingEngine",
    "modeling_engine",
]


def get_modeling_engine(config: dict | None = None):
    """Return the module-level `modeling_engine` when config is None.

    If `config` is provided, create and return a fresh `ModelingEngine`
    instance. This helper centralizes the default access pattern while
    allowing callers to obtain a configured instance when needed.
    """
    if config is None:
        return modeling_engine
    return ModelingEngine()


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


def apply_angle_quality_weighting(angle_stacks, angles, normalize=True):
    """Apply quality weights to angle-dependent stacks.

    Args:
        angle_stacks: List of angle-dependent seismic stacks
        angles: Corresponding angles in degrees
        normalize: Whether to normalize weights to sum to 1

    Returns:
        Weighted stack combining all angles
    """
    if len(angle_stacks) != len(angles):
        raise ValueError("Number of angle stacks must match number of angles")

    weights = np.array([get_angle_weight(a) for a in angles])

    if normalize:
        weights = weights / weights.sum()

    weighted_stack = np.zeros_like(angle_stacks[0])
    for stack, weight in zip(angle_stacks, weights):
        weighted_stack += stack * weight

    return weighted_stack


# ============================================================================
# END OF IMPROVEMENTS
# ============================================================================


def run_convolution_3d(rc_cube, wavelet, use_gpu=True):
    """Run 3D convolution on reflectivity cube with wavelet.

    Args:
        rc_cube: Reflectivity cube
        wavelet: Source wavelet
        use_gpu: Whether to use GPU acceleration (currently unused)

    Returns:
        Convolved seismogram cube
    """

    def convolve_trace(trace):
        return convolve(trace, wavelet, mode="same", method="fft")

    return np.apply_along_axis(convolve_trace, axis=-1, arr=rc_cube)


def create_avo_synthetics(
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


class ModelingEngine:
    """Object-oriented facade for core modeling routines.

    This class provides a minimal, stable API that wraps the existing
    module-level functions. It is intentionally a thin facade so callers
    can adopt an OOP access pattern without changing behaviour.

        Methods mirror the top-level functions in this module:
            - create_avo_synthetics(props_time, angles, wavelet, ...)
            - run_convolution_3d(rc_cube, wavelet, ...)
            - apply_angle_quality_weighting(angle_stacks, angles, ...)

        The instance holds no mutable shared state by default.
    """

    def create_avo_synthetics(self, *args, **kwargs):
        """Create AVO synthetics using the module-level function."""
        return create_avo_synthetics(*args, **kwargs)

    def run_convolution_3d(self, *args, **kwargs):
        """Run 3D convolution using the module-level function."""
        return run_convolution_3d(*args, **kwargs)

    def apply_angle_quality_weighting(self, *args, **kwargs):
        """Apply angle quality weighting using the module-level function."""
        return apply_angle_quality_weighting(*args, **kwargs)

    # Additional thin wrappers to expose more of the module API via the
    # ModelingEngine facade. This allows callers to use the
    # object-oriented proxy `modeling_engine` without changing behaviour.

    def cached_avo(self, *args, **kwargs):
        # Import locally to avoid circular imports
        from src.modeling import model_cache

        return model_cache.cached_avo(*args, **kwargs)

    def cached_avo_depth(self, *args, **kwargs):
        from src.modeling import model_cache

        return model_cache.cached_avo_depth(*args, **kwargs)

    # Technique-specific caching is not part of the current API


# Module-level singleton used by callers that prefer an object instance.
modeling_engine = LazyObjectProxy(lambda: ModelingEngine())


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


# Elastic-style computations are not part of the public API. This module focuses on AVO
# modeling helpers: create_avo_synthetics, convolution and angle weighting.
