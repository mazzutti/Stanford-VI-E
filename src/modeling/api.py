"""Programmatic API for running the full modeling pipeline.

This module provides a single entrypoint `run_full_modeling` which encapsulates
the orchestration previously duplicated across callers. It keeps the heavy
compute logic in `src.modeling` and the data loading in `src.io` while
exposing a small, testable function for external callers (for example
`src.analysis.seismograms`).
"""

from __future__ import annotations

import os
import hashlib
import numpy as np

from src.io import data_loader
from src.io.grid import GridSpec
from src.modeling import modeling as modeling_utils
from src.signal import wavelets
from src.signal.reflectivity import reflectivity_calc
from src.processing.seismic_operator import SeismicOperator
from src.utils.units import UnitRegistry
from src.utils.quantity import Quantity
import logging


def run_full_modeling(
    cache_dir: str = ".cache",
    skip_cleanup: bool = False,
    verbose: bool = False,
    add_avo_noise: bool = False,
    add_ei_noise: bool = False,
    ei_noise_snr: float | None = None,
    ei_noise_seed: int | None = None,
):
    """Run the full modeling pipeline in-process and save caches.

    Returns a dict with keys similar to the previous pipeline output, for
    example: {'ei_cache_file': ..., 'save_dict': ..., 'ei_angle_seismograms': ...}
    """
    # Delegate to facade implementation
    return modeling_api.run_full_modeling(
        cache_dir=cache_dir,
        skip_cleanup=skip_cleanup,
        verbose=verbose,
        add_avo_noise=add_avo_noise,
        add_ei_noise=add_ei_noise,
        ei_noise_snr=ei_noise_snr,
        ei_noise_seed=ei_noise_seed,
    )


def _impl_run_full_modeling(
    cache_dir: str = ".cache",
    skip_cleanup: bool = False,
    verbose: bool = False,
    add_avo_noise: bool = False,
    add_ei_noise: bool = False,
    ei_noise_snr: float | None = None,
    ei_noise_seed: int | None = None,
):
    # Original implementation from run_full_modeling (kept as canonical impl)
    # Load depth data defaults (kept intentionally simple)
    DATA_PATH = "."
    FILE_MAP = {
        "vp": "P-wave Velocity",
        "vs": "S-wave Velocity",
        "rho": "Density",
        "facies": "Facies",
    }
    # Default grid spec for standalone runs (inline to avoid module-level constants)
    grid_spec = GridSpec((150, 200, 200), dz=1.0, dt=0.001)
    dm = data_loader.DatasetManager.from_stanfordsix(DATA_PATH, FILE_MAP, grid_spec)
    props_depth = dm.data

    # Normalize units and validate properties via RockPhysicsModel
    from src.processing.rock_physics import RockPhysicsModel

    # Build RockPhysicsModel and normalize units (this stores Quantity internally)
    rpm = RockPhysicsModel.from_props(props_depth, grid_spec)
    rpm.ensure_units()
    # When downstream code expects raw arrays, use rpm.to_props_dict(); but
    # prefer keeping rpm available so consumers can access Quantity metadata.
    props_depth = rpm.to_props_dict()

    # Multi-angle EI (depth)
    ei_multiangle_results = modeling_utils.modeling_engine.run_multiangle_analysis(
        props_depth, angles_deg=[0, 5, 10, 15, 20, 25]
    )
    ei_depth = ei_multiangle_results.get("ei_optimal")
    props_depth["ei"] = ei_depth

    # Weighted product EI
    # Ensure canonical SI units first, then produce km/s representations
    # props_depth may already hold numpy arrays (to_props_dict returns arrays).
    vp_si, _ = UnitRegistry.ensure_m_per_s(props_depth["vp"], copy_on_convert=True)
    vs_si, _ = UnitRegistry.ensure_m_per_s(props_depth["vs"], copy_on_convert=True)
    vp_kms = vp_si / 1000.0
    vs_kms = vs_si / 1000.0
    weighted_results = modeling_utils.modeling_engine.compute_ei_weighted_product(
        vp_kms,
        vs_kms,
        props_depth["rho"],
        litho_angles=[15, 10, 20, 25],
        fluid_angles=[30, 35, 25, 40],
        litho_weight=0.7,
        fluid_weight=0.3,
        show_progress=not verbose,
    )

    props_depth["ei_litho"] = weighted_results["ei_litho"]
    props_depth["ei_fluid"] = weighted_results["ei_fluid"]
    props_depth["ei_product"] = weighted_results["ei_product"]

    # Update EI cache with weighted product if a cache was produced
    ei_cache_file = ei_multiangle_results.get("cache_file")
    if ei_cache_file:
        try:
            ei_cache_data = dict(np.load(ei_cache_file))
            ei_cache_data["ei_litho"] = weighted_results["ei_litho"]
            ei_cache_data["ei_fluid"] = weighted_results["ei_fluid"]
            ei_cache_data["ei_product"] = weighted_results["ei_product"]
            ei_cache_data["weighted_config"] = str(weighted_results.get("config"))
            from src.io.cache import cache_for_dir

            cache_for_dir(cache_dir).save_npz(ei_cache_file, ei_cache_data)
        except Exception:
            pass

    # Depth -> time resampling using DepthTimeResampler
    from src.processing.resampler import resampler_factory

    resampler = resampler_factory.get_resampler(grid_spec)

    # Unwrap vp (handle Quantity or ndarray)
    vp_val = props_depth["vp"].array if hasattr(props_depth["vp"], "array") else np.asarray(props_depth["vp"])  # type: ignore
    ni, nj, nz = vp_val.shape

    # Use a shared ResamplePlanCache to avoid recomputing the same plan
    # across repeated modeling runs within the same process.
    from src.processing.resample_cache import get_resample_plan_cache

    plan_cache = get_resample_plan_cache()
    plan = plan_cache.get_plan(grid_spec, vp_val, target_dt=grid_spec.dt)

    # Resample each property using the computed vp and a common dt
    props_time: dict = {}
    nt_samples = None
    for k, v in props_depth.items():
        was_quantity = hasattr(v, "array")
        data_arr = v.array if was_quantity else np.asarray(v)  # type: ignore

        data_time, dt = resampler.depth_to_time_cube(
            data_arr, vp_val, target_dt=grid_spec.dt, plan=plan
        )
        if was_quantity:
            props_time[k] = Quantity(data_time, v.unit)  # type: ignore
        else:
            props_time[k] = data_time

        if nt_samples is None:
            nt_samples = data_time.shape[2]

    nx, ny, nt_samples = props_time["vp"].shape

    # AVO and AI seismograms
    wavelet_avo = wavelets.ricker_wavelet(f_peak=26, dt=grid_spec.dt)
    modeling_utils.modeling_engine.cached_avo(
        props_time,
        [0, 5, 10, 15],
        wavelet_avo,
        use_quality_weighting=True,
        add_noise=add_avo_noise,
        snr_db=20,
    )

    wavelet_ai = wavelets.ricker_wavelet(f_peak=30, dt=grid_spec.dt)
    modeling_utils.modeling_engine.cached_ai_seismogram(props_time, wavelet_ai)

    # Cache depth-domain AVO/AI
    modeling_utils.modeling_engine.cached_avo_depth(props_depth, [0, 5, 10, 15])
    modeling_utils.modeling_engine.cached_ai_depth(props_depth)

    # EI time-domain seismograms and optimal stacking
    from scipy.ndimage import sobel

    EI_ANGLES = [0, 5, 10, 15, 20, 25]
    ei_angle_seismograms = []
    for angle_idx, angle in enumerate(EI_ANGLES):
        ei_time_angle = modeling_utils.modeling_engine.compute_ei_angle(
            props_time["vp"], props_time["vs"], props_time["rho"], angle
        )
        ei_refl_angle = reflectivity_calc.reflectivity_from_ai(ei_time_angle)
        ei_seis_angle = SeismicOperator.convolve_reflectivity_with_wavelet(
            ei_refl_angle, wavelet_avo, mode="same"
        )
        if add_ei_noise:
            seed = (ei_noise_seed or 42) + angle_idx
            from src.modeling.modeling import modeling_engine

            ei_seis_angle = modeling_engine.add_ei_noise(
                ei_seis_angle,
                frequency_hz=45,
                snr_db=ei_noise_snr,
                include_rock_physics_error=True,
                spatial_correlation_length=3,
                seed=seed,
            )
        ei_angle_seismograms.append(ei_seis_angle)

    boundary_correlations = []
    for ei_seis in ei_angle_seismograms:
        grad_time = sobel(ei_seis, axis=2, mode="constant")
        boundary_quality = np.percentile(np.abs(grad_time), 90)
        boundary_correlations.append(boundary_quality)
    boundary_correlations = np.array(boundary_correlations)
    weights = boundary_correlations / boundary_correlations.sum()
    ei_optimal_stack = np.zeros_like(ei_angle_seismograms[0])
    for seis, weight in zip(ei_angle_seismograms, weights):
        ei_optimal_stack += weight * seis

    ei_refl = reflectivity_calc.reflectivity_from_ai(props_time["ei"])

    os.makedirs(cache_dir, exist_ok=True)
    noise_suffix = "_noise" if add_ei_noise else ""
    config_str_ei = (
        f"ei_time_multiangle_{grid_spec.dt}_{grid_spec.dz}_"
        f"{'_'.join(map(str, grid_spec.shape))}{noise_suffix}"
    )
    config_hash_ei = hashlib.md5(config_str_ei.encode()).hexdigest()[:20]
    ei_cache_file = f"{cache_dir}/ei_time_{config_hash_ei}.npz"

    save_dict = {
        **{f"angle_{i}": seis for i, seis in enumerate(ei_angle_seismograms)},
        "optimal_stack": ei_optimal_stack,
        "ei_refl": ei_refl,
        "time_axis": np.arange(nt_samples) * grid_spec.dt,
        "facies": props_time["facies"],
        "config": {
            "source": "multi-angle seismograms (time-domain stacking)",
            "angles": EI_ANGLES,
            "method": "variance-weighted stack in time domain",
            "f_peak": 45,
            "dt": grid_spec.dt,
            "dz": grid_spec.dz,
            "grid_shape": grid_spec.shape,
            "noise_enabled": add_ei_noise,
            "noise_snr_db": ei_noise_snr,
            "noise_seed": ei_noise_seed,
            "num_angles": len(EI_ANGLES),
        },
    }

    from src.io.cache import cache_for_dir

    cache_for_dir(cache_dir).save_npz(ei_cache_file, save_dict)

    return {
        "ei_cache_file": ei_cache_file,
        "save_dict": save_dict,
        "ei_angle_seismograms": ei_angle_seismograms,
        "ei_optimal_stack": ei_optimal_stack,
    }


class ModelingAPI:
    """Object-oriented facade for the modeling API.

    This thin wrapper delegates to `run_full_modeling` so callers can adopt
    an instance-based API while preserving existing function behavior.
    """

    def run_full_modeling(self, *args, **kwargs):
        return _impl_run_full_modeling(*args, **kwargs)


from src.utils.facades import LazyObjectProxy


# Module-level singleton facade using shared LazyObjectProxy
modeling_api = LazyObjectProxy(lambda: ModelingAPI())

__all__ = ["run_full_modeling", "ModelingAPI", "modeling_api"]

# module logger available for callers that want to tune verbosity
logger = logging.getLogger(__name__)


def get_modeling_api(inst: ModelingAPI | None = None) -> "ModelingAPI":
    """Return the provided ModelingAPI instance or the module-level lazy singleton.

    Useful for dependency injection in tests.
    """
    return inst if inst is not None else modeling_api


__all__.append("get_modeling_api")
