"""Programmatic API for running the full modeling pipeline.

This module provides a single entrypoint `run_full_modeling` which encapsulates
the orchestration previously duplicated across callers. It keeps the heavy
compute logic in `src.modeling` and the data loading in `src.io` while
exposing a small, testable function for external callers (for example
`src.analysis.seismograms`).
"""

from __future__ import annotations

import numpy as np

from src.io import data_loader
from src.io.grid import GridSpec
from src.modeling import modeling as modeling_utils
from src.signal import wavelets
from src.utils.quantity import Quantity
import logging

__all__ = ["run_full_modeling"]

# module logger available for callers that want to tune verbosity
logger = logging.getLogger(__name__)


def run_full_modeling(
    cache_dir: str = ".cache",
    skip_cleanup: bool = False,
    verbose: bool = False,
    add_avo_noise: bool = False,
):
    """Run the full modeling pipeline from depth to time domain.

    Orchestrates the complete AVO modeling workflow including:
    - Data loading and unit normalization
    - Depth to time resampling
    - AVO synthesis
    - Caching and cleanup
    """
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
    from src.analysis.types.base import DatasetManagerFactory

    dm = DatasetManagerFactory().create(DATA_PATH, FILE_MAP, grid_spec)
    props_depth = {
        "vp": dm.vp,
        "vs": dm.vs,
        "rho": dm.rho,
        "facies": dm.facies,
        "full_stack": dm.full_stack,
    }

    # Normalize units and validate properties via RockPhysicsModel
    from src.processing.rock_physics import RockPhysicsModel

    # Build RockPhysicsModel and normalize units (this stores Quantity internally)
    rpm = RockPhysicsModel.from_props(props_depth, grid_spec)
    rpm.ensure_units()
    # When downstream code expects raw arrays, use rpm.to_props_dict(); but
    # prefer keeping rpm available so consumers can access Quantity metadata.
    props_depth = rpm.to_props_dict()

    # This pipeline focuses on AVO computations and resampling.

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

    # AVO seismograms
    wavelet_avo = wavelets.ricker_wavelet(f_peak=26, dt=grid_spec.dt)
    modeling_utils.modeling_engine.cached_avo(
        props_time,
        [0, 5, 10, 15],
        wavelet_avo,
        use_quality_weighting=True,
        add_noise=add_avo_noise,
        snr_db=20,
    )
    # Cache depth-domain AVO
    modeling_utils.modeling_engine.cached_avo_depth(props_depth, [0, 5, 10, 15])

    # The function returns AVO-related results only.
    return {
        "avo_cached": True,
    }
