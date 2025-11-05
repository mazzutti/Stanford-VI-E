"""Programmatic API for the modeling pipeline.

Orchestrates the complete AVO modeling workflow: data loading, resampling,
and AVO synthesis with sensible defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import logging

from src.io.grid import GridSpec
from src.modeling.modeling import (
    AVOSynthesizer,
    AngleModel,
    SynthesisConfig,
    _unwrap_quantity,
)
from src.modeling.model_cache import CacheManager
from src.signal import wavelets
from src.utils.quantity import Quantity

if TYPE_CHECKING:
    from src.processing.rock_physics import RockPhysicsModel

__all__ = ["run_full_modeling"]

logger = logging.getLogger(__name__)


def run_full_modeling(
    cache_dir: str = ".cache",
    add_avo_noise: bool = False,
) -> dict[str, bool | list[np.ndarray] | None | np.ndarray]:
    """Run the full modeling pipeline from depth to time domain.

    Orchestrates: data loading, depth-to-time resampling, and AVO synthesis.

    Args:
        cache_dir: Cache directory for synthetics
        add_avo_noise: Add realistic angle-dependent noise

    Returns:
        Dictionary with modeling results
    """
    # Configuration
    data_path = "."
    file_map: dict[str, str] = {
        "vp": "P-wave Velocity",
        "vs": "S-wave Velocity",
        "rho": "Density",
        "facies": "Facies",
    }
    grid_spec = GridSpec((150, 200, 200), dz=1.0, dt=0.001)

    # Load and prepare properties
    dm = _load_dataset(data_path, file_map, grid_spec)
    props_depth = dm.to_props_dict()

    # Resample to time domain
    props_time = _resample_to_time(props_depth, grid_spec)

    # Generate synthetics
    angle_model = AngleModel()
    synthesizer = AVOSynthesizer(angle_model)
    cache_manager = CacheManager(cache_dir)

    config = SynthesisConfig(
        use_quality_weighting=True,
        add_noise=add_avo_noise,
        snr_db=20,
    )

    wavelet_avo = wavelets.ricker_wavelet(f_peak=26, dt=grid_spec.dt)
    angles: list[float] = [0.0, 5.0, 10.0, 15.0]

    def create_synthetics_wrapper(
        props_unwrapped: dict[str, np.ndarray],
        angles_in: list[float],
        wavelet_in: np.ndarray,
        config_in: SynthesisConfig | None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Wrapper that satisfies callable signature for cache manager."""
        return synthesizer.create_synthetics(
            cast(dict[str, np.ndarray | Quantity], props_unwrapped),
            angles_in,
            wavelet_in,
            config_in,
        )

    angle_stacks, full_stack = cache_manager.get_avo_synthetics(
        props_time,
        angles,
        wavelet_avo,
        create_fn=create_synthetics_wrapper,
        config=config,
    )

    return {
        "avo_cached": True,
        "angle_stacks": angle_stacks,
        "full_stack": full_stack,
    }


def _load_dataset(
    data_path: str, file_map: dict[str, str], grid_spec: GridSpec
) -> "RockPhysicsModel":
    """Load dataset and prepare rock physics model."""
    from src.io.data_loader import DatasetManager
    from src.processing.rock_physics import RockPhysicsModel

    dm = DatasetManager.from_stanfordsix(data_path, file_map, grid_spec)
    props_depth: dict[str, np.ndarray | None] = {
        "vp": dm.vp,
        "vs": dm.vs,
        "rho": dm.rho,
        "facies": dm.facies,
        "full_stack": dm.full_stack,
    }

    rpm = RockPhysicsModel.from_props(props_depth, grid_spec)
    rpm.ensure_units()
    return rpm


def _resample_to_time(
    props_depth: dict[str, np.ndarray | Quantity], grid_spec: GridSpec
) -> dict[str, np.ndarray | Quantity]:
    """Resample rock properties from depth to time domain."""
    from src.processing.resampler import resampler_factory
    from src.processing.resample_cache import get_resample_plan_cache

    resampler = resampler_factory.get_resampler(grid_spec)
    vp_val = _unwrap_quantity(props_depth["vp"])

    plan_cache = get_resample_plan_cache()
    plan = plan_cache.get_plan(grid_spec, vp_val, target_dt=grid_spec.dt)

    props_time: dict[str, np.ndarray | Quantity] = {}
    for k, v in props_depth.items():
        if isinstance(v, Quantity):
            v_qty = v
            data_arr = v_qty.array
            data_time, dt = resampler.depth_to_time_cube(
                data_arr, vp_val, target_dt=grid_spec.dt, plan=plan
            )
            props_time[k] = Quantity(data_time, v_qty.unit)
        else:
            v_arr = v
            data_time, dt = resampler.depth_to_time_cube(
                v_arr, vp_val, target_dt=grid_spec.dt, plan=plan
            )
            props_time[k] = data_time

    return props_time
