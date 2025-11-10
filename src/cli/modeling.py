"""Data loading and modeling orchestration for seismic workflows.

This module provides utilities for loading depth properties, performing
depth-to-time resampling, and running the core modeling pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.io.loader import DatasetManager
from src.io.utilities import load_depth_properties
from src.io.grid import GridSpec
from src.utils.quantity import Quantity
from src.utils.units import UnitRegistry

logger = logging.getLogger(__name__)

__all__ = ["load_data", "run_modeling", "save_results"]


def load_data() -> tuple[Any, str, dict[str, str], GridSpec]:
    """Load static dataset used by the modeling pipeline.

    Returns
    -------
    tuple
        (props_depth, DATA_PATH, FILE_MAP, grid_spec)
    """
    DATA_PATH = "."
    FILE_MAP = {
        "vp": "P-wave Velocity",
        "vs": "S-wave Velocity",
        "rho": "Density",
        "facies": "Facies",
    }
    grid_spec = GridSpec((150, 200, 200), dz=1.0, dt=0.001)

    logger.info("%s", "\n" + "=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("%s", "=" * 70)
    t0 = time.time()

    dm = DatasetManager.from_stanfordsix(DATA_PATH, FILE_MAP, grid_spec)
    props_depth = load_depth_properties(dm)
    t1 = time.time()
    logger.info("✓ Loaded data in %.2fs", (t1 - t0))

    # Convert velocity to m/s
    from src.processing.materials.velocity import VelocityModel

    if props_depth["vp"] is not None:
        try:
            vm = VelocityModel(vp=props_depth["vp"], grid_spec=grid_spec)
            converted = vm.ensure_m_per_s()
            props_depth["vp"] = vm.vp.array if hasattr(vm.vp, "array") else vm.vp
        except Exception:
            try:
                out, converted = UnitRegistry.ensure_m_per_s(
                    props_depth["vp"], copy_on_convert=True
                )
                if converted:
                    props_depth["vp"] = out
            except Exception:
                pass

    # Convert vs and rho
    if props_depth["vs"] is not None and props_depth["rho"] is not None:
        try:
            from src.processing.materials import VsModel, DensityModel

            vsm = VsModel(props_depth["vs"])
            vsm.ensure_m_per_s()
            props_depth["vs"] = cast(NDArray[np.floating[Any]], vsm.vs)

            drm = DensityModel(props_depth["rho"])
            drm.ensure_kg_per_m3()
            props_depth["rho"] = cast(NDArray[np.floating[Any]], drm.rho)
        except Exception:
            try:
                out, converted = UnitRegistry.ensure_m_per_s(
                    props_depth["vs"], copy_on_convert=True
                )
                if converted:
                    props_depth["vs"] = out
            except Exception:
                pass

    return props_depth, DATA_PATH, FILE_MAP, grid_spec


def run_modeling(
    props_depth: dict[str, Any],
    args: Any,
    grid_spec: GridSpec,
) -> dict[str, Any]:
    """Run the core modeling steps (depth->time, AVO).

    Parameters
    ----------
    props_depth : dict
        Depth-domain properties
    args : argparse.Namespace
        Parsed command-line arguments
    grid_spec : GridSpec
        Grid specification

    Returns
    -------
    dict
        Dictionary with modeling results
    """
    _t0 = time.time()

    from src.processing.resampling._resampler import resampler_factory
    from src.processing.resampling._cache import get_resample_plan_cache

    resampler = resampler_factory.get_resampler(grid_spec)
    vp_for_twt = (
        props_depth["vp"].array
        if hasattr(props_depth["vp"], "array")
        else props_depth["vp"]
    )

    plan_cache = get_resample_plan_cache()
    plan = plan_cache.get_plan(grid_spec, vp_for_twt, target_dt=grid_spec.dt)

    # Resample each property
    props_time = {}
    for k, v in props_depth.items():
        if v is None:
            props_time[k] = None
            continue
        was_q = hasattr(v, "array")
        data_arr = v.array if was_q else v
        # Ensure data_arr is a numpy array
        if not isinstance(data_arr, np.ndarray):
            data_arr = np.asarray(data_arr)
        data_time, _ = resampler.depth_to_time_cube(data_arr, vp_for_twt, plan=plan)
        props_time[k] = Quantity(data_time, v.unit) if was_q else data_time

    nt = props_time["vp"].shape[2]
    _t1 = time.time()
    logger.info("Depth->Time resampling completed in %.2fs", (_t1 - _t0))
    nx, ny, nt_samples = props_time["vp"].shape

    return {
        "props_depth": props_depth,
        "props_time": props_time,
        "nt": nt,
        "nx": nx,
        "ny": ny,
        "nt_samples": nt_samples,
    }


def save_results() -> None:
    """Display modeling summary and next steps."""
    logger.info("%s", "\n" + "=" * 70)
    logger.info("SUMMARY - ALL MODELING COMPLETE")
    logger.info("%s", "=" * 70)
    logger.info("\n✓ Generated techniques: AVO")
    logger.info("\nNext steps: see README or run plotting modules under src.plotting")
