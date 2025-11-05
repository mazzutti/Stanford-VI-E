"""Resampling service for depth-to-time domain conversion.

Encapsulates the resampling logic, making it independently testable
and keeping ModelingPipeline simpler.
"""

from __future__ import annotations

import logging

import numpy as np

from src.io.grid import GridSpec
from src.modeling.modeling import _unwrap_quantity
from src.utils.quantity import Quantity

logger = logging.getLogger(__name__)

__all__ = ["ResamplingService"]


class ResamplingService:
    """Handles depth-to-time resampling of rock properties."""

    @staticmethod
    def resample_to_time(
        props_depth: dict[str, np.ndarray | Quantity],
        grid_spec: GridSpec,
    ) -> dict[str, np.ndarray | Quantity]:
        """Resample rock properties from depth to time domain.

        Args:
            props_depth: Depth-domain properties dictionary with 'vp', 'vs', 'rho', etc.
            grid_spec: Grid specification with target dt

        Returns:
            Time-domain properties dictionary with same keys
        """
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
