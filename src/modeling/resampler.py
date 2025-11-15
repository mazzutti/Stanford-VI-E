"""Resampling service for depth-to-time domain conversion.

Encapsulates the resampling logic, making it independently testable
and keeping ModelingPipeline simpler.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.io.grid import GridSpec
from src.modeling.modeling import unwrap_quantity
from src.analysis.factories import ConversionStrategyFactory
from src.utils.quantity import Quantity

logger = logging.getLogger(__name__)

__all__ = ["ResamplingService"]


_CONVERSION_FACTORY = ConversionStrategyFactory()


class ResamplingService:
    """Handles depth-to-time resampling of rock properties."""

    @staticmethod
    def resample_to_time(
        props_depth: dict[str, NDArray[np.floating[Any]] | Quantity],
        grid_spec: GridSpec,
    ) -> dict[str, NDArray[np.floating[Any]] | Quantity]:
        """Resample rock properties from depth to time domain.

        Args:
            props_depth: Depth-domain properties dictionary with 'vp', 'vs', 'rho', etc.
            grid_spec: Grid specification with target dt

        Returns:
            Time-domain properties dictionary with same keys
        """
        from src.processing.resampling._resampler import resampler_factory
        from src.processing.resampling._cache import get_resample_plan_cache

        resampler = resampler_factory.get_resampler(grid_spec)

        processed_props: dict[str, NDArray[np.floating[Any]] | Quantity] = dict(
            props_depth
        )

        velocity_strategy = _CONVERSION_FACTORY.get_velocity_strategy()

        vp_source = processed_props.get("vp")
        if isinstance(vp_source, Quantity):
            # Convert to SI units for numerical stability
            vp_converted = cast(
                Quantity, velocity_strategy.convert(vp_source, vp_source.unit, "m/s")
            )
            processed_props["vp"] = vp_converted

        vs_source = processed_props.get("vs")
        if isinstance(vs_source, Quantity):
            processed_props["vs"] = velocity_strategy.convert(
                vs_source, vs_source.unit, "m/s"
            )

        vp_val = unwrap_quantity(processed_props["vp"])

        plan_cache = get_resample_plan_cache()
        plan = plan_cache.get_plan(grid_spec, vp_val, target_dt=grid_spec.dt)

        props_time: dict[str, NDArray[np.floating[Any]] | Quantity] = {}
        for key, original_value in props_depth.items():
            processed_value = processed_props[key]

            if isinstance(processed_value, Quantity):
                data_arr = processed_value.array
                data_time, _ = resampler.depth_to_time_cube(
                    data_arr, vp_val, target_dt=grid_spec.dt, plan=plan
                )
                result_unit = processed_value.unit
                quantity_result = Quantity(data_time, result_unit)

                # Convert back to original unit if we normalized earlier
                if (
                    isinstance(original_value, Quantity)
                    and original_value.unit != result_unit
                ):
                    if key in {"vp", "vs"}:
                        quantity_result = cast(
                            Quantity,
                            velocity_strategy.convert(
                                quantity_result, result_unit, original_value.unit
                            ),
                        )

                if isinstance(original_value, Quantity):
                    props_time[key] = (
                        quantity_result
                        if original_value.unit == quantity_result.unit
                        else Quantity(quantity_result.array, original_value.unit)
                    )
                else:
                    props_time[key] = quantity_result.array
            else:
                v_arr = processed_value
                data_time, _ = resampler.depth_to_time_cube(
                    v_arr, vp_val, target_dt=grid_spec.dt, plan=plan
                )
                props_time[key] = data_time

        return props_time
