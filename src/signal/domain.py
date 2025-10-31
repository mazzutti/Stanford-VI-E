"""Domain conversion helpers.

Helpers to convert depth-domain property cubes into irregular two-way time
(TWT) cubes and to resample properties onto a regular time grid.
"""

import os
import numpy as np

# tqdm is only used in some code paths; import lazily if needed to avoid
# top-level unused-import warnings and heavy deps during import.
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Try to import numba for faster resampling
from src.utils.compat import _NUMBA_AVAILABLE, njit, prange

import logging

from src.io.grid import GridSpec

from src.utils.quantity import Quantity
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)

__all__ = [
    "convert_depth_to_twt",
    "resample_properties_to_time",
    "DepthTimeConverter",
]


class DepthTimeConverter:
    """Encapsulates depth-to-time conversion and property resampling."""

    def __init__(self, grid_spec: GridSpec):
        self.grid_spec = grid_spec

    def convert_depth_to_twt(self, vp_depth):
        logger.info("Converting from depth to two-way time using DepthTimeResampler...")

        # Use the canonical DepthTimeResampler which already handles Quantity.
        # resampler not required here; ResamplePlan is computed below via cache

        input_was_quantity = isinstance(vp_depth, Quantity)
        vp_arr = vp_depth.array if input_was_quantity else np.asarray(vp_depth)

        from src.processing.resample_cache import get_resample_plan_cache

        # Use ResamplePlan to compute one-way/twt and time axis centrally
        plan = get_resample_plan_cache().get_plan(self.grid_spec, vp_arr)
        twt_irregular = plan.twt_arr

        if input_was_quantity:
            return Quantity(twt_irregular, "s")
        return twt_irregular

    def resample_properties_to_time(self, properties_depth, twt_irregular):
        # Unwrap Quantity inputs if present for shape calculations
        vp_prop = properties_depth["vp"]
        vp_arr = vp_prop.array if isinstance(vp_prop, Quantity) else vp_prop
        ni, nj, _ = vp_arr.shape

        twt_arr = (
            twt_irregular.array
            if isinstance(twt_irregular, Quantity)
            else twt_irregular
        )
        max_twt = np.max(twt_arr)
        dt = self.grid_spec.dt
        time_axis = np.arange(0, max_twt, dt)

        # Create a ResamplePlan to reuse for each property resampling. Use the
        # vp array we extracted earlier and match the time axis sampling.
        from src.processing.resample_cache import get_resample_plan_cache

        plan = get_resample_plan_cache().get_plan(
            self.grid_spec, vp_arr, target_dt=dt, target_nt=len(time_axis)
        )

        resampled_properties = {}
        for key, cube in properties_depth.items():
            cube_arr = cube.array if isinstance(cube, Quantity) else cube
            resampled_properties[key] = np.zeros(
                (ni, nj, len(time_axis)), dtype=cube_arr.dtype
            )

        # If numba is available and enabled, use a compiled nearest-neighbor resampler
        use_numba = (
            os.environ.get("RESAMPLE_USE_NUMBA", "1") == "1" and _NUMBA_AVAILABLE
        )
        if use_numba:
            # Use a parallel numba kernel that walks each trace with a two-pointer
            # approach (O(nz+nt) per trace) which is very efficient and avoids
            # Python overhead for large grids.
            @njit(parallel=True)
            def _resample_numba(twt_ir, props, t_axis, out):
                ni, nj, nz = props.shape
                nt = t_axis.shape[0]
                for ii in prange(ni):
                    for jj in range(nj):
                        twt = twt_ir[ii, jj]
                        prop = props[ii, jj]
                        k = 0
                        for ti in range(nt):
                            t = t_axis[ti]
                            # advance k while next sample is still less than t
                            while k + 1 < nz and twt[k + 1] < t:
                                k += 1
                            # choose nearest between k and k+1
                            if k + 1 < nz:
                                if abs(twt[k] - t) <= abs(twt[k + 1] - t):
                                    out[ii, jj, ti] = prop[k]
                                else:
                                    out[ii, jj, ti] = prop[k + 1]
                            else:
                                out[ii, jj, ti] = prop[k]

            logger.info("Resampling properties onto regular time grid (numba-parallel)")
            for key, cube in properties_depth.items():
                cube_arr = cube.array if isinstance(cube, Quantity) else cube
                out = np.zeros((ni, nj, len(time_axis)), dtype=cube_arr.dtype)
                _resample_numba(twt_arr, cube_arr, time_axis, out)
                # Wrap output in Quantity if input was a Quantity
                if isinstance(cube, Quantity):
                    resampled_properties[key] = Quantity(out, cube.unit)
                else:
                    resampled_properties[key] = out
            return resampled_properties, time_axis

        # Fallback: CPU implementation using centralized helper that accepts
        # twt_irregular
        logger.info("Resampling properties onto regular time grid (CPU fallback)")
        from src.processing.resampler import resampler_factory

        for key, cube in properties_depth.items():
            logger.info("  Resampling %s...", key)
            cube_arr = cube.array if isinstance(cube, Quantity) else cube
            # Use DepthTimeResampler's helper method on numeric arrays
            resampler = resampler_factory.get_resampler(grid_spec=self.grid_spec)
            # Reuse the plan so the resampler can take fast paths when possible.
            out = resampler.depth_to_time_from_twt(
                cube_arr,
                twt_arr,
                time_axis,
                is_categorical=True,
                progress_every=max(1, ni // 10),
                prefix="    ",
                plan=plan,
            )
            # Wrap output in Quantity if input was a Quantity
            if isinstance(cube, Quantity):
                resampled_properties[key] = Quantity(out, cube.unit)
            else:
                resampled_properties[key] = out

        return resampled_properties, time_axis


def convert_depth_to_twt(vp_depth, grid_spec: GridSpec):
    return _impl_convert_depth_to_twt(vp_depth, grid_spec)


def resample_properties_to_time(properties_depth, twt_irregular, grid_spec: GridSpec):
    return _impl_resample_properties_to_time(properties_depth, twt_irregular, grid_spec)


def _impl_convert_depth_to_twt(vp_depth, grid_spec: GridSpec):
    """Canonical implementation for convert_depth_to_twt.

    This function centralizes the conversion entrypoint for easier testing
    and to provide a canonical callable. It preserves the original
    behavior.
    """
    # Prefer the get_* helper which returns either a provided instance or
    # the module-level lazy proxy when grid_spec is None.
    converter = get_depth_time_converter(grid_spec=grid_spec)
    return converter.convert_depth_to_twt(vp_depth)


def _impl_resample_properties_to_time(
    properties_depth, twt_irregular, grid_spec: GridSpec
):
    """Canonical implementation for resample_properties_to_time.

    Delegates to DepthTimeConverter while preserving Quantity handling.
    """
    converter = get_depth_time_converter(grid_spec=grid_spec)
    return converter.resample_properties_to_time(properties_depth, twt_irregular)


# Module-level lazy converter
depth_time_converter = LazyObjectProxy(lambda gs: DepthTimeConverter(gs))


def get_depth_time_converter(
    grid_spec: GridSpec | None = None, instance: DepthTimeConverter | None = None
) -> DepthTimeConverter:
    """Return provided DepthTimeConverter instance or a module-level lazy one.

    If `instance` is provided it is returned directly. Otherwise a new
    DepthTimeConverter is created for the provided `grid_spec` or the
    module-level lazy proxy is returned when `grid_spec` is None.
    """
    return _impl_get_depth_time_converter(grid_spec=grid_spec, instance=instance)


def _impl_get_depth_time_converter(
    grid_spec: GridSpec | None = None, instance: DepthTimeConverter | None = None
) -> DepthTimeConverter:
    if instance is not None:
        return instance
    if grid_spec is not None:
        return DepthTimeConverter(grid_spec)
    return depth_time_converter
