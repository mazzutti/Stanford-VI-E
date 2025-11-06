"""Domain conversion helpers.

Converts depth-domain seismic properties to two-way time (TWT) domain and
resamples properties onto regular time grids.
"""

import logging
import numpy as np
from numba import njit, prange

from src.io.grid import GridSpec
from src.utils.quantity import Quantity

logger = logging.getLogger(__name__)

__all__ = [
    "DepthTimeConverter",
]


class DepthTimeConverter:
    """Converts depth-domain properties to two-way time (TWT) domain.

    Provides methods to convert depth-domain velocity fields to TWT and
    to resample depth-domain properties onto regular time grids.

    Attributes:
        grid_spec: GridSpec defining the grid geometry and sampling
    """

    def __init__(self, grid_spec: GridSpec):
        """Initialize converter with grid specification.

        Args:
            grid_spec: GridSpec object defining grid geometry
        """
        self.grid_spec = grid_spec

    def convert_depth_to_twt(
        self, vp_depth: np.ndarray | Quantity
    ) -> np.ndarray | Quantity:
        """Convert depth-domain velocity to two-way time (TWT).

        Computes TWT grid from a depth-domain P-wave velocity field.

        Args:
            vp_depth: P-wave velocity in depth domain (m/s)

        Returns:
            TWT array with same shape as input, as Quantity if input was Quantity
        """
        logger.info("Converting depth to two-way time (TWT)...")

        input_was_quantity = isinstance(vp_depth, Quantity)
        vp_arr = vp_depth.array if input_was_quantity else np.asarray(vp_depth)

        from src.processing.resampling._cache import get_resample_plan_cache

        # Compute TWT using cached resample plan
        plan = get_resample_plan_cache().get_plan(self.grid_spec, vp_arr)
        twt_irregular = plan.twt_arr

        if input_was_quantity:
            return Quantity(twt_irregular, "s")
        return twt_irregular

    def resample_properties_to_time(
        self, properties_depth: dict, twt_irregular: np.ndarray | Quantity
    ) -> tuple[dict, np.ndarray]:
        """Resample depth-domain properties to regular time grid.

        Resamples a collection of depth-domain property cubes onto a regular
        time grid using the provided TWT field.

        Args:
            properties_depth: Dictionary of property cubes in depth domain
            twt_irregular: TWT values in depth domain (from convert_depth_to_twt)

        Returns:
            Tuple of (resampled_properties_dict, time_axis)
        """
        # Extract and unwrap Quantity objects
        vp_prop = properties_depth["vp"]
        vp_arr = vp_prop.array if isinstance(vp_prop, Quantity) else vp_prop
        ni, nj, _ = vp_arr.shape

        twt_arr = (
            twt_irregular.array
            if isinstance(twt_irregular, Quantity)
            else twt_irregular
        )

        # Compute regular time axis
        max_twt = np.max(twt_arr)
        dt = self.grid_spec.dt
        time_axis = np.arange(0, max_twt, dt)

        # Initialize output dictionary
        resampled_properties = {}
        for key, cube in properties_depth.items():
            cube_arr = cube.array if isinstance(cube, Quantity) else cube
            resampled_properties[key] = np.zeros(
                (ni, nj, len(time_axis)), dtype=cube_arr.dtype
            )

        # Use Numba-optimized resampling (Numba is a required dependency)
        logger.info("Resampling properties onto regular time grid...")
        self._resample_numba_parallel(
            properties_depth, twt_arr, time_axis, resampled_properties
        )

        return resampled_properties, time_axis

    @staticmethod
    def _resample_numba_parallel(
        properties_depth: dict, twt_arr: np.ndarray, time_axis: np.ndarray, output: dict
    ) -> None:
        """Resample using Numba-optimized parallel kernel.

        Uses a two-pointer approach for efficient nearest-neighbor resampling.

        Args:
            properties_depth: Input properties (depth domain)
            twt_arr: TWT values
            time_axis: Regular time grid
            output: Output dictionary to fill
        """

        @njit(parallel=True)
        def _kernel(twt_ir, props, t_axis, out):
            """Numba kernel for parallel resampling."""
            ni, nj, nz = props.shape
            nt = t_axis.shape[0]
            for ii in prange(ni):
                for jj in range(nj):
                    twt = twt_ir[ii, jj]
                    prop = props[ii, jj]
                    k = 0
                    for ti in range(nt):
                        t = t_axis[ti]
                        # Advance k to next sample less than t
                        while k + 1 < nz and twt[k + 1] < t:
                            k += 1
                        # Choose nearest sample
                        if k + 1 < nz:
                            if abs(twt[k] - t) <= abs(twt[k + 1] - t):
                                out[ii, jj, ti] = prop[k]
                            else:
                                out[ii, jj, ti] = prop[k + 1]
                        else:
                            out[ii, jj, ti] = prop[k]

        # Apply kernel to each property
        for key, cube in properties_depth.items():
            cube_arr = cube.array if isinstance(cube, Quantity) else cube
            out = output[key]
            _kernel(twt_arr, cube_arr, time_axis, out)
            # Wrap in Quantity if input was Quantity
            if isinstance(cube, Quantity):
                output[key] = Quantity(out, cube.unit)
