"""VelocityModel abstraction.

Provides a small wrapper around a P-wave velocity cube and GridSpec with
convenience methods for unit conversion, validation, smoothing, and TWT
computation delegating to DepthTimeResampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from src.io.grid import GridSpec
from src.utils.quantity import Quantity


__all__ = ["VelocityModel"]

import logging

logger = logging.getLogger(__name__)


@dataclass
class VelocityModel:
    """Wrapper for a P-wave velocity cube.

    Attributes:
        vp: numpy array (ni, nj, nz) containing P-wave velocities (m/s)
        grid_spec: GridSpec defining shape/dz/dt
    """

    # `vp` may be a raw numpy array or a `Quantity`.
    vp: Union[Quantity, np.ndarray]
    grid_spec: GridSpec

    def __post_init__(self):
        # Wrap raw arrays into Quantity if needed
        if not isinstance(self.vp, Quantity):
            # assume unitless numeric array (likely m/s or km/s) — store without unit metadata
            self.vp = Quantity(self.vp, "m/s")

        if self.vp.array.ndim != 3:
            raise ValueError("vp must be a 3D array (ni, nj, nz)")
        ni, nj, nz = self.vp.array.shape
        if (ni, nj, nz) != self.grid_spec.shape:
            raise ValueError("vp shape must match grid_spec.shape")

    def to_m_per_s(self) -> None:
        """Ensure velocities are in meters per second (in-place).

        If the velocities are in km/s (values < 100), assume they are km/s
        and convert them to m/s. This is a pragmatic heuristic used widely in
        the repository.
        """
        # Convert Quantity in-place to m/s
        if isinstance(self.vp, Quantity):
            q = self.vp.to("m/s", copy=True)
            self.vp = q
        else:
            q = Quantity(self.vp, "m/s").to("m/s", copy=True)
            self.vp = q

    def ensure_m_per_s(self) -> bool:
        """Ensure velocities are in m/s. Return True if a conversion was applied."""
        # Use Quantity -> attempt conversion and inspect whether units changed
        if not isinstance(self.vp, Quantity):
            self.vp = Quantity(self.vp, "m/s")
        before_unit = self.vp.unit
        self.vp = self.vp.to("m/s", copy=True)
        return before_unit != self.vp.unit

    def validate(self) -> None:
        """Validate vp array for physical plausibility.

        Raises ValueError if values are non-finite or non-positive.
        """
        arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
        if not np.all(np.isfinite(arr)):
            raise ValueError("vp contains non-finite values")
        if np.any(arr <= 0.0):
            raise ValueError("vp contains non-positive values")

    def smooth(self, sigma: float = 1.0, truncate: Optional[float] = None) -> None:
        """Apply Gaussian smoothing in-place to the velocity model.

        Args:
            sigma: standard deviation for Gaussian kernel (applied to each axis)
        """
        # Apply smoothing to the numeric array and preserve unit metadata
        arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
        smoothed = gaussian_filter(arr, sigma=sigma)
        if isinstance(self.vp, Quantity):
            self.vp = Quantity(smoothed, self.vp.unit)
        else:
            self.vp = smoothed

    def compute_one_way_time(self, i: int, j: int) -> np.ndarray:
        """Compute one-way cumulative time for trace at (i, j) using GridSpec.

        Returns a 1D array of one-way times (seconds) at each depth sample.
        """
        vp_trace = self.vp.array[i, j, :]
        from src.processing.resampler import resampler_factory

        resampler = resampler_factory.get_resampler(self.grid_spec)
        return resampler.compute_one_way_time(vp_trace)

    def compute_twt_trace(self, i: int, j: int) -> np.ndarray:
        """Return two-way travel time (TWT) trace for column (i, j)."""
        one_way = self.compute_one_way_time(i, j)
        return 2.0 * one_way

    def compute_twt_cube(self) -> np.ndarray:
        """Compute the irregular TWT cube for all traces (ni, nj, nz).

        Returns a (ni, nj, nz) array containing the one-way cumulative time
        at each depth sample multiplied by two (TWT).
        """
        ni, nj, nz = self.vp.array.shape
        out = np.zeros_like(self.vp.array, dtype=float)
        for i in range(ni):
            for j in range(nj):
                out[i, j, :] = self.compute_twt_trace(i, j)
        return out

    @classmethod
    def from_dataset(cls, dataset_manager, vp_key: str = "vp") -> "VelocityModel":
        """Construct a VelocityModel from a DatasetManager instance.

        This reads the velocity cube using the provided key and the
        DatasetManager.grid_spec. The returned model is converted to m/s and
        validated.
        """
        vp = dataset_manager.data[vp_key]
        grid_spec = dataset_manager.grid_spec
        vm = cls(vp=vp.copy(), grid_spec=grid_spec)
        vm.to_m_per_s()
        vm.validate()
        return vm

    def resample_to_time(
        self,
        data_depth: np.ndarray,
        is_categorical: bool = False,
        target_dt: Optional[float] = None,
        target_nt: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Resample a depth-sampled property cube to regular time sampling.

        Returns (data_time, dt).
        """
        from src.processing.resampler import resampler_factory

        resampler = resampler_factory.get_resampler(self.grid_spec)
        dt = target_dt if target_dt is not None else self.grid_spec.dt
        vp_arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
        # build a ResamplePlan once for this vp and reuse it via shared cache
        from src.processing.resample_cache import get_resample_plan_cache

        plan = get_resample_plan_cache().get_plan(
            self.grid_spec, vp_arr, target_dt=dt, target_nt=target_nt
        )
        data_time, dt_out = resampler.depth_to_time_cube(
            data_depth, vp_arr, target_dt=dt, target_nt=target_nt, plan=plan
        )
        return data_time, dt_out
