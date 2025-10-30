"""SeismoCube helper.

Provides a small wrapper around a seismic cube (time or depth) and a
`GridSpec` describing its sampling. Intended to centralize simple helpers
used by plotting and analysis code: domain tagging, slicing, normalization,
and thin wrappers to DepthTimeResampler for domain conversions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

from src.io.grid import GridSpec

__all__ = ["SeismoCube"]


@dataclass
class SeismoCube:
    """Wrapper around a 3D seismic/instrument cube.

    Attributes:
        data: numpy array (ni, nj, nk) where the third axis is either depth
            samples (nz) or time samples (nt) depending on `domain`.
        grid_spec: GridSpec describing the dataset sampling (dz/dt used
            appropriately).
        domain: either 'depth' or 'time'.
    """

    data: np.ndarray
    grid_spec: GridSpec
    domain: str = "depth"

    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError("data must be a 3D numpy array")
        if self.domain not in ("depth", "time"):
            raise ValueError("domain must be 'depth' or 'time'")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def to_time(self, vp_depth: ArrayLike, target_dt: Optional[float] = None):
        """Return a new SeismoCube resampled to regular time sampling.

        Args:
            vp_depth: (ni, nj, nz) P-wave velocity in depth domain (m/s)
        """
        if self.domain != "depth":
            raise ValueError("to_time can only be called on depth-domain cubes")

        from src.processing.resampler import resampler_factory
        from src.processing.resample_cache import get_resample_plan_cache

        resampler = resampler_factory.get_resampler(self.grid_spec)
        vp_arr = np.asarray(vp_depth)
        plan = get_resample_plan_cache().get_plan(
            self.grid_spec, vp_arr, target_dt=target_dt
        )
        data_time, dt = resampler.depth_to_time_cube(
            self.data, vp_arr, target_dt=target_dt, plan=plan
        )
        return SeismoCube(data=data_time, grid_spec=self.grid_spec, domain="time"), dt

    def to_depth(self, vp_depth: ArrayLike):
        """Return a new SeismoCube resampled to depth sampling using vp_depth.

        Args:
            vp_depth: (ni, nj, nz) P-wave velocity in depth domain (m/s)
        """
        if self.domain != "time":
            raise ValueError("to_depth can only be called on time-domain cubes")

        from src.processing.resampler import resampler_factory
        from src.processing.resample_cache import get_resample_plan_cache

        resampler = resampler_factory.get_resampler(self.grid_spec)
        vp_arr = np.asarray(vp_depth)

        plan = get_resample_plan_cache().get_plan(self.grid_spec, vp_arr)
        data_depth = resampler.time_to_depth_cube(self.data, vp_arr, plan=plan)
        return SeismoCube(data=data_depth, grid_spec=self.grid_spec, domain="depth")

    def normalize(self, method: str = "std") -> None:
        """In-place normalization helpers (return None to mutate).

        Methods supported:
            - 'std': divide by standard deviation (per-trace)
            - 'max': divide by max absolute value
        """
        if method == "std":
            ni, nj, nk = self.data.shape
            for i in range(ni):
                for j in range(nj):
                    trace = self.data[i, j, :]
                    s = np.std(trace)
                    if s > 0:
                        self.data[i, j, :] = trace / s
        elif method == "max":
            m = np.max(np.abs(self.data))
            if m > 0:
                self.data = self.data / m
        else:
            raise ValueError(f"Unknown normalization method: {method}")
