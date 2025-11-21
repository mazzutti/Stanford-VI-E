"""ResamplePlan

Helpers to precompute time axes, two-way time (TWT) arrays, padded axes and
block layout for resampling operations. This centralizes the decision logic
used by resamplers (uniform TWT detection, blocks, padded arrays) to avoid
duplication across callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.io.grid import GridSpec
from src.utils.quantity import Quantity
from src.utils.units import UnitRegistry

# Alias for clarity
IndexTuple = tuple[int, int, int]

# ResamplePlan intentionally stores several computed arrays and helper
# fields as attributes for efficient reuse. These attributes increase
# the dataclass attribute count but are correct for the data model.

@dataclass
class ResamplePlan:
    """Plan describing time axes, TWT arrays and blocking for resampling.

    This dataclass precomputes time axes, two-way travel times (TWT), and
    provides helpers to produce padded arrays used by the resampler. Keeping
    these computations centralized avoids duplication and keeps resampling
    callers simple.
    """

    grid_spec: GridSpec
    vp_arr: NDArray[Any]
    dt: float
    nt: int
    ni: int
    nj: int
    nz: int
    one_way: NDArray[Any]
    twt_arr: NDArray[Any]
    uniform_twt: bool
    block_size: int = 65536

    @classmethod
    def create(
        cls,
        grid_spec: GridSpec,
        vp_depth: NDArray[Any] | Quantity,
        target_dt: float | None = None,
        target_nt: int | None = None,
        block_size: int = 65536,
    ) -> ResamplePlan:
        """Construct a ResamplePlan from a vp_depth cube.

        vp_depth may be a Quantity or ndarray with shape (ni, nj, nz).
        Returns a ResamplePlan with computed time axis and twt arrays.
        """
        # This construction method computes several temporaries for plan
        # generation; keep the linter quiet about the local variable count.

        # Unwrap quantity (Quantity -> ndarray)
        if isinstance(vp_depth, Quantity):
            vp_val = vp_depth.array
        else:
            vp_val = np.asarray(vp_depth)

        if vp_val.ndim != 3:
            raise ValueError("vp_depth must be a 3D array (ni, nj, nz)")

        ni, nj, nz = vp_val.shape

        # Ensure velocities in m/s
        vp_conv, _ = UnitRegistry.ensure_m_per_s(vp_val, copy_on_convert=True)
        vp_float = np.asarray(vp_conv, dtype=float)

        if np.any(vp_float <= 0) or np.isnan(vp_float).any():
            raise ValueError("vp_depth contains non-positive or NaN values")

        dz_val, _ = UnitRegistry.ensure_meters(grid_spec.dz)

        slowness = 1.0 / vp_float
        one_way = np.cumsum(slowness * dz_val, axis=2)

        dt = target_dt if target_dt is not None else grid_spec.dt
        max_twt = float(np.max(2.0 * one_way[:, :, -1]))
        nt = int(np.ceil(max_twt / dt)) + 1
        if target_nt is not None:
            nt = int(target_nt)

        twt_arr = 2.0 * one_way

        uniform_twt = np.allclose(
            twt_arr, np.broadcast_to(twt_arr[0, 0, :], twt_arr.shape)
        )

        return cls(
            grid_spec=grid_spec,
            vp_arr=vp_float,
            dt=dt,
            nt=nt,
            ni=ni,
            nj=nj,
            nz=nz,
            one_way=one_way,
            twt_arr=twt_arr,
            uniform_twt=uniform_twt,
            block_size=block_size,
        )

    @property
    def time_axis(self) -> NDArray[Any]:
        """Return the regularly sampled time axis for this plan."""
        return np.arange(self.nt) * self.dt

    @property
    def ntr(self) -> int:
        """Return number of traces (ni * nj) for this plan."""
        return self.ni * self.nj

    def twt_padded(self) -> NDArray[Any]:
        """Return twt padded axis: either 1D (nz+1,) if uniform or 2D
        (nz+1, ntr) if non-uniform.
        """
        if self.uniform_twt:
            arr: NDArray[Any] = np.concatenate([[0.0], self.twt_arr[0, 0, :]])
            return arr
        # build per-column padded twt
        twt_padded = np.concatenate(
            [np.zeros((self.ni, self.nj, 1)), self.twt_arr], axis=2
        )
        # reshape to (nz+1, ntr)
        result: NDArray[Any] = twt_padded.transpose(2, 0, 1).reshape(self.nz + 1, -1)
        return result

    def prepare_depth_padded_flat(self, data_arr: NDArray[Any]) -> NDArray[Any]:
        """Given a depth-sampled `data_arr` shaped (ni, nj, nz), produce
        a flattened padded depth array shaped (nz+1, ntr) suitable for
        passing to BatchedInterpolator.
        """
        if data_arr.shape != (self.ni, self.nj, self.nz):
            raise ValueError("data_arr shape must match vp dimensions")
        depth_padded = np.concatenate([data_arr[:, :, 0:1], data_arr], axis=2)
        result: NDArray[Any] = depth_padded.transpose(2, 0, 1).reshape(self.nz + 1, -1)
        return result

    def blocks(self) -> list[tuple[int, int]]:
        """Return list of (start, end) index pairs for block iteration.

        The blocks partition the flattened trace axis into chunks of size
        `self.block_size` for batched processing.
        """
        ntr = self.ntr
        b = self.block_size
        return [(start, min(start + b, ntr)) for start in range(0, ntr, b)]

__all__ = ["ResamplePlan"]

# Module logger
logger = logging.getLogger(__name__)

# ResamplePlan contains compact helpers for preparing padded arrays and
# block iteration. The methods are intentionally simple and directly
# mirror the mathematical operations used by resamplers.
