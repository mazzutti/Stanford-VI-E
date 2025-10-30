"""BatchedInterpolator

Provides a small helper to perform vectorized or block-wise interpolation of
depth->time (and nearest-neighbor categorical) for many traces at once.

This centralizes the flatten/reshape/block logic used in the resampler and
provides a single place to tune block size vs memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


@dataclass
class BatchedInterpolator:
    time_axis: ArrayLike
    kind: str = "linear"
    block_size: int = 65536

    def interpolate(
        self, twt_padded: np.ndarray, depth_padded_flat: np.ndarray
    ) -> np.ndarray:
        """Interpolate depth_padded_flat (shape (nz+1, ntraces)) onto self.time_axis.

        Returns: array shaped (nt, ntraces)
        """
        nzp1, ntr = depth_padded_flat.shape
        nt = len(self.time_axis)

        # If twt_padded is 1D (shared across traces) we can use the fast
        # vectorized path (possibly blocked on traces). If twt_padded is 2D
        # (nz+1, ntr), then fall back to block-wise per-block interp using the
        # appropriate twt columns.
        twt_is_2d = twt_padded.ndim == 2

        if not twt_is_2d:
            # twt_padded is 1D: fast path
            if ntr <= self.block_size:
                interp_func = interp1d(
                    twt_padded,
                    depth_padded_flat,
                    kind=self.kind,
                    axis=0,
                    bounds_error=False,
                    fill_value=0.0,
                )
                return interp_func(self.time_axis)

            out = np.zeros((nt, ntr), dtype=depth_padded_flat.dtype)
            for start in range(0, ntr, self.block_size):
                end = min(start + self.block_size, ntr)
                block = depth_padded_flat[:, start:end]
                interp_func = interp1d(
                    twt_padded,
                    block,
                    kind=self.kind,
                    axis=0,
                    bounds_error=False,
                    fill_value=0.0,
                )
                out[:, start:end] = interp_func(self.time_axis)
            return out

        # twt_padded is 2D: process in blocks and use per-block interp1d with
        # the corresponding twt columns. depth_padded_flat shape is (nz+1, ntr)
        out = np.zeros((nt, ntr), dtype=depth_padded_flat.dtype)
        for start in range(0, ntr, self.block_size):
            end = min(start + self.block_size, ntr)
            twt_block = twt_padded[:, start:end]
            depth_block = depth_padded_flat[:, start:end]

            # If all columns in block share identical twt, we can vectorize
            if np.allclose(
                twt_block, np.broadcast_to(twt_block[:, 0:1], twt_block.shape)
            ):
                interp_func = interp1d(
                    twt_block[:, 0],
                    depth_block,
                    kind=self.kind,
                    axis=0,
                    bounds_error=False,
                    fill_value=0.0,
                )
                out[:, start:end] = interp_func(self.time_axis)
            else:
                # Mixed twt: perform per-column interp using loop (small block)
                for col in range(start, end):
                    col_idx = col - start
                    twt_col = twt_block[:, col_idx]
                    depth_col = depth_block[:, col_idx]
                    interp_func = interp1d(
                        twt_col,
                        depth_col,
                        kind=self.kind,
                        bounds_error=False,
                        fill_value=0.0,
                    )
                    out[:, col] = interp_func(self.time_axis)

        return out

    def nearest(
        self, twt_padded: np.ndarray, depth_padded_flat: np.ndarray
    ) -> np.ndarray:
        """Vectorized nearest-neighbor selection using searchsorted.

        twt_padded: 1D of length nz+1
        depth_padded_flat: (nz+1, ntr)
        returns (nt, ntr)
        """
        nt = len(self.time_axis)

        # twt_padded may be 1D or 2D. If 1D, do the fast vectorized path.
        twt_is_2d = twt_padded.ndim == 2
        ntr = depth_padded_flat.shape[1]

        if not twt_is_2d:
            nzp1 = twt_padded.shape[0]
            # use searchsorted on the padded twt axis
            idx = np.searchsorted(twt_padded, self.time_axis, side="left")
            upper = np.minimum(idx, nzp1 - 1)
            lower = np.maximum(idx - 1, 0)

            t_lower = twt_padded[lower]
            t_upper = twt_padded[upper]
            choose_lower = np.abs(self.time_axis - t_lower) <= np.abs(
                t_upper - self.time_axis
            )
            nearest_idx = np.where(choose_lower, lower, upper)

            # depth_padded_flat shape (nz+1, ntr)
            # gather rows nearest_idx for all traces
            if ntr <= self.block_size:
                res = depth_padded_flat[nearest_idx, :]
                return res

            out = np.zeros((nt, ntr), dtype=depth_padded_flat.dtype)
            for start in range(0, ntr, self.block_size):
                end = min(start + self.block_size, ntr)
                out[:, start:end] = depth_padded_flat[nearest_idx, start:end]
            return out

        # 2D twt_padded: compute nearest per-block. For each target time sample
        # and each trace in the block compute whether lower or upper is closer.
        out = np.zeros((nt, ntr), dtype=depth_padded_flat.dtype)
        for start in range(0, ntr, self.block_size):
            end = min(start + self.block_size, ntr)
            twt_block = twt_padded[:, start:end]
            depth_block = depth_padded_flat[:, start:end]

            # For each column in block compute searchsorted indices
            # twt_block shape (nzp1, nblock)
            nzp1 = twt_block.shape[0]
            # compute idx per column by applying searchsorted per column
            # We'll vectorize by using broadcasting over time_axis
            # For each column, do searchsorted by using np.searchsorted on flattened
            # but simplest here is a loop over columns within the block.
            nblock = end - start
            for col in range(nblock):
                twt_col = twt_block[:, col]
                idx = np.searchsorted(twt_col, self.time_axis, side="left")
                upper = np.minimum(idx, nzp1 - 1)
                lower = np.maximum(idx - 1, 0)
                t_lower = twt_col[lower]
                t_upper = twt_col[upper]
                choose_lower = np.abs(self.time_axis - t_lower) <= np.abs(
                    t_upper - self.time_axis
                )
                nearest_idx = np.where(choose_lower, lower, upper)
                out[:, start + col] = depth_block[nearest_idx, col]

        return out


__all__ = ["BatchedInterpolator"]
