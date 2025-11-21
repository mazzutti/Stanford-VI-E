"""BatchedInterpolator

Provides a small helper to perform vectorized or block-wise interpolation of
depth->time (and nearest-neighbor categorical) for many traces at once.

This centralizes the flatten/reshape/block logic used in the resampler and
provides a single place to tune block size vs memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

@dataclass
class BatchedInterpolator:
    """Helper to perform batched/block interpolation over many traces.

    This dataclass centralizes flatten/reshape and block logic so callers
    can interpolate many traces with a single vectorized or blocked call.
    """

    time_axis: ArrayLike
    kind: str = "linear"
    block_size: int = 65536

    def interpolate(
        self, twt_padded: NDArray[Any], depth_padded_flat: NDArray[Any]
    ) -> NDArray[Any]:
        """Interpolate depth_padded_flat (shape (nz+1, ntraces)) onto self.time_axis.

        Returns: array shaped (nt, ntraces)
        """
        # Keep the top-level flow minimal by delegating heavy work to helpers.
        _, ntr = depth_padded_flat.shape
        time_axis_arr = np.asarray(self.time_axis)
        nt = len(time_axis_arr)

        twt_is_2d = twt_padded.ndim == 2

        if not twt_is_2d:
            # 1D twt: either full vectorized or block-vectorized
            if ntr <= self.block_size:
                return self._interp_vectorized(
                    twt_padded, depth_padded_flat, time_axis_arr
                )

            out = np.zeros((nt, ntr), dtype=depth_padded_flat.dtype)
            for start in range(0, ntr, self.block_size):
                end = min(start + self.block_size, ntr)
                block = depth_padded_flat[:, start:end]
                out[:, start:end] = self._interp_vectorized(
                    twt_padded, block, time_axis_arr
                )
            return out

        # 2D twt: per-block handling (vectorized when block columns share twt)
        out = np.zeros((nt, ntr), dtype=depth_padded_flat.dtype)
        for start in range(0, ntr, self.block_size):
            end = min(start + self.block_size, ntr)
            twt_block = twt_padded[:, start:end]
            depth_block = depth_padded_flat[:, start:end]

            if np.allclose(
                twt_block, np.broadcast_to(twt_block[:, 0:1], twt_block.shape)
            ):
                out[:, start:end] = self._interp_vectorized(
                    twt_block[:, 0], depth_block, time_axis_arr
                )
            else:
                out[:, start:end] = self._interp_per_column(
                    twt_block, depth_block, time_axis_arr
                )

        return out

    # Helper methods extracted to reduce local-variable footprint in public methods
    def _interp_vectorized(
        self,
        twt_1d: NDArray[Any],
        depth_block: NDArray[Any],
        time_axis_arr: NDArray[Any],
    ) -> NDArray[Any]:
        interp_func = interp1d(
            cast(NDArray[np.float64], twt_1d),
            cast(NDArray[np.float64], depth_block),
            kind=cast(Any, self.kind),
            axis=0,
            bounds_error=False,
            fill_value=0.0,
        )
        return cast(NDArray[Any], interp_func(cast(NDArray[np.float64], time_axis_arr)))

    def _interp_per_column(
        self,
        twt_block: NDArray[Any],
        depth_block: NDArray[Any],
        time_axis_arr: NDArray[Any],
    ) -> NDArray[Any]:
        nblock = twt_block.shape[1]
        nt = len(time_axis_arr)
        out = np.zeros((nt, nblock), dtype=depth_block.dtype)
        for col in range(nblock):
            twt_col = twt_block[:, col]
            depth_col = depth_block[:, col]
            interp_func = interp1d(
                cast(NDArray[np.float64], twt_col),
                cast(NDArray[np.float64], depth_col),
                kind=cast(Any, self.kind),
                bounds_error=False,
                fill_value=0.0,
            )
            out[:, col] = cast(
                NDArray[Any], interp_func(cast(NDArray[np.float64], time_axis_arr))
            )
        return out

    def _nearest_vectorized(
        self,
        twt_1d: NDArray[Any],
        depth_padded_flat: NDArray[Any],
        time_axis_arr: NDArray[Any],
    ) -> NDArray[Any]:
        nzp1 = twt_1d.shape[0]
        idx = np.searchsorted(twt_1d, time_axis_arr, side="left")
        upper = np.minimum(idx, nzp1 - 1)
        lower = np.maximum(idx - 1, 0)

        nearest_idx = np.where(
            np.abs(time_axis_arr - twt_1d[lower])
            <= np.abs(twt_1d[upper] - time_axis_arr),
            lower,
            upper,
        )

        ntr = depth_padded_flat.shape[1]
        if ntr <= self.block_size:
            return depth_padded_flat[nearest_idx, :]

        out = np.zeros((len(time_axis_arr), ntr), dtype=depth_padded_flat.dtype)
        for start in range(0, ntr, self.block_size):
            end = min(start + self.block_size, ntr)
            out[:, start:end] = depth_padded_flat[nearest_idx, start:end]
        return out

    def _nearest_per_block(
        self,
        twt_block: NDArray[Any],
        depth_block: NDArray[Any],
        time_axis_arr: NDArray[Any],
    ) -> NDArray[Any]:
        nzp1 = twt_block.shape[0]
        nblock = twt_block.shape[1]
        out = np.zeros((len(time_axis_arr), nblock), dtype=depth_block.dtype)
        for col in range(nblock):
            twt_col = twt_block[:, col]
            idx = np.searchsorted(twt_col, time_axis_arr, side="left")
            upper = np.minimum(idx, nzp1 - 1)
            lower = np.maximum(idx - 1, 0)
            nearest_idx = np.where(
                np.abs(time_axis_arr - twt_col[lower])
                <= np.abs(twt_col[upper] - time_axis_arr),
                lower,
                upper,
            )
            out[:, col] = depth_block[nearest_idx, col]
        return out

    # public `nearest` is implemented later (below helpers) to keep helpers grouped

    def nearest(
        self, twt_padded: NDArray[Any], depth_padded_flat: NDArray[Any]
    ) -> NDArray[Any]:
        """Vectorized nearest-neighbor selection using searchsorted.

        twt_padded: 1D or 2D of length nz+1
        depth_padded_flat: (nz+1, ntr)
        returns (nt, ntr)
        """
        time_axis_arr = np.asarray(self.time_axis)
        nt = len(time_axis_arr)

        twt_is_2d = twt_padded.ndim == 2
        ntr = depth_padded_flat.shape[1]

        if not twt_is_2d:
            return self._nearest_vectorized(
                twt_padded, depth_padded_flat, time_axis_arr
            )

        out = np.zeros((nt, ntr), dtype=depth_padded_flat.dtype)
        for start in range(0, ntr, self.block_size):
            end = min(start + self.block_size, ntr)
            twt_block = twt_padded[:, start:end]
            depth_block = depth_padded_flat[:, start:end]
            out[:, start:end] = self._nearest_per_block(
                twt_block, depth_block, time_axis_arr
            )

        return out

__all__ = ["BatchedInterpolator"]

# BatchedInterpolator is a focused helper extracted from the resampler to
# centralize block-vectorized interpolation logic. Keep implementations
# straightforward for readability and testability.
