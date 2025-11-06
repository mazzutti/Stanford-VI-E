"""ResamplePlanCache

A small LRU cache for ResamplePlan instances keyed by the velocity array
content and grid/time parameters. Avoids recomputing the same plan when the
same vp cube is reused across multiple resampling calls.

The cache stores a bounded number of entries and evicts least-recently-used
plans when over capacity.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Optional

import numpy as np
import os
import logging

from src.processing.resampling._plan import ResamplePlan
from src.io.grid import GridSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CacheKey:
    grid_shape: tuple
    dz: float
    dt: float
    target_dt: Optional[float]
    target_nt: Optional[int]
    vp_hash: str


class ResamplePlanCache:
    """LRU cache for ResamplePlan objects.

    Usage:
        cache = ResamplePlanCache(maxsize=16)
        plan = cache.get_plan(grid_spec, vp_arr, target_dt=0.001)
    """

    def __init__(self, maxsize: int = 16):
        self.maxsize = int(maxsize)
        self._store: OrderedDict[_CacheKey, ResamplePlan] = OrderedDict()

    def _hash_vp(self, vp_arr: np.ndarray) -> str:
        # Use a fast hash based on shape, dtype and a sample of bytes.
        # For small arrays we hash the whole buffer; for larger arrays we
        # hash a few evenly spaced chunks to avoid excessive memory.
        arr = np.asarray(vp_arr)
        h = hashlib.md5()
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        # For arrays under 1MB hash whole buffer
        nbytes = arr.nbytes
        if nbytes <= 1024 * 1024:
            h.update(arr.tobytes())
        else:
            # sample up to 4 chunks
            chunks = 4
            step = max(1, nbytes // chunks)
            buf = arr.ravel().view(np.uint8)
            for start in range(0, min(nbytes, step * chunks), step):
                h.update(buf[start : start + min(step, nbytes - start)])
        return h.hexdigest()

    def _make_key(
        self,
        grid_spec: GridSpec,
        vp_arr: np.ndarray,
        target_dt: Optional[float],
        target_nt: Optional[int],
    ) -> _CacheKey:
        return _CacheKey(
            grid_shape=tuple(grid_spec.shape),
            dz=float(grid_spec.dz),
            dt=float(grid_spec.dt),
            target_dt=float(target_dt) if target_dt is not None else None,
            target_nt=int(target_nt) if target_nt is not None else None,
            vp_hash=self._hash_vp(vp_arr),
        )

    def get_plan(
        self,
        grid_spec: GridSpec,
        vp_arr: np.ndarray,
        target_dt: Optional[float] = None,
        target_nt: Optional[int] = None,
        block_size: int = 65536,
    ) -> ResamplePlan:
        """Return a ResamplePlan for the given arguments, using cache when possible."""
        key = self._make_key(grid_spec, vp_arr, target_dt, target_nt)
        if key in self._store:
            # move to end (most-recently used)
            plan = self._store.pop(key)
            self._store[key] = plan
            return plan

        # create plan and insert
        plan = ResamplePlan.create(
            grid_spec,
            vp_arr,
            target_dt=target_dt,
            target_nt=target_nt,
            block_size=block_size,
        )
        self._store[key] = plan
        # evict if necessary
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return plan


__all__ = ["ResamplePlanCache", "get_resample_plan_cache", "set_resample_plan_cache"]


# Module-level default cache (singleton). Consumers may override it by
# calling `set_resample_plan_cache(...)` prior to first use.
_DEFAULT_CACHE: Optional[ResamplePlanCache] = None


def get_resample_plan_cache(maxsize: int = 16) -> ResamplePlanCache:
    """Return the module-level ResamplePlanCache singleton, creating it
    if necessary. If a different maxsize is required, call this with
    the desired maxsize before other modules import the cache.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = ResamplePlanCache(maxsize=maxsize)
    return _DEFAULT_CACHE


def set_resample_plan_cache(cache: ResamplePlanCache) -> None:
    """Replace the module-level default cache with a caller-provided one."""
    global _DEFAULT_CACHE
    _DEFAULT_CACHE = cache
