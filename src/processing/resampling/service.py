"""ResamplerService - High-level resampling service.

Lightweight service that composes DepthTimeResampler, ResamplePlanCache,
and BackendManager to provide a single entrypoint for depth<->time resampling.

The service is intentionally thin: it delegates heavy work to DepthTimeResampler
and to the registered backends while ensuring the shared plan cache is used when
possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import time
from numpy.typing import ArrayLike
import logging

from src.io.grid import GridSpec
from src.processing.resampling.plan import ResamplePlan
from src.processing.resampling.cache import get_resample_plan_cache
from src.processing.resampling.backends.manager import get_backend_manager
from src.utils.quantity import Quantity
from src.processing.metrics import BackendMetrics, PlanFingerprint, get_global_metrics

__all__ = ["ResamplerService"]

# module logger
logger = logging.getLogger(__name__)


@dataclass
class ResamplerService:
    """Service wrapper around DepthTimeResampler that centralizes plan cache
    and backend selection.

    Attributes:
        grid_spec: GridSpec used to build ResamplePlan objects
        cache: optional ResamplePlanCache instance (defaults to module singleton)
    """

    grid_spec: GridSpec
    cache = None

    metrics: BackendMetrics | None = None

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = get_resample_plan_cache()
        from src.processing.resampling.resampler import get_resampler_factory

        self._inner = get_resampler_factory().get_resampler(self.grid_spec)
        self._backend_mgr = get_backend_manager()
        if self.metrics is None:
            # use the shared global metrics collector by default
            self.metrics = get_global_metrics()

    def depth_to_time(
        self,
        data_depth: np.ndarray | Quantity | ArrayLike,
        vp_depth: np.ndarray | Quantity | ArrayLike,
        target_dt: Optional[float] = None,
        target_nt: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray | Quantity, float]:
        """Resample depth-sampled property to regular time using a ResamplePlan.

        This method will attempt to fetch a cached ResamplePlan for the given
        vp_depth (if `use_cache` is True) before creating a new one.
        """
        # unwrap quantities to compute plan key
        vp_arr = (
            vp_depth.array if isinstance(vp_depth, Quantity) else np.asarray(vp_depth)
        )
        if use_cache:
            plan = self.cache.get_plan(
                self.grid_spec, vp_arr, target_dt=target_dt, target_nt=target_nt
            )
        else:
            plan = ResamplePlan.create(
                self.grid_spec, vp_arr, target_dt=target_dt, target_nt=target_nt
            )

        # Compute fingerprint for metrics
        fingerprint = PlanFingerprint.from_plan(plan)

        # Delegate to the inner resampler but time the selected backend path
        # The DepthTimeResampler consults the BackendManager internally; we
        # record the selection by querying the manager here (best-effort).
        backend = self._backend_mgr.get_best(plan)
        backend_name = (
            getattr(backend, "name", "none") if backend is not None else "none"
        )
        # Record selection into the global metrics collector so all services
        # and components share the same statistics.
        gm = get_global_metrics()
        if gm is not None and backend_name != "none":
            gm.record_selection(backend_name)

        start = time.time()
        out = self._inner.depth_to_time_cube(data_depth, vp_depth, plan=plan)
        elapsed = time.time() - start

        if gm is not None and backend_name != "none":
            gm.record_runtime(backend_name, fingerprint, elapsed)

        return out

    def time_to_depth(
        self,
        seismogram_time: np.ndarray | Quantity | ArrayLike,
        vp_depth: np.ndarray | Quantity | ArrayLike,
        use_cache: bool = True,
    ) -> np.ndarray | Quantity:
        """Resample a time-domain seismogram to depth using a ResamplePlan.

        Uses the shared cache similarly to `depth_to_time`.
        """
        vp_arr = (
            vp_depth.array if isinstance(vp_depth, Quantity) else np.asarray(vp_depth)
        )
        if use_cache:
            plan = self.cache.get_plan(self.grid_spec, vp_arr)
        else:
            plan = ResamplePlan.create(self.grid_spec, vp_arr)

        backend = self._backend_mgr.get_best(plan)
        backend_name = (
            getattr(backend, "name", "none") if backend is not None else "none"
        )
        gm = get_global_metrics()
        if gm is not None and backend_name != "none":
            gm.record_selection(backend_name)
        start = time.time()
        out = self._inner.time_to_depth_cube(seismogram_time, vp_depth, plan=plan)
        elapsed = time.time() - start
        if gm is not None and backend_name != "none":
            fingerprint = PlanFingerprint.from_plan(plan)
            gm.record_runtime(backend_name, fingerprint, elapsed)
        return out
