"""ResamplerService - High-level resampling service.


Lightweight service that composes DepthTimeResampler, ResamplePlanCache,
and BackendManager to provide a single entrypoint for depth<->time resampling.


The service is intentionally thin: it delegates heavy work to DepthTimeResampler
and to the registered backends while ensuring the shared plan cache is used when
possible.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Optional, Tuple, Any


from numpy.typing import ArrayLike, NDArray
import logging


from src.io.grid import GridSpec
from src.processing.resampling._plan import ResamplePlan
from src.processing.resampling._cache import get_resample_plan_cache
from src.processing.resampling.backends._manager import BackendManager
from src.utils.quantity import Quantity, to_ndarray


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

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = get_resample_plan_cache()
        from src.processing.resampling._resampler import DepthTimeResampler

        self._inner = DepthTimeResampler(grid_spec=self.grid_spec)
        self._backend_mgr = BackendManager()

    def depth_to_time(
        self,
        data_depth: NDArray[Any] | Quantity | ArrayLike,
        vp_depth: NDArray[Any] | Quantity | ArrayLike,
        target_dt: Optional[float] = None,
        target_nt: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[NDArray[Any] | Quantity, float]:
        """Resample depth-sampled property to regular time using a ResamplePlan.

        This method will attempt to fetch a cached ResamplePlan for the given
        vp_depth (if `use_cache` is True) before creating a new one.
        """
        # unwrap quantities to compute plan key
        vp_arr = to_ndarray(vp_depth)
        if use_cache:
            if self.cache is not None:
                plan = self.cache.get_plan(
                    self.grid_spec, vp_arr, target_dt=target_dt, target_nt=target_nt
                )
            else:
                plan = ResamplePlan.create(
                    self.grid_spec, vp_arr, target_dt=target_dt, target_nt=target_nt
                )
        else:
            plan = ResamplePlan.create(
                self.grid_spec, vp_arr, target_dt=target_dt, target_nt=target_nt
            )

        # Delegate to the inner resampler
        # Cast to proper types to ensure type safety
        data_arr = to_ndarray(data_depth)
        vp_arr_resampler = to_ndarray(vp_depth)
        out = self._inner.depth_to_time_cube(data_arr, vp_arr_resampler, plan=plan)
        return out

    def time_to_depth(
        self,
        seismogram_time: NDArray[Any] | Quantity | ArrayLike,
        vp_depth: NDArray[Any] | Quantity | ArrayLike,
        use_cache: bool = True,
    ) -> NDArray[Any] | Quantity:
        """Resample a time-domain seismogram to depth using a ResamplePlan.

        Uses the shared cache similarly to `depth_to_time`.
        """
        vp_arr = to_ndarray(vp_depth)
        if use_cache:
            if self.cache is not None:
                plan = self.cache.get_plan(self.grid_spec, vp_arr)
            else:
                plan = ResamplePlan.create(self.grid_spec, vp_arr)
        else:
            plan = ResamplePlan.create(self.grid_spec, vp_arr)

        # Cast to proper types to ensure type safety
        data_arr = to_ndarray(seismogram_time)
        vp_arr_resampler = to_ndarray(vp_depth)
        out = self._inner.time_to_depth_cube(data_arr, vp_arr_resampler, plan=plan)
        return out
