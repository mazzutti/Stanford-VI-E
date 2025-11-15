from __future__ import annotations


from dataclasses import dataclass
from typing import Optional, Protocol, Any
from numpy.typing import NDArray


import logging


from src.processing.resampling._plan import ResamplePlan


class BackendError(RuntimeError):
    pass


class ResamplerBackend(Protocol):
    """Protocol that resampler backends should follow.

    Minimal methods required by the registry and DepthTimeResampler.
    """

    name: str

    def supports(self, plan: ResamplePlan) -> bool: ...

    def depth_to_time(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult: ...

    def time_to_depth(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult: ...


@dataclass
class BackendResult:
    """Standardized result returned by resampler backends.

    - array: the output ndarray (depth->time or time->depth)
    - dt: optional sampling interval (for depth->time results)
    """

    array: NDArray[Any]
    dt: Optional[float] = None


def validate_backend_result(obj: BackendResult | object) -> bool:
    """Return True if `obj` is a BackendResult with a numpy array inside.

    Useful for callers that must enforce the backend contract.
    """
    if not isinstance(obj, BackendResult):
        return False
    if not hasattr(obj, "array"):
        return False
    try:
        import numpy as _np

        # np.asarray(...) will either return an ndarray or raise; if it returns,
        # the backend provided a convertible array-like object so the validation
        # succeeds.
        _np.asarray(obj.array)
        return True
    except Exception:
        return False


__all__ = [
    "BackendError",
    "ResamplerBackend",
    "BackendResult",
    "validate_backend_result",
]


# Module logger
logger = logging.getLogger(__name__)
