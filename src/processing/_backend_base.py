from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import logging

from src.processing.resample_plan import ResamplePlan


class BackendError(RuntimeError):
    pass


class ResamplerBackend(Protocol):
    """Protocol that resampler backends should follow.

    Minimal methods required by the registry and DepthTimeResampler.
    """

    name: str

    def supports(self, plan: ResamplePlan) -> bool: ...

    def depth_to_time(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs: object
    ) -> BackendResult: ...

    def time_to_depth(
        self, data: np.ndarray, vp: np.ndarray, plan: ResamplePlan, **kwargs: object
    ) -> BackendResult: ...


@dataclass
class BackendResult:
    """Standardized result returned by resampler backends.

    - array: the output ndarray (depth->time or time->depth)
    - dt: optional sampling interval (for depth->time results)
    """

    array: np.ndarray
    dt: Optional[float] = None


def validate_backend_result(obj: object) -> bool:
    """Return True if `obj` is a BackendResult with a numpy array inside.

    Useful for callers that must enforce the backend contract.
    """
    if not isinstance(obj, BackendResult):
        return False
    if not hasattr(obj, "array"):
        return False
    try:
        import numpy as _np

        return _np.asarray(obj.array) is not None
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
