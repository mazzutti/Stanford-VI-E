"""Resampling backend base types and utilities.

This module defines the backend protocol, a standard BackendResult
dataclass and helper functions used by resampling backend
implementations. Backends are small adapters that provide depth<->time
resampling implementations and must satisfy the :class:`ResamplerBackend`
protocol described here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.processing.resampling._plan import ResamplePlan


class BackendError(RuntimeError):
    """Raised for backend-specific errors during resampling operations.

    Backends should raise this error for internal failures that callers may
    want to catch and handle specially.
    """


class ResamplerBackend(Protocol):
    """Protocol that resampler backends should follow.

    Minimal methods required by the registry and DepthTimeResampler.
    """

    name: str

    def supports(self, plan: ResamplePlan) -> bool:
        """Return True when this backend supports the provided resample plan."""
        raise NotImplementedError()

    def depth_to_time(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        """Resample `data` from depth to time according to `plan`.

        Return a :class:`BackendResult` containing the output array and optional dt.
        """
        raise NotImplementedError()

    def time_to_depth(
        self, data: NDArray[Any], vp: NDArray[Any], plan: ResamplePlan, **kwargs: Any
    ) -> BackendResult:
        """Resample `data` from time to depth according to `plan`.

        Return a :class:`BackendResult` containing the output array.
        """
        raise NotImplementedError()


@dataclass
class BackendResult:
    """Standardized result returned by resampler backends.

    - array: the output ndarray (depth->time or time->depth)
    - dt: optional sampling interval (for depth->time results)
    """

    array: NDArray[Any]
    dt: float | None = None


def validate_backend_result(obj: BackendResult | object) -> bool:
    """Return True if `obj` is a BackendResult with a numpy array inside.

    Useful for callers that must enforce the backend contract.
    """
    if not isinstance(obj, BackendResult):
        return False
    if not hasattr(obj, "array"):
        return False
    try:
        # np.asarray(...) will either return an ndarray or raise; if it returns,
        # the backend provided a convertible array-like object so the validation
        # succeeds.
        np.asarray(obj.array)
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "BackendError",
    "ResamplerBackend",
    "BackendResult",
    "validate_backend_result",
]


# Module logger
logger = logging.getLogger(__name__)
