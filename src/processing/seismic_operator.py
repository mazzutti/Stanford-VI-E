"""SeismicOperator utilities.

Provides common seismic operations: compute reflectivity using existing
reflectivity helpers and convolve a reflectivity cube with a wavelet using
an efficient 3D convolution helper present in `src.modeling` or `src.signal`.
This module centralizes these operations so callers don't reimplement loops.
"""

from __future__ import annotations

import numpy as np

from scipy.signal import fftconvolve

from src.utils.quantity import Quantity
from src.utils.facades import LazyObjectProxy
import logging

__all__ = ["SeismicOperator"]

# Module logger
logger = logging.getLogger(__name__)


class SeismicOperator:
    """High-level operator to create seismograms from reflectivity and
    related derived attributes.

    All methods are pure functions operating on numpy arrays. They intentionally
    avoid side effects and return new numpy arrays for clarity.
    """

    @staticmethod
    # Per-technique helpers are not provided here; use reflectivity helpers
    @staticmethod
    def convolve_reflectivity_with_wavelet(
        reflectivity_cube: np.ndarray,
        wavelet: np.ndarray,
        mode: str = "same",
    ) -> np.ndarray:
        """Convolve a reflectivity cube with a 1D wavelet along the sample axis.

        This is a simple wrapper around fftconvolve to keep callers concise.
        """
        # Accept Quantity-wrapped reflectivity_cube
        if isinstance(reflectivity_cube, Quantity):
            refl_arr = reflectivity_cube.array
        else:
            refl_arr = np.asarray(reflectivity_cube)

        ni, nj, nk = refl_arr.shape
        out = np.zeros_like(refl_arr, dtype=float)
        for i in range(ni):
            for j in range(nj):
                out[i, j, :] = fftconvolve(refl_arr[i, j, :], wavelet, mode=mode)

        # If caller passed a Quantity for reflectivity, return a Quantity for
        # the seismic result (use 'seismic' unit label); otherwise return ndarray.
        try:
            if isinstance(reflectivity_cube, Quantity):
                return Quantity(out, "seismic")
        except Exception:
            pass

        return out


# Module-level singleton for convenience (placed after class definition)


# Module-level lazy proxy using shared LazyObjectProxy
seismic_operator = LazyObjectProxy(lambda: SeismicOperator())


def convolve_reflectivity_with_wavelet(
    reflectivity_cube: np.ndarray, wavelet: np.ndarray, mode: str = "same"
) -> np.ndarray:
    return _impl_convolve_reflectivity_with_wavelet(
        reflectivity_cube, wavelet, mode=mode
    )


__all__.extend(["seismic_operator", "convolve_reflectivity_with_wavelet"])


def get_seismic_operator(op: SeismicOperator | None = None) -> "SeismicOperator":
    """Return the provided SeismicOperator or the module-level lazy singleton."""
    return _impl_get_seismic_operator(op)


__all__.append("get_seismic_operator")


def _impl_convolve_reflectivity_with_wavelet(
    reflectivity_cube: np.ndarray, wavelet: np.ndarray, mode: str = "same"
) -> np.ndarray:
    """Canonical implementation: delegate to SeismicOperator static method."""
    return SeismicOperator.convolve_reflectivity_with_wavelet(
        reflectivity_cube, wavelet, mode=mode
    )


def _impl_get_seismic_operator(op: SeismicOperator | None = None) -> "SeismicOperator":
    """Canonical implementation for obtaining a SeismicOperator instance.

    Maintains DI-friendly behaviour: return provided instance or the
    module-level lazy proxy.
    """
    return op if op is not None else seismic_operator
