"""SeismicOperator utilities.

Provides common seismic operations: compute reflectivity using existing
reflectivity helpers and convolve a reflectivity cube with a wavelet using
an efficient 3D convolution helper present in `src.modeling` or `src.signal`.
This module centralizes these operations so callers don't reimplement loops.
"""

from __future__ import annotations

import numpy as np

from scipy.signal import fftconvolve

from src.signal.wavelets import ricker_wavelet
from src.utils.quantity import Quantity
import logging

__all__ = ["SeismicOperator"]

# Module logger
logger = logging.getLogger(__name__)


class SeismicOperator:
    """High-level operator to create seismograms from impedance/AI/reflectivity.

    All methods are pure functions operating on numpy arrays. They intentionally
    avoid side effects and return new numpy arrays for clarity.
    """

    @staticmethod
    def impedance_to_seismogram_depth(
        impedance: np.ndarray,
        grid_dz: float,
        f_peak: float = 30.0,
        dt_equiv_vp: float = 2500.0,
        mode: str = "same",
    ) -> np.ndarray:
        """Convert depth-domain impedance to a depth-sampled seismogram.

        Args:
            impedance: (ni, nj, nz) impedance cube in depth
            grid_dz: vertical sampling in meters
            f_peak: peak frequency of Ricker wavelet (Hz)
            dt_equiv_vp: average velocity to approximate dt_equiv = dz / vp
            mode: convolution mode for fftconvolve

        Returns:
            seismogram: (ni, nj, nz) depth-sampled seismogram
        """
        # Accept Quantity-wrapped impedance
        if isinstance(impedance, Quantity):
            imp_arr = impedance.array
        else:
            imp_arr = np.asarray(impedance)

        # Compute reflectivity using ReflectivityCalculator
        try:
            from src.signal.reflectivity import reflectivity_calc

            refl = reflectivity_calc.reflectivity_from_ai(imp_arr)
        except Exception:
            # Fallback to older module-style call if available
            from src.signal import reflectivity as refl_module

            refl = refl_module.reflectivity_from_ai(imp_arr)
        refl = np.clip(refl, -0.99, 0.99)

        # Estimate dt equivalent for depth-domain by dividing dz by typical vp
        dt_equiv = grid_dz / dt_equiv_vp
        wavelet = ricker_wavelet(f_peak=f_peak, dt=dt_equiv)

        ni, nj, nk = imp_arr.shape
        seismogram = np.zeros_like(imp_arr, dtype=float)

        for i in range(ni):
            for j in range(nj):
                seismogram[i, j, :] = fftconvolve(refl[i, j, :], wavelet, mode=mode)

        # If input was a Quantity, wrap output as Quantity('seismic') to
        # preserve semantic metadata; otherwise return plain ndarray. Use a
        # best-effort try/except to avoid failing when Quantity construction
        # is unavailable in minimal environments.
        try:
            if isinstance(impedance, Quantity):
                return Quantity(seismogram, "seismic")
        except Exception:
            pass

        return seismogram

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
from src.utils.facades import LazyObjectProxy


# Module-level lazy proxy using shared LazyObjectProxy
seismic_operator = LazyObjectProxy(lambda: SeismicOperator())


def impedance_to_seismogram_depth(
    impedance: np.ndarray,
    grid_dz: float,
    f_peak: float = 30.0,
    dt_equiv_vp: float = 2500.0,
    mode: str = "same",
):
    return seismic_operator.impedance_to_seismogram_depth(
        impedance, grid_dz, f_peak=f_peak, dt_equiv_vp=dt_equiv_vp, mode=mode
    )


def convolve_reflectivity_with_wavelet(
    reflectivity_cube: np.ndarray, wavelet: np.ndarray, mode: str = "same"
) -> np.ndarray:
    return seismic_operator.convolve_reflectivity_with_wavelet(
        reflectivity_cube, wavelet, mode=mode
    )


__all__.extend(
    [
        "seismic_operator",
        "impedance_to_seismogram_depth",
        "convolve_reflectivity_with_wavelet",
    ]
)


def get_seismic_operator(op: SeismicOperator | None = None) -> "SeismicOperator":
    """Return the provided SeismicOperator or the module-level lazy singleton."""
    return op if op is not None else seismic_operator


__all__.append("get_seismic_operator")



