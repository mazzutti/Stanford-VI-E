"""Signal-processing helpers.

Utilities for common seismic signal-processing tasks (wavelet application,
reflectivity, seismogram generation).
"""

from typing import Optional
import numpy as np
import logging

from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)


__all__ = [
    "apply_wavelet_to_cube",
    "SeismicProcessor",
]


class SeismicProcessor:
    """Object-oriented utility for seismic processing tasks.

    Groups related functions and holds configuration defaults where useful.
    """

    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)

    def apply_wavelet_to_cube(
        self,
        refl_cube: np.ndarray,
        wavelet: np.ndarray,
        mode: str = "same",
        progress_every: Optional[int] = 30,
        prefix: str = "",
    ) -> np.ndarray:
        try:
            from scipy.signal import fftconvolve
        except Exception:
            raise ImportError(
                "scipy required for apply_wavelet_to_cube. Install scipy."
            )

        ni, nj, nk = refl_cube.shape
        seismogram = np.zeros_like(refl_cube)

        for i in range(ni):
            if progress_every and i % progress_every == 0:
                self.logger.debug(
                    "%sProgress: %d/%d (%d%%)", prefix, i, ni, i * 100 // ni
                )
            row = refl_cube[i]
            for j in range(nj):
                seismogram[i, j, :] = fftconvolve(row[j, :], wavelet, mode=mode)

        return seismogram

    def to_seismogram(
        self,
        property_cube,
        dt,
        f_peak=30,
        progress_every: Optional[int] = 30,
        prefix: str = "",
    ):
        raise AttributeError("to_seismogram is not provided")

    def to_seismogram_depth(
        self,
        property_cube,
        dz,
        f_peak=30,
        progress_every: Optional[int] = 30,
        prefix: str = "",
    ):
        raise AttributeError("to_seismogram_depth is not provided")


# Module-level lazy singleton processor
seismic_processor: SeismicProcessor = LazyObjectProxy(lambda: SeismicProcessor())


def get_seismic_processor(instance: SeismicProcessor | None = None) -> SeismicProcessor:
    """Get the seismic processor instance."""
    return instance if instance is not None else seismic_processor


_default_processor = SeismicProcessor()


def apply_wavelet_to_cube(
    refl_cube: np.ndarray,
    wavelet: np.ndarray,
    mode: str = "same",
    progress_every: Optional[int] = 30,
    prefix: str = "",
) -> np.ndarray:
    """Apply wavelet to reflection cube."""
    return _default_processor.apply_wavelet_to_cube(
        refl_cube, wavelet, mode=mode, progress_every=progress_every, prefix=prefix
    )


# Property-to-seismogram helpers are intentionally not part of the public API.
