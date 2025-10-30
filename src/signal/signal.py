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
    "impedance_to_seismogram",
    "impedance_to_seismogram_depth",
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

    def impedance_to_seismogram(
        self,
        impedance,
        dt,
        f_peak=30,
        progress_every: Optional[int] = 30,
        prefix: str = "",
    ):
        # Prefer local package implementations but gracefully fallback
        try:
            from src.signal import wavelets
            from src.signal.reflectivity import ReflectivityCalculator

            refl_calc = ReflectivityCalculator()
            refl = refl_calc.reflectivity_from_ai(impedance)
        except Exception:
            # Fallback to utils-level imports if relative import fails
            try:
                from src.utils import wavelets
                from src.utils import reflectivity as refl_module

                refl = refl_module.reflectivity_from_ai(impedance)
            except Exception:
                raise
        self.logger.info("  Reflectivity range: [%.6f, %.6f]", refl.min(), refl.max())
        wavelet = wavelets.ricker_wavelet(f_peak=f_peak, dt=dt)
        self.logger.info("  Wavelet: %d samples at %s Hz", len(wavelet), f_peak)
        seismogram = self.apply_wavelet_to_cube(
            refl, wavelet, mode="same", progress_every=progress_every, prefix=prefix
        )
        self.logger.info(
            "  Seismogram range: [%.6f, %.6f]", seismogram.min(), seismogram.max()
        )
        return seismogram

    def impedance_to_seismogram_depth(
        self,
        impedance,
        dz,
        f_peak=30,
        progress_every: Optional[int] = 30,
        prefix: str = "",
    ):
        dt_equiv = dz / 2500.0
        return self.impedance_to_seismogram(
            impedance,
            dt=dt_equiv,
            f_peak=f_peak,
            progress_every=progress_every,
            prefix=prefix,
        )


# Backwards-compatible top-level wrapper functions

# Module-level lazy singleton processor
seismic_processor: SeismicProcessor = LazyObjectProxy(lambda: SeismicProcessor())


def get_seismic_processor(instance: SeismicProcessor | None = None) -> SeismicProcessor:
    return _impl_get_seismic_processor(instance)


def _impl_get_seismic_processor(
    instance: SeismicProcessor | None = None,
) -> SeismicProcessor:
    return instance if instance is not None else seismic_processor


_default_processor = SeismicProcessor()


def apply_wavelet_to_cube(
    refl_cube: np.ndarray,
    wavelet: np.ndarray,
    mode: str = "same",
    progress_every: Optional[int] = 30,
    prefix: str = "",
) -> np.ndarray:
    return _impl_apply_wavelet_to_cube(
        refl_cube, wavelet, mode=mode, progress_every=progress_every, prefix=prefix
    )


def impedance_to_seismogram(
    impedance, dt, f_peak=30, progress_every: Optional[int] = 30, prefix: str = ""
):
    return _impl_impedance_to_seismogram(
        impedance, dt, f_peak=f_peak, progress_every=progress_every, prefix=prefix
    )


def impedance_to_seismogram_depth(
    impedance, dz, f_peak=30, progress_every: Optional[int] = 30, prefix: str = ""
):
    return _impl_impedance_to_seismogram_depth(
        impedance, dz, f_peak=f_peak, progress_every=progress_every, prefix=prefix
    )


def _impl_apply_wavelet_to_cube(
    refl_cube: np.ndarray,
    wavelet: np.ndarray,
    mode: str = "same",
    progress_every: Optional[int] = 30,
    prefix: str = "",
) -> np.ndarray:
    """Canonical implementation delegating to the default SeismicProcessor.

    This provides a single implementation point that callers and tests can
    reference without depending on module-level mutable defaults.
    """
    return _default_processor.apply_wavelet_to_cube(
        refl_cube, wavelet, mode=mode, progress_every=progress_every, prefix=prefix
    )


def _impl_impedance_to_seismogram(
    impedance, dt, f_peak=30, progress_every: Optional[int] = 30, prefix: str = ""
) -> np.ndarray:
    return _default_processor.impedance_to_seismogram(
        impedance, dt, f_peak=f_peak, progress_every=progress_every, prefix=prefix
    )


def _impl_impedance_to_seismogram_depth(
    impedance, dz, f_peak=30, progress_every: Optional[int] = 30, prefix: str = ""
) -> np.ndarray:
    return _default_processor.impedance_to_seismogram_depth(
        impedance, dz, f_peak=f_peak, progress_every=progress_every, prefix=prefix
    )
