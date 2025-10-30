"""Wavelet helpers.

Utilities for generating common seismic wavelets.
"""

import numpy as np
import logging
from src.utils.facades import LazyObjectProxy

__all__ = ["Wavelet", "WaveletHelper", "wavelet_helper", "get_wavelet_helper"]

# Module logger
logger = logging.getLogger(__name__)


class Wavelet:
    """Simple container for wavelet samples and metadata.

    The class is intentionally small — it provides a named object to carry
    wavelet samples, sampling interval, and convenience methods for common
    conversions.
    """

    def __init__(self, samples: np.ndarray, dt: float):
        self.samples = np.asarray(samples)
        self.dt = float(dt)

    @property
    def nsamples(self) -> int:
        return self.samples.shape[0]

    def as_array(self) -> np.ndarray:
        return self.samples


# Thin facade helper for callers that prefer an OO API
class WaveletHelper:
    def ricker_wavelet(
        self, f_peak: float, length: float = 0.128, dt: float = 0.002
    ) -> np.ndarray:
        # time axis centered at zero
        t = np.arange(-length / 2, length / 2, dt, dtype=float)
        pi_sq = np.pi**2
        f_sq = f_peak**2
        t_sq = t**2

        # Ricker wavelet formula
        term1 = 1 - 2 * pi_sq * f_sq * t_sq
        term2 = np.exp(-pi_sq * f_sq * t_sq)

        samples = term1 * term2
        return samples


# Module-level lazy proxy using the shared LazyObjectProxy
wavelet_helper = LazyObjectProxy(lambda: WaveletHelper())

__all__.extend(["WaveletHelper", "wavelet_helper"])


def get_wavelet_helper(config: dict | None = None):
    """Return the module-level `wavelet_helper` proxy when `config` is None,
    otherwise return a new `WaveletHelper` instance. This mirrors the
    `get_default_*` helpers used elsewhere and centralizes access patterns.
    """
    return _impl_get_wavelet_helper(config)


def _impl_get_wavelet_helper(config: dict | None = None):
    """Canonical getter for the module-level wavelet_helper proxy.

    When `config` is None the lazy proxy is returned. If a config dict is
    provided a new WaveletHelper instance is returned so callers can inject
    configured helpers during testing.
    """
    if config is None:
        return wavelet_helper
    return WaveletHelper()


__all__.append("get_wavelet_helper")


# A simple function-level `ricker_wavelet(...)` helper is provided for
# convenience. Callers may use this function or the `WaveletHelper` facade
# via the `wavelet_helper` proxy depending on style preference.
def ricker_wavelet(f_peak: float, length: float = 0.128, dt: float = 0.002):
    """Return Ricker wavelet samples (compatibility wrapper).

    Callers may continue to import and use this function while the module
    also exposes an OO facade via `WaveletHelper` and the `wavelet_helper`
    proxy.
    """
    return _impl_ricker_wavelet(f_peak, length=length, dt=dt)


def _impl_ricker_wavelet(f_peak: float, length: float = 0.128, dt: float = 0.002):
    return wavelet_helper.ricker_wavelet(f_peak=f_peak, length=length, dt=dt)


__all__.extend(["ricker_wavelet", "_impl_ricker_wavelet"])
