"""Signal processing for seismic data.

Provides a clean OOP interface for applying wavelets to seismic reflectivity
data to produce synthetic seismograms.
"""

from typing import Any, TYPE_CHECKING, Optional, Union
import numpy as np
from numpy.typing import NDArray
import logging

if TYPE_CHECKING:
    from src.signal.wavelets import Wavelet

logger = logging.getLogger(__name__)

__all__ = [
    "SeismicSignalProcessor",
]


class SeismicSignalProcessor:
    """Processes seismic signals through wavelet convolution.

    Handles application of wavelets to reflectivity data to produce
    synthetic seismograms using FFT-based convolution.
    """

    def __init__(self, progress_every: Optional[int] = None):
        """Initialize the signal processor.

        Args:
            progress_every: Log progress every N traces (None to disable)
        """
        self.progress_every = progress_every
        self._scipy_available = self._check_scipy()

    @staticmethod
    def _check_scipy() -> bool:
        """Check if scipy is available."""
        try:
            import importlib.util

            return importlib.util.find_spec("scipy.signal") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def apply_wavelet(
        self,
        reflectivity: NDArray[np.floating[Any]],
        wavelet: Union[NDArray[np.floating[Any]], "Wavelet"],
        mode: str = "same",
        prefix: str = "",
    ) -> NDArray[np.floating[Any]]:
        """Apply wavelet to reflectivity cube via convolution.

        Performs trace-by-trace convolution of the reflectivity cube with
        the given wavelet to produce a seismogram cube using FFT convolution
        for efficiency.

        Args:
            reflectivity: Reflectivity cube of shape (ni, nj, nk)
            wavelet: Wavelet samples (1D array) or Wavelet object
            mode: Convolution mode ('same', 'full', 'valid')
            prefix: Prefix string for progress messages

        Returns:
            Seismogram cube of same shape as reflectivity

        Raises:
            ImportError: If scipy is not available
            ValueError: If shapes are incompatible
        """
        if not self._scipy_available:
            raise ImportError(
                "scipy required for wavelet convolution. "
                "Install with: pip install scipy"
            )

        # Extract samples from Wavelet object if needed
        from src.signal.wavelets import Wavelet

        wavelet_arr = wavelet.samples if isinstance(wavelet, Wavelet) else wavelet
        wavelet_arr = np.asarray(wavelet_arr, dtype=np.float64)

        # Validate inputs
        if reflectivity.ndim != 3:
            raise ValueError(f"reflectivity must be 3D, got shape {reflectivity.shape}")
        if wavelet_arr.ndim != 1:
            raise ValueError(f"wavelet must be 1D, got shape {wavelet_arr.shape}")

        # Import here to allow graceful failure if scipy unavailable
        from scipy.signal import fftconvolve

        ni, nj, _ = reflectivity.shape
        seismogram = np.zeros_like(reflectivity, dtype=np.float64)

        # Apply wavelet to each trace
        for i in range(ni):
            if self.progress_every and i % self.progress_every == 0:
                pct = (i * 100) // ni
                logger.debug("%sProgress: %d/%d (%d%%)", prefix, i, ni, pct)
            for j in range(nj):
                trace = reflectivity[i, j, :]
                seismogram[i, j, :] = fftconvolve(trace, wavelet_arr, mode=mode)

        return seismogram
