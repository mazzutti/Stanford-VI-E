"""Wavelet generation and manipulation.

Provides a clean OOP interface for creating and working with seismic wavelets.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Wavelet",
    "RickerWavelet",
]

# Module logger
logger = logging.getLogger(__name__)


class Wavelet:
    """Base class representing a seismic wavelet.

    Encapsulates wavelet samples and metadata with convenient access methods.

    Attributes:
        samples: Wavelet amplitude samples (numpy array)
        dt: Sampling interval in seconds
    """

    def __init__(self, samples: NDArray[np.floating[Any]], dt: float):
        """Initialize a wavelet.

        Args:
            samples: Array of wavelet amplitudes
            dt: Sampling interval in seconds

        Raises:
            ValueError: If dt is not positive
        """
        self.samples = np.asarray(samples, dtype=np.float64)
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = float(dt)

    @property
    def nsamples(self) -> int:
        """Number of samples in the wavelet."""
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """Duration of the wavelet in seconds."""
        return self.dt * self.nsamples

    def __len__(self) -> int:
        """Return number of samples."""
        return self.nsamples

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"nsamples={self.nsamples}, dt={self.dt:.6f}s, "
            f"duration={self.duration:.6f}s)"
        )


class RickerWavelet(Wavelet):
    """A Ricker (zero-phase) wavelet.

    A commonly used wavelet in seismic modeling with known analytical properties.

    Attributes:
        f_peak: Peak frequency in Hz
        length: Total wavelet length in seconds
    """

    def __init__(
        self,
        f_peak: float,
        length: float = 0.128,
        dt: float = 0.002,
    ):
        """Generate a Ricker wavelet.

        Args:
            f_peak: Peak frequency in Hz (must be positive)
            length: Wavelet length in seconds (default 0.128)
            dt: Sampling interval in seconds (default 0.002)

        Raises:
            ValueError: If f_peak or length are not positive
        """
        if f_peak <= 0:
            raise ValueError(f"f_peak must be positive, got {f_peak}")
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")

        self.f_peak = float(f_peak)
        self.length = float(length)

        # Generate Ricker wavelet samples
        samples = self._compute_samples(f_peak, length, dt)
        super().__init__(samples, dt)

    @staticmethod
    def _compute_samples(
        f_peak: float, length: float, dt: float
    ) -> NDArray[np.floating[Any]]:
        """Compute Ricker wavelet samples.

        Uses the analytical formula:
        $ \\psi(t) = (1 - 2\\pi^2 f^2 t^2) \\exp(-\\pi^2 f^2 t^2) $

        Args:
            f_peak: Peak frequency in Hz
            length: Wavelet length in seconds
            dt: Sampling interval in seconds

        Returns:
            Array of wavelet samples
        """
        # Time axis centered at zero
        t = np.arange(-length / 2, length / 2, dt, dtype=np.float64)
        pi_sq = np.pi**2
        f_sq = f_peak**2
        t_sq = t**2

        # Ricker wavelet formula
        term1 = 1 - 2 * pi_sq * f_sq * t_sq
        term2 = np.exp(-pi_sq * f_sq * t_sq)
        return np.asarray(term1 * term2, dtype=np.float64)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RickerWavelet(f_peak={self.f_peak:.2f}Hz, "
            f"nsamples={self.nsamples}, dt={self.dt:.6f}s)"
        )
