"""Signal processing package.

Provides utilities for seismic signal processing including:
- Wavelet generation (Ricker wavelets via OOP interface)
- Signal processing (wavelet convolution)
- Zoeppritz equation solving for reflection coefficients
- Domain conversions (depth to time)

Example:
    >>> from src.signal import RickerWavelet, SeismicSignalProcessor
    >>> wavelet = RickerWavelet(f_peak=25.0)
    >>> processor = SeismicSignalProcessor(progress_every=30)
    >>> seismogram = processor.apply_wavelet(reflectivity, wavelet)
"""

from . import domain, reflectivity, signal, wavelets
from .domain import DepthTimeConverter
from .reflectivity import ZoeppritzSolver
from .signal import SeismicSignalProcessor

# Re-export main classes and functions
from .wavelets import RickerWavelet, Wavelet

__all__ = [
    # Submodules
    "wavelets",
    "signal",
    "reflectivity",
    "domain",
    # Wavelet classes
    "Wavelet",
    "RickerWavelet",
    # Signal processing classes
    "SeismicSignalProcessor",
    # Reflectivity classes
    "ZoeppritzSolver",
    # Domain conversion classes
    "DepthTimeConverter",
]
