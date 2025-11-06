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

from . import wavelets, signal, reflectivity, domain

# Re-export main classes and functions
from .wavelets import Wavelet, RickerWavelet
from .signal import SeismicSignalProcessor
from .reflectivity import ZoeppritzSolver
from .domain import DepthTimeConverter

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
