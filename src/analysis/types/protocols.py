"""Protocol definitions for analysis workflows.

This module defines structural types (Protocols) used across analysis modules.
These protocols enable duck-typing and flexible implementations while
maintaining static type safety through Python's Protocol system.
"""

from os import PathLike
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import numpy as np
from matplotlib.figure import Figure
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from src.io.grid import GridSpec
from src.io.loader import DatasetManager

if TYPE_CHECKING:
    from src.analysis.models import AvoResults

__all__ = [
    "T",
    "ResamplePlan",
    "Resampler",
    "ResamplerFactory",
    "TimeResampler",
    "CacheLoaderProtocol",
    "CacheProtocol",
    "SelectorProtocol",
    "ArchiveExtractorProtocol",
    "DatasetManagerFactory",
    "PlotterProtocol",
]
# Protocol helper objects are intentionally compact; suppress noisy
# too-few-public-methods warnings for these thin protocol shims.

# Type variable for generic protocols
T = TypeVar("T")

# ============================================================================
# Resampling Protocols
# ============================================================================
# Protocols for converting seismic data between time and depth domains.

class ResamplePlan(Protocol):
    """Opaque plan marker used by the resampler.

    This is a marker protocol that represents a resampling plan.
    Implementations may store plan data and state needed for resampling.
    """

class Resampler(Protocol):
    """Protocol for resampling seismic data in depth-time conversion.

    A resampler handles conversion of seismic arrays between time and depth
    domains, taking into account velocity information and resampling plans.
    """

    def time_to_depth_cube(
        self,
        seismogram_time: NDArray[np.float64],
        vp_depth: NDArray[np.float64],
        plan: ResamplePlan,
    ) -> NDArray[np.float64]:
        """Convert time-domain seismic data to depth domain.

        Args:
            seismogram_time: Seismic array in time domain with shape (time_samples, ...)
            vp_depth: P-wave velocity in depth domain for interpolation
            plan: Resampling plan containing conversion parameters

        Returns:
            Seismic array converted to depth domain
        """
        raise NotImplementedError()

class ResamplerFactory(Protocol):
    """Factory for creating Resampler instances.

    Provides a factory method to create resampler instances configured
    for a specific grid specification.
    """

    def get_resampler(self, grid_spec: GridSpec) -> Resampler:
        """Get or create a Resampler for the given grid.

        Args:
            grid_spec: Grid specification defining the resampling domain

        Returns:
            Configured Resampler instance ready for use
        """
        raise NotImplementedError()

class TimeResampler(Protocol):
    """Protocol for resampling data to uniform time intervals.

    Handles resampling of arrays (categorical or continuous) to target
    time intervals, maintaining data integrity.
    """

    def resample_to_time(
        self, arr: NDArray[np.int64], is_categorical: bool, target_dt: float
    ) -> tuple[NDArray[np.int64], float]:
        """Resample array to target time interval.

        Args:
            arr: Input array to resample
            is_categorical: If True, array contains categorical data (preserve values)
            target_dt: Target time interval for resampling

        Returns:
            Tuple of (resampled_array, actual_dt_achieved)
        """
        raise NotImplementedError()

# ============================================================================
# Cache Protocols
# ============================================================================
# Protocols for caching and loading seismic analysis data.

class CacheLoaderProtocol(Protocol):
    """Protocol for loading cache files from disk.

    Defines methods for selecting and loading cache files, supporting
    both individual file selection and full stack loading.
    """

    def select_cache_file(
        self, cache_dir: str | PathLike[str], domain: str
    ) -> str | None:
        """Select a cache file from the cache directory.

        Args:
            cache_dir: Directory containing cache files
            domain: Domain identifier (e.g., 'depth', 'time')

        Returns:
            Path to selected cache file, or None if no suitable file found
        """
        raise NotImplementedError()

    def load_full_stack(
        self, filename: str | PathLike[str]
    ) -> NDArray[np.float64] | None:
        """Load a full stack array from a cache file.

        Args:
            filename: Path to the cache file

        Returns:
            Loaded array or None if file cannot be loaded
        """
        raise NotImplementedError()

class CacheProtocol(Protocol, Generic[T]):
    """Protocol for generic caching implementations.

    Defines a minimal cache interface supporting get/set operations,
    key enumeration, clearing, and introspection.

    This protocol uses duck-typing to work with any cache implementation
    that provides the required methods, without inheritance requirements.
    """

    def get(self, key: str) -> T | None:
        """Retrieve a value from cache.

        Args:
            key: Cache key to look up

        Returns:
            Cached value or None if key not found
        """
        raise NotImplementedError()

    def set(self, key: str, value: T) -> None:
        """Store a value in cache.

        Args:
            key: Cache key to store under
            value: Value to cache
        """
        raise NotImplementedError()

    def keys(self) -> list[str]:
        """Get all keys currently in cache.

        Returns:
            List of all cache keys
        """
        raise NotImplementedError()

    def clear(self) -> None:
        """Clear all entries from cache."""
        raise NotImplementedError()

    def info(self) -> dict[str, Any]:
        """Get cache metadata and statistics.

        Returns:
            Dictionary with cache info (size, hits, misses, etc.)
        """
        raise NotImplementedError()

# ============================================================================
# Factory & Strategy Protocols
# ============================================================================
# Protocols for creating objects and implementing custom strategies.

class SelectorProtocol(Protocol):
    """Protocol for cache file selection strategies.

    A callable that selects appropriate cache files based on domain
    and cache directory, implementing custom selection logic.
    """

    def __call__(self, cache_dir: str, domain: str) -> str | None:
        """Select a cache file for the given domain.

        Args:
            cache_dir: Directory to search for cache files
            domain: Domain identifier to filter files

        Returns:
            Selected file path or None if no suitable file found
        """
        raise NotImplementedError()

class ArchiveExtractorProtocol(Protocol):
    """Protocol for extracting data from archive files.

    A callable that handles extraction of data from NPZ archive files,
    enabling flexible archive handling with type-safe return values.
    """

    def __call__(self, archive: NpzFile) -> NDArray[np.float64] | None:
        """Extract data from an NPZ archive.

        Args:
            archive: NPZ file object from np.load()

        Returns:
            Extracted data array or None if extraction fails
        """
        raise NotImplementedError()

class DatasetManagerFactory(Protocol):
    """Factory for creating DatasetManager instances.

    Creates configured DatasetManager instances that can load and manage
    datasets for analysis workflows.
    """

    def create(
        self, data_path: str, file_map: dict[str, str], grid_spec: GridSpec
    ) -> DatasetManager:
        """Create a DatasetManager with the given configuration.

        Args:
            data_path: Root path to dataset files
            file_map: Mapping of data keys to file paths
            grid_spec: Grid specification for the dataset

        Returns:
            Configured DatasetManager instance ready for use
        """
        raise NotImplementedError()

class PlotterProtocol(Protocol):
    """Protocol for creating analysis result visualizations.

    Defines interface for creating summary plots from AVO analysis results,
    supporting both depth and time domain visualizations.
    """

    def create_summary_plots(
        self, avo_results: "AvoResults", cache_dir: str, domain: str
    ) -> Figure:
        """Create summary plots from AVO analysis results.

        Args:
            avo_results: AVO analysis results to visualize
            cache_dir: Directory for caching intermediate plot data
            domain: Domain for visualization ('depth' or 'time')

        Returns:
            Matplotlib Figure containing all summary plots
        """
        raise NotImplementedError()
