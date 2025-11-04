"""Abstract base classes for computational components.

This module provides core abstractions for building analysis systems:
- Computer, AnalysisSchema: Computational component abstractions
- Protocols: Type contracts for resampling, caching, factories, and visualization

For analyzer abstractions, see analyzer.py instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    TYPE_CHECKING,
)

import numpy as np
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
from matplotlib.figure import Figure

from src.io.grid import GridSpec
from src.io.data_loader import DatasetManager

if TYPE_CHECKING:
    from src.analysis.models import AvoResults

__all__ = [
    "Computer",
    "AnalysisSchema",
    "ComputationResult",
    # Type variables
    "T",
    # Resampling protocols
    "ResamplePlan",
    "Resampler",
    "ResamplerFactory",
    "TimeResampler",
    # Cache protocols
    "CacheLoaderProtocol",
    "CacheProtocol",
    "SelectorProtocol",
    "ArchiveExtractorProtocol",
    # Factory protocols
    "DatasetManagerFactory",
    "PlotterProtocol",
]

# Type variables for generic constraints
T_In = TypeVar("T_In")  # Input type
T_Out = TypeVar("T_Out")  # Output type
T = TypeVar("T")  # Generic type


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
        ...


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
        ...


class TimeResampler(Protocol):
    """Protocol for resampling data to uniform time intervals.

    Handles resampling of arrays (categorical or continuous) to target
    time intervals, maintaining data integrity.
    """

    def resample_to_time(
        self, arr: NDArray[np.int64], is_categorical: bool, target_dt: float
    ) -> Tuple[NDArray[np.int64], float]:
        """Resample array to target time interval.

        Args:
            arr: Input array to resample
            is_categorical: If True, array contains categorical data (preserve values)
            target_dt: Target time interval for resampling

        Returns:
            Tuple of (resampled_array, actual_dt_achieved)
        """
        ...


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
        self, cache_dir: Union[str, PathLike[str]], domain: str
    ) -> Optional[str]:
        """Select a cache file from the cache directory.

        Args:
            cache_dir: Directory containing cache files
            domain: Domain identifier (e.g., 'depth', 'time')

        Returns:
            Path to selected cache file, or None if no suitable file found
        """
        ...

    def load_full_stack(
        self, filename: Union[str, PathLike[str]]
    ) -> Optional[NDArray[np.float64]]:
        """Load a full stack array from a cache file.

        Args:
            filename: Path to the cache file

        Returns:
            Loaded array or None if file cannot be loaded
        """
        ...


class CacheProtocol(Protocol, Generic[T]):
    """Protocol for generic caching implementations.

    Defines a minimal cache interface supporting get/set operations,
    key enumeration, clearing, and introspection.

    This protocol uses duck-typing to work with any cache implementation
    that provides the required methods, without inheritance requirements.
    """

    def get(self, key: str) -> Optional[T]:
        """Retrieve a value from cache.

        Args:
            key: Cache key to look up

        Returns:
            Cached value or None if key not found
        """
        ...

    def set(self, key: str, value: T) -> None:
        """Store a value in cache.

        Args:
            key: Cache key to store under
            value: Value to cache
        """
        ...

    def keys(self) -> List[str]:
        """Get all keys currently in cache.

        Returns:
            List of all cache keys
        """
        ...

    def clear(self) -> None:
        """Clear all entries from cache."""
        ...

    def info(self) -> Dict[str, Any]:
        """Get cache metadata and statistics.

        Returns:
            Dictionary with cache info (size, hits, misses, etc.)
        """
        ...


# ============================================================================
# Factory & Strategy Protocols
# ============================================================================
# Protocols for creating objects and implementing custom strategies.


class SelectorProtocol(Protocol):
    """Protocol for cache file selection strategies.

    A callable that selects appropriate cache files based on domain
    and cache directory, implementing custom selection logic.
    """

    def __call__(self, cache_dir: str, domain: str) -> Optional[str]:
        """Select a cache file for the given domain.

        Args:
            cache_dir: Directory to search for cache files
            domain: Domain identifier to filter files

        Returns:
            Selected file path or None if no suitable file found
        """
        ...


class ArchiveExtractorProtocol(Protocol):
    """Protocol for extracting data from archive files.

    A callable that handles extraction of data from NPZ archive files,
    enabling flexible archive handling with type-safe return values.
    """

    def __call__(self, archive: NpzFile) -> Optional[NDArray[np.float64]]:
        """Extract data from an NPZ archive.

        Args:
            archive: NPZ file object from np.load()

        Returns:
            Extracted data array or None if extraction fails
        """
        ...


class DatasetManagerFactory(Protocol):
    """Factory for creating DatasetManager instances.

    Creates configured DatasetManager instances that can load and manage
    datasets for analysis workflows.
    """

    def create(
        self, data_path: str, file_map: Dict[str, str], grid_spec: GridSpec
    ) -> DatasetManager:
        """Create a DatasetManager with the given configuration.

        Args:
            data_path: Root path to dataset files
            file_map: Mapping of data keys to file paths
            grid_spec: Grid specification for the dataset

        Returns:
            Configured DatasetManager instance ready for use
        """
        ...


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
        ...


# ============================================================================
# Core Computational Abstractions
# ============================================================================


@dataclass
class ComputationResult(Generic[T_Out]):
    """Result wrapper for computation operations.

    Provides a structured way to return computation results with metadata
    about success/failure and any errors.
    """

    is_valid: bool
    """Whether computation succeeded."""

    data: Optional[T_Out] = None
    """Computed data (None if invalid)."""

    error_message: str = ""
    """Error message if computation failed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about computation (performance, validation details, etc)."""


@dataclass
class AnalysisSchema:
    """Describes the input/output contract for an analyzer.

    Provides self-documenting information about what data an analyzer
    expects and what it produces.
    """

    input_fields: Dict[str, Type]
    """Required input field names and their types."""

    output_fields: Dict[str, Type]
    """Output field names and their types."""

    description: str = ""
    """Human-readable description of the analysis."""

    constraints: Dict[str, str] = field(default_factory=dict)
    """Any constraints on the analysis (e.g., 'vp >= 1000 m/s')."""


class Computer(ABC, Generic[T_In, T_Out]):
    """Abstract base for domain-specific computational components.

    Computers encapsulate specific computational tasks that transform
    input data into output data. They provide consistent interfaces for:
    - Input validation
    - Computation execution
    - Schema documentation

    This abstraction enables polymorphic treatment of different computers
    and makes them easy to compose into larger analysis pipelines.

    Type Parameters
    ----------------
    T_In
        Type of input data accepted by this computer.
    T_Out
        Type of output data produced by this computer.

    Examples
    --------
    A concrete computer implementation:

    >>> class MyComputer(Computer[np.ndarray, Dict[str, np.ndarray]]):
    ...     def compute(self, data: np.ndarray) -> Dict[str, np.ndarray]:
    ...         # Do computation
    ...         return {"result": computed_array}
    ...
    ...     def validate(self, data: np.ndarray) -> bool:
    ...         return data.shape == expected_shape
    """

    @abstractmethod
    def validate(self, inputs: T_In) -> bool:
        """Validate that inputs are suitable for computation.

        Parameters
        ----------
        inputs
            Input data to validate.

        Returns
        -------
        bool
            True if inputs are valid, False otherwise.

        Notes
        -----
        Implementation should not modify inputs. If validation fails,
        the compute() method should raise an exception with details.
        """
        pass

    @abstractmethod
    def compute(self, inputs: T_In) -> T_Out:
        """Execute the computational task.

        Parameters
        ----------
        inputs
            Validated input data.

        Returns
        -------
        T_Out
            Computed output data.

        Raises
        ------
        ValueError
            If computation fails or inputs are invalid.
        """
        pass

    @abstractmethod
    def get_schema(self) -> AnalysisSchema:
        """Return schema describing this computer's inputs/outputs.

        Returns
        -------
        AnalysisSchema
            Self-documenting schema of computation contract.
        """
        pass
