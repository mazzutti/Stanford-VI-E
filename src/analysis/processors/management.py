"""Unified processor management: registry, configuration, and utilities.

This module consolidates all processor lifecycle management concerns:
- Registry Pattern for processor discovery and instantiation
- Configuration classes for processor operations
- Utility functions for numerical operations and statistics

By merging these concerns, we improve:
- Code cohesion (all processor lifecycle logic in one place)
- Reduced module interdependencies
- Clearer responsibility boundaries
- Easier testing and configuration

Pattern: Registry Pattern + Factory Pattern + Strategy Pattern
- Central registry for processor creation (ProcessorRegistry)
- Type-safe configuration with immutable dataclasses
- Pluggable statistics strategies for numerical operations
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    TypedDict,
    cast,
    TYPE_CHECKING,
)
from dataclasses import dataclass, field
import logging

import numpy as np
from numpy.typing import NDArray

from src.analysis.models import FaciesStats
from src.analysis.strategies import (
    ArrayStatisticsStrategy,
    StandardArrayStatistics,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray as NPArray
    from .boundary import CubeAligner

logger = logging.getLogger(__name__)

__all__ = [
    # Registry
    "ProcessorMetadata",
    "ProcessorRegistry",
    "get_default_processor_registry",
    "register_processor",
    "create_processor",
    # Configuration
    "PadConfig",
    "DilationConfig",
    "ValidationResult",
    "ProcessorConfig",
    "BoundaryComputationConfig",
    # Utilities (as module-level functions)
    "convert_numpy_scalars_to_float",
    "compute_quartiles",
    "filter_finite_values",
    "flatten_and_filter_finite",
    "reshape_3d_to_2d",
    "align_and_reshape",
    "compute_vertical_gradient",
    "extract_amplitude_subset",
    "compute_amplitude_stats",
]

T = TypeVar("T")  # Generic processor type

# ============================================================================
# REGISTRY SECTION
# ============================================================================


@dataclass
class ProcessorMetadata:
    """Metadata about a registered processor.

    Attributes
    ----------
    name : str
        Unique identifier for the processor.
    domain : str
        Domain/category this processor belongs to (e.g., 'facies', 'avo').
    version : str
        Version of the processor (e.g., '1.0', '2.1.0').
    tags : List[str]
        Keywords describing processor capabilities.
    description : str
        Human-readable description of what processor does.
    dependencies : List[str]
        Names of other processors this one depends on.
    """

    name: str
    domain: str = "default"
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    dependencies: List[str] = field(default_factory=list)

    def matches_tags(self, required_tags: List[str]) -> bool:
        """Check if processor has all required tags."""
        return all(tag in self.tags for tag in required_tags)


class ProcessorRegistry:
    """Central registry for processor creation and management."""

    def __init__(self) -> None:
        """Initialize an empty processor registry."""
        self._processors: Dict[str, Callable[[], Any]] = {}
        self._metadata: Dict[str, ProcessorMetadata] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        domain: str = "default",
        version: str = "1.0",
        tags: Optional[List[str]] = None,
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a processor factory."""
        if not callable(factory):
            raise TypeError(f"factory must be callable, got {type(factory)}")

        if name in self._processors:
            raise ValueError(f"Processor '{name}' already registered")

        self._processors[name] = factory
        self._metadata[name] = ProcessorMetadata(
            name=name,
            domain=domain,
            version=version,
            tags=tags or [],
            description=description,
            dependencies=dependencies or [],
        )
        logger.debug(f"Registered processor: {name} ({domain}/{version})")

    def unregister(self, name: str) -> bool:
        """Unregister a processor.

        Parameters
        ----------
        name : str
            Name of processor to unregister.

        Returns
        -------
        bool
            True if processor was registered and removed, False otherwise.
        """
        if name in self._processors:
            del self._processors[name]
            del self._metadata[name]
            logger.debug(f"Unregistered processor: {name}")
            return True
        return False

    def create(self, name: str) -> Any:
        """Create a processor instance by name.

        Parameters
        ----------
        name : str
            Name of processor to create.

        Returns
        -------
        Any
            New instance of the processor.

        Raises
        ------
        ValueError
            If processor name not found in registry.
        """
        if name not in self._processors:
            available = list(self._processors.keys())
            raise ValueError(f"Unknown processor '{name}'. Available: {available}")

        try:
            instance = self._processors[name]()
            logger.debug(f"Created processor instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create processor '{name}': {e}")
            raise

    def create_all(self, names: List[str]) -> Dict[str, Any]:
        """Create multiple processor instances.

        Parameters
        ----------
        names : List[str]
            Names of processors to create.

        Returns
        -------
        Dict[str, Any]
            Mapping of processor names to instances.

        Raises
        ------
        ValueError
            If any processor name not found.
        """
        return {name: self.create(name) for name in names}

    def list_processors(
        self,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> List[str]:
        """List registered processor names with optional filtering."""
        results = []
        for name, meta in self._metadata.items():
            if domain and meta.domain != domain:
                continue
            if tags and not meta.matches_tags(tags):
                continue
            if version and meta.version != version:
                continue
            results.append(name)
        return results

    def get_metadata(self, name: str) -> ProcessorMetadata:
        """Get metadata about a processor.

        Parameters
        ----------
        name : str
            Name of processor.

        Returns
        -------
        ProcessorMetadata
            Metadata describing the processor.

        Raises
        ------
        ValueError
            If processor not found.
        """
        if name not in self._metadata:
            raise ValueError(f"Unknown processor: {name}")
        return self._metadata[name]

    def has(self, name: str) -> bool:
        """Check if a processor is registered."""
        return name in self._processors

    def get_all_metadata(self) -> Dict[str, ProcessorMetadata]:
        """Get metadata for all registered processors."""
        return dict(self._metadata)

    def __repr__(self) -> str:
        """Return string representation showing registry state."""
        count = len(self._processors)
        domains = set(m.domain for m in self._metadata.values())
        return f"ProcessorRegistry({count} processors in domains: {domains})"


# Global default processor registry
_default_registry: Optional[ProcessorRegistry] = None


def get_default_processor_registry() -> ProcessorRegistry:
    """Get or create the global default processor registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ProcessorRegistry()
    return _default_registry


def register_processor(
    name: str,
    factory: Callable[[], Any],
    *,
    domain: str = "default",
    **kwargs: Any,
) -> None:
    """Register a processor in the default registry.

    Convenience function for registering processors globally.

    Parameters
    ----------
    name : str
        Unique processor identifier.
    factory : Callable[[], Any]
        Callable that creates processor instances.
    domain : str, optional
        Processor domain (default: "default").
    **kwargs
        Additional metadata (version, tags, description, dependencies).
    """
    get_default_processor_registry().register(name, factory, domain=domain, **kwargs)


def create_processor(name: str) -> Any:
    """Create a processor from the default registry.

    Convenience function for creating processors.

    Parameters
    ----------
    name : str
        Name of processor to create.

    Returns
    -------
    Any
        New processor instance.

    Raises
    ------
    ValueError
        If processor not found.
    """
    return get_default_processor_registry().create(name)


# ============================================================================
# CONFIGURATION SECTION
# ============================================================================


class PadConfig(TypedDict):  # type: ignore
    """Type definition for padding configuration dictionaries.

    Used by numpy.pad() function for array padding operations.
    """

    pad_width: tuple[tuple[int, int], ...] | tuple[int, int] | int
    """Width of padding for each axis."""

    mode: str
    """Padding mode ('edge', 'constant', 'reflect', etc.)."""


class DilationConfig(TypedDict):  # type: ignore
    """Type definition for binary dilation configuration dictionaries.

    Used by scipy.ndimage.binary_dilation() function.
    """

    iterations: int
    """Number of dilation iterations."""


@dataclass(frozen=True)
class ValidationResult:
    """Result of array validation operations.

    Frozen to ensure immutability and hashability for caching.

    Attributes
    ----------
    is_valid : bool
        Whether validation passed.
    arr1 : Optional[NDArray[np.float64]]
        First validated/filtered array.
    arr2 : Optional[NDArray[np.float64]]
        Second validated/filtered array.
    n_removed : int
        Number of elements removed during filtering.
    error_message : str
        Error message if validation failed.
    """

    is_valid: bool
    arr1: Optional[NDArray[np.float64]] = None
    arr2: Optional[NDArray[np.float64]] = None
    n_removed: int = 0
    error_message: str = ""


@dataclass(frozen=True)
class ProcessorConfig:
    """Immutable configuration for processor operations.

    Consolidates all processor configuration parameters into a single
    type-safe, immutable object. Using frozen=True ensures configuration
    cannot be accidentally modified after creation.
    """

    # Boundary detection settings
    boundary_dilation_default: int = 2
    """Default dilation radius for boundary zone expansion."""

    pad_mode: str = "edge"
    """Padding mode for boundary detection."""

    # Amplitude window settings
    amplitude_window_radius: int = 2
    """Radius of amplitude window around transition points."""

    # Statistical computation settings
    percentile_q1: int = 25
    """First quartile percentile for amplitude statistics."""

    percentile_q3: int = 75
    """Third quartile percentile for amplitude statistics."""

    # Numerical stability
    separation_matrix_epsilon: float = 1e-10
    """Epsilon value for numerical stability in division operations."""

    min_valid_samples: int = 2
    """Minimum number of valid samples required for statistical computations."""

    # Performance monitoring thresholds (milliseconds)
    boundary_detection_threshold_ms: float = 100.0
    """Timing threshold for boundary detection operations."""

    cube_alignment_threshold_ms: float = 50.0
    """Timing threshold for cube alignment operations."""

    boundary_amplitude_extraction_threshold_ms: float = 100.0
    """Timing threshold for boundary amplitude extraction operations."""

    gradient_correlation_threshold_ms: float = 100.0
    """Timing threshold for gradient correlation calculations."""

    interface_reflection_analysis_threshold_ms: float = 150.0
    """Timing threshold for interface reflection analysis operations."""

    facies_discrimination_threshold_ms: float = 80.0
    """Timing threshold for facies discrimination calculations."""

    @property
    def amplitude_window_size(self) -> int:
        """Total amplitude window size."""
        return 2 * self.amplitude_window_radius + 1

    @property
    def boundary_pad_config(self) -> PadConfig:
        """Standard padding configuration for 3D boundary detection."""
        return {"pad_width": ((0, 0), (1, 1), (1, 1)), "mode": self.pad_mode}

    @property
    def amplitude_pad_config(self) -> PadConfig:
        """Get amplitude padding configuration."""
        return {
            "pad_width": (self.amplitude_window_radius, self.amplitude_window_radius),
            "mode": self.pad_mode,
        }

    def __str__(self) -> str:
        """Return human-readable string representation."""
        lines = [
            "ProcessorConfig:"
            f"  boundary_dilation: {self.boundary_dilation_default}"
            f"  amplitude_window_radius: {self.amplitude_window_radius}"
            f"  amplitude_window_size: {self.amplitude_window_size}"
            f"  q1_percentile: {self.percentile_q1}%"
            f"  q3_percentile: {self.percentile_q3}%"
            f"  epsilon: {self.separation_matrix_epsilon:.2e}"
            f"  min_valid_samples: {self.min_valid_samples}"
            f"  pad_mode: '{self.pad_mode}'"
            "  Timing thresholds (ms):"
            f"    boundary_detection: {self.boundary_detection_threshold_ms}"
            f"    cube_alignment: {self.cube_alignment_threshold_ms}"
            f"    boundary_amplitude_extraction: {self.boundary_amplitude_extraction_threshold_ms}"
            f"    gradient_correlation: {self.gradient_correlation_threshold_ms}"
            f"    interface_reflection_analysis: {self.interface_reflection_analysis_threshold_ms}"
            f"    facies_discrimination: {self.facies_discrimination_threshold_ms}"
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BoundaryComputationConfig:
    """Immutable configuration for boundary detection and dilation operations.

    Consolidates boundary-specific parameters for boundary detection algorithms.
    """

    dilation_iterations: int = 2
    """Default number of iterations for binary dilation of boundaries."""

    pad_mode: str = "edge"
    """Padding mode: 'edge' replicates boundary values."""

    connectivity: str = "4-connected"
    """Connectivity type: 4-connected within same depth slice."""

    @property
    def pad_config(self) -> PadConfig:
        """Standard padding: no padding on i-axis, 1 pixel on j and k axes."""
        return {"pad_width": ((0, 0), (1, 1), (1, 1)), "mode": self.pad_mode}

    def get_dilation_config(self, iterations: Optional[int] = None) -> DilationConfig:
        """Get binary dilation configuration with specified iterations.

        Parameters
        ----------
        iterations : int, optional
            Number of dilation iterations. Uses default if None.

        Returns
        -------
        DilationConfig
            Configuration dict for binary_dilation().
        """
        if iterations is None:
            iterations = self.dilation_iterations
        return {"iterations": iterations}

    def __str__(self) -> str:
        """Return human-readable string representation."""
        lines = [
            "BoundaryComputationConfig:"
            f"  connectivity: {self.connectivity}"
            f"  dilation_iterations: {self.dilation_iterations}"
            f"  pad_mode: '{self.pad_mode}'"
            f"  pad_width: {self.pad_config['pad_width']}"
        ]
        return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS SECTION
# ============================================================================

# Statistics strategy support (simplified, no abstraction overhead)
_default_strategy: ArrayStatisticsStrategy = StandardArrayStatistics()


def set_default_statistics_strategy(strategy: ArrayStatisticsStrategy) -> None:
    """Set default statistics strategy for all operations.

    Parameters
    ----------
    strategy
        New default strategy to use.
    """
    global _default_strategy
    _default_strategy = strategy


def get_default_statistics_strategy() -> ArrayStatisticsStrategy:
    """Get current default statistics strategy.

    Returns
    -------
    ArrayStatisticsStrategy
        Current default strategy.
    """
    return _default_strategy


def convert_numpy_scalars_to_float(
    *values: NDArray[np.floating[Any]] | np.floating[Any],
) -> Tuple[float, ...] | float:
    """Efficiently convert one or more NumPy scalars/arrays to Python floats.

    Parameters
    ----------
    *values : numpy.ndarray | numpy.floating
        One or more NumPy scalars or arrays to convert.

    Returns
    -------
    float | tuple of float
        Converted Python float values. Single float if one argument,
        tuple of floats if multiple arguments.
    """
    if len(values) == 1:
        val = values[0]
        return float(val.item() if hasattr(val, "item") else val)
    return tuple(float(v.item() if hasattr(v, "item") else v) for v in values)


def compute_quartiles(amps: NDArray[np.float64]) -> Tuple[float, float]:
    """Efficiently compute Q1 and Q3 percentiles from amplitude array.

    Parameters
    ----------
    amps : numpy.ndarray(dtype=float64)
        Array of amplitude values.

    Returns
    -------
    tuple of float
        (q1, q3) quartile values as Python floats.
    """
    percentiles = np.percentile(amps, [25, 75])
    result = convert_numpy_scalars_to_float(*percentiles)
    q1, q3 = cast(Tuple[float, float], result)
    return q1, q3


def filter_finite_values(
    arr1: NDArray[np.float64], arr2: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Filter out NaN and Inf values from paired arrays.

    Parameters
    ----------
    arr1 : numpy.ndarray(dtype=float64)
        First input array to filter.
    arr2 : numpy.ndarray(dtype=float64)
        Second input array to filter.

    Returns
    -------
    tuple
        (filtered_arr1, filtered_arr2, n_removed) where n_removed is count
        of elements removed due to non-finite values.
    """
    valid_mask = np.isfinite(arr1) & np.isfinite(arr2)
    n_removed = int((~valid_mask).sum())

    if n_removed > 0:
        logger.debug(
            "Filtered %d non-finite values from arrays (original size: %d)",
            n_removed,
            len(arr1),
        )

    return arr1[valid_mask], arr2[valid_mask], n_removed


def flatten_and_filter_finite(
    arr: NDArray[np.float64], bool_mask: NDArray[np.bool_]
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """Flatten and filter finite values from array and mask."""
    arr_flat = arr.flatten()
    mask_flat = bool_mask.flatten().astype(float)
    arr_filtered, mask_filtered, _ = filter_finite_values(arr_flat, mask_flat)
    return arr_filtered, mask_filtered


def reshape_3d_to_2d(
    seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Reshape 3D cubes to 2D trace-sample format (n_traces, nk).

    Parameters
    ----------
    seismic_cube : numpy.ndarray(dtype=float64)
        3D seismic cube with shape (ni, nj, nk).
    facies_cube : numpy.ndarray(dtype=int64)
        3D facies cube with shape (ni, nj, nk).

    Returns
    -------
    tuple
        (seismic_2d, facies_2d) both with shape (n_traces, nk).

    Raises
    ------
    ValueError
        If reshape fails due to shape mismatch.
    """
    ni, nj, nk = facies_cube.shape
    n_traces = ni * nj

    try:
        seismic_2d = seismic_cube.reshape(n_traces, nk)
        facies_2d = facies_cube.reshape(n_traces, nk).astype(int, copy=False)
    except ValueError as e:
        raise ValueError(
            f"Failed to reshape cubes to (n_traces={n_traces}, nk={nk}): {e}"
        )

    return seismic_2d, facies_2d


def align_and_reshape(
    aligner: CubeAligner,
    seismic_cube: NDArray[np.float64],
    facies_cube: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Align 3D cubes and reshape to 2D trace-sample format.

    Parameters
    ----------
    aligner : CubeAligner
        Alignment operator initialized with target dimensions.
    seismic_cube : numpy.ndarray(dtype=float64)
        3D seismic amplitude cube.
    facies_cube : numpy.ndarray(dtype=int64)
        3D facies classification cube.

    Returns
    -------
    tuple
        (seismic_2d, facies_2d) with shape (n_traces, nk).

    Raises
    ------
    ValueError
        If alignment or reshape fails.
    """
    seismic_aligned = aligner.align(seismic_cube)
    facies_aligned = aligner.align(facies_cube)
    return reshape_3d_to_2d(seismic_aligned, facies_aligned)


def compute_vertical_gradient(
    seismic_cube: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute vertical (k-axis) gradient of seismic cube with absolute values.

    Parameters
    ----------
    seismic_cube : numpy.ndarray(dtype=float64)
        3D seismic amplitude cube with shape (i, j, k).

    Returns
    -------
    numpy.ndarray(dtype=float64)
        Absolute vertical gradient array with same shape as input.

    Notes
    -----
    Uses numpy.gradient() with axis=2 (vertical k-axis) for stable
    finite-difference approximation.
    """
    seismic_grad = np.gradient(seismic_cube, axis=2)
    seismic_grad_abs = np.abs(seismic_grad)
    logger.debug(
        "Computed vertical gradient: range=[%.4e, %.4e]",
        np.min(seismic_grad_abs),
        np.max(seismic_grad_abs),
    )
    return cast(NDArray[np.float64], seismic_grad_abs)


def extract_amplitude_subset(
    data: NDArray[np.float64], mask: NDArray[np.bool_], mask_value: bool = True
) -> NDArray[np.float64]:
    """Extract amplitude values where mask has specified value.

    Parameters
    ----------
    data : numpy.ndarray(dtype=float64)
        Amplitude data array.
    mask : numpy.ndarray(dtype=bool)
        Boolean mask array (same shape as data).
    mask_value : bool, optional
        Value to match in mask (True or False). Default is True.

    Returns
    -------
    numpy.ndarray(dtype=float64)
        Extracted amplitude values where mask matches mask_value.
    """
    if mask_value:
        return data[mask]
    else:
        return data[~mask]


def compute_amplitude_stats(amps: NDArray[np.float64]) -> FaciesStats:
    """Compute statistical summary of amplitude array.

    Parameters
    ----------
    amps : numpy.ndarray(dtype=float64)
        Array of amplitude values.

    Returns
    -------
    FaciesStats
        Statistical summary. Empty FaciesStats if array is empty.
    """
    if amps.size == 0:
        logger.debug("Computing stats for empty amplitude array")
        return FaciesStats()

    # Compute all statistics with single conversion call
    result = convert_numpy_scalars_to_float(
        np.mean(amps), np.std(amps), np.median(amps), np.min(amps), np.max(amps)
    )
    mean_val, std_val, median_val, min_val, max_val = cast(
        Tuple[float, float, float, float, float], result
    )

    # Compute quartiles using helper function
    q1, q3 = compute_quartiles(amps)

    stats = FaciesStats(
        count=len(amps),
        mean=mean_val,
        std=std_val,
        median=median_val,
        q25=q1,
        q75=q3,
        min=min_val,
        max=max_val,
    )

    logger.debug(
        "Computed stats from %d samples: mean=%.4f, std=%.4f",
        stats.count,
        stats.mean,
        stats.std,
    )

    return stats
