"""Configuration classes for processor operations."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__ = [
    "PadConfig"
    "DilationConfig"
    "NeighborDirection"
    "ValidationResult"
    "ProcessorConfig"
    "BoundaryComputationConfig"
]


class PadConfig(TypedDict):
    """Type definition for padding configuration dictionaries.

    Used by numpy.pad() function for array padding operations.
    """

    pad_width: tuple[tuple[int, int], ...] | tuple[int, int] | int
    """Width of padding for each axis."""

    mode: str
    """Padding mode ('edge', 'constant', 'reflect', etc.)."""


class DilationConfig(TypedDict):
    """Type definition for binary dilation configuration dictionaries.

    Used by scipy.ndimage.binary_dilation() function.
    """

    iterations: int
    """Number of dilation iterations."""


class NeighborDirection(Enum):
    """Enum for 4-connected neighbor directions in boundary detection.

    Provides type-safe access to slice indices for neighbor voxel comparison.
    All indices are relative to the center after padding.
    """

    UP = slice(0, -2)
    """Slice for upper neighbor (same depth slice, row-1)."""

    DOWN = slice(2, None)
    """Slice for lower neighbor (same depth slice, row+1)."""

    LEFT = slice(0, -2)
    """Slice for left neighbor (same depth slice, col-1)."""

    RIGHT = slice(2, None)
    """Slice for right neighbor (same depth slice, col+1)."""

    CENTER = slice(1, -1)
    """Slice for center voxels (after padding, selects original data)."""

    @classmethod
    def all_directions(cls) -> list["NeighborDirection"]:
        """Get all neighbor directions except center."""
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]


@dataclass(frozen=True)
class ValidationResult:
    """Result of array validation operations.

    Provides structured validation results with detailed information about
    what was validated and how many elements were removed. Frozen to ensure
    immutability and hashability for use in caching and comparisons.

    Attributes
    ----------
    is_valid : bool
        Whether validation passed (all checks successful).
    arr1 : Optional[NDArray[np.float64]]
        First validated/filtered array, or None if validation failed.
    arr2 : Optional[NDArray[np.float64]]
        Second validated/filtered array, or None if validation failed.
    n_removed : int
        Number of elements removed during filtering (0 if none removed).
    error_message : str
        Error message if validation failed, empty string if successful.
    """

    is_valid: bool
    arr1: Optional["NDArray[np.float64]"] = None
    arr2: Optional["NDArray[np.float64]"] = None
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
    """Padding mode for boundary detection (edge replicates boundary values)."""

    # Amplitude window settings
    amplitude_window_radius: int = 2
    """Radius of amplitude window around transition points (samples on each side)."""

    # Statistical computation settings
    percentile_q1: int = 25
    """First quartile percentile for amplitude statistics."""

    percentile_q3: int = 75
    """Third quartile percentile for amplitude statistics."""

    # Numerical stability
    separation_matrix_epsilon: float = 1e-10
    """Epsilon value for numerical stability in division operations."""

    min_valid_samples: int = 2
    """Minimum number of valid samples required for statistical/correlation computations."""

    # Performance monitoring thresholds (milliseconds) - logged as warnings if exceeded
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
        """Total amplitude window size (must be odd for symmetric windows)."""
        return 2 * self.amplitude_window_radius + 1

    @property
    def boundary_pad_config(self) -> PadConfig:
        """Standard padding configuration for 3D boundary detection (pad j and k axes)."""
        return {"pad_width": ((0, 0), (1, 1), (1, 1)), "mode": self.pad_mode}

    @property
    def amplitude_pad_config(self) -> PadConfig:
        """Get amplitude padding configuration based on current window radius."""
        return {
            "pad_width": (self.amplitude_window_radius, self.amplitude_window_radius),
            "mode": self.pad_mode,
        }

    def __str__(self) -> str:
        """Return human-readable string representation of processor configuration."""
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

    Consolidates boundary-specific parameters to improve maintainability and
    allow easy configuration adjustments for boundary detection algorithms.
    """

    # Dilation configuration
    dilation_iterations: int = 2
    """Default number of iterations for binary dilation of boundaries."""

    # Edge handling configuration
    pad_mode: str = "edge"
    """Padding mode: 'edge' replicates boundary values for consistent edge comparison."""

    # Connectivity type
    connectivity: str = "4-connected"
    """Connectivity type: 4-connected within same depth slice (2D in-plane)."""

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
            Configuration dict with 'iterations' key for binary_dilation().
        """
        if iterations is None:
            iterations = self.dilation_iterations
        return {"iterations": iterations}

    def __str__(self) -> str:
        """Return human-readable string representation of boundary configuration."""
        lines = [
            "BoundaryComputationConfig:"
            f"  connectivity: {self.connectivity}"
            f"  dilation_iterations: {self.dilation_iterations}"
            f"  pad_mode: '{self.pad_mode}'"
            f"  pad_width: {self.pad_config['pad_width']}"
        ]
        return "\n".join(lines)
