"""Statistical analysis result models.

This module contains dataclasses for storing and analyzing statistical
results from various analysis techniques.

Validation: Uses validators from src.analysis.validators for improved
error handling and reduced code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..validators import RangeValidator, ValidationError
from .base import ModelUtilities, StatisticalResult, ValidationConfig
from .config import Transition
from .facies import FaciesStats
from .formatters import FormattableModel

# Type aliases for common patterns
TransitionStatsMap = dict[Transition, FaciesStats | None]
TransitionArrayMap = dict[Transition, NDArray[np.float64] | None]

__all__ = [
    "GradientCorrelationResult",
    "BoundaryAmpsResult",
    "FaciesDiscriminationResult",
    "InterfaceReflectionResult",
    "AvoAnalysisResult",
]


@dataclass(slots=True)
class GradientCorrelationResult(StatisticalResult, FormattableModel):
    """Represents gradient correlation analysis results with validation.

    Inherits FormattableModel for consistent __repr__/__str__ formatting.
    """

    pearson_correlation: float
    pearson_pvalue: float
    spearman_correlation: float
    spearman_pvalue: float
    seismic_gradient: NDArray[np.float64]
    boundaries: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate correlation values after initialization."""
        try:
            RangeValidator.validate_correlation(
                self.pearson_correlation, "pearson_correlation", allow_nan=True
            )
            RangeValidator.validate_correlation(
                self.spearman_correlation, "spearman_correlation", allow_nan=True
            )
            RangeValidator.validate_pvalue(
                self.pearson_pvalue, "pearson_pvalue", allow_nan=True
            )
            RangeValidator.validate_pvalue(
                self.spearman_pvalue, "spearman_pvalue", allow_nan=True
            )
        except ValidationError as e:
            raise ValidationError(f"Invalid gradient correlation result: {e}") from e

    def is_valid(self) -> bool:
        """Check if results are statistically significant."""
        return (
            self.pearson_pvalue < ValidationConfig.SIGNIFICANCE_THRESHOLD
            and self.spearman_pvalue < ValidationConfig.SIGNIFICANCE_THRESHOLD
            and len(self.boundaries) > 0
        )

    @cached_property
    def strongest_correlation(self) -> tuple[str, float]:
        """Get cached strongest correlation method and its value.

        Returns the correlation method with the highest absolute value,
        along with that correlation coefficient.
        """
        pearson_abs = abs(self.pearson_correlation)
        spearman_abs = abs(self.spearman_correlation)
        if pearson_abs >= spearman_abs:
            return ("Pearson", self.pearson_correlation)
        return ("Spearman", self.spearman_correlation)

    @property
    def boundary_count(self) -> int:
        """Return the number of identified boundaries."""
        return int(np.sum(self.boundaries))

    def get_stats_dict(self) -> dict[str, float]:
        """Return statistics dictionary for FormattableModel formatting.

        Provides statistics for consistent __repr__/__str__ formatting
        via FormattableModel.

        Returns:
            Dictionary mapping stat names to float values suitable for
            formatted display (strongest correlation, boundary count, validity).
        """
        _, strongest_value = self.strongest_correlation
        return {
            "strongest": float(strongest_value),
            "boundary_count": float(self.boundary_count),
            "pearson_corr": self.pearson_correlation,
            "spearman_corr": self.spearman_correlation,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - pearson_correlation: Pearson correlation coefficient (-1 to 1)
            - pearson_pvalue: Pearson test p-value (0 to 1)
            - spearman_correlation: Spearman correlation coefficient (-1 to 1)
            - spearman_pvalue: Spearman test p-value (0 to 1)
            - boundary_count: Number of identified boundaries
            - strongest_method: Name of strongest correlation method
            - strongest_value: Value of strongest correlation
            - valid: Whether result is statistically significant
        """
        strongest_method, strongest_value = self.strongest_correlation
        return {
            "pearson_correlation": self.pearson_correlation,
            "pearson_pvalue": self.pearson_pvalue,
            "spearman_correlation": self.spearman_correlation,
            "spearman_pvalue": self.spearman_pvalue,
            "boundary_count": self.boundary_count,
            "strongest_method": strongest_method,
            "strongest_value": strongest_value,
            "valid": self.is_valid(),
        }

    def summary(self) -> str:
        """Return a human-readable summary of the result.

        Returns:
            String containing:
            - strongest: Name of method with highest absolute correlation
            - r: Strongest correlation coefficient (rounded to 4 decimals)
            - boundaries: Number of identified boundaries
            - valid: Whether result is statistically significant
        """
        strongest_method, strongest_value = self.strongest_correlation
        return (
            f"GradientCorrelationResult(strongest={strongest_method} "
            f"r={strongest_value:.4f}, "
            f"boundaries={self.boundary_count}, "
            f"valid={self.is_valid()})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GradientCorrelationResult:
        """Create result from dictionary representation.

        Args:
            data: Dictionary with keys:
                - pearson_correlation: Pearson correlation coefficient
                - pearson_pvalue: Pearson test p-value
                - spearman_correlation: Spearman correlation coefficient
                - spearman_pvalue: Spearman test p-value
                - seismic_gradient: Gradient array (will be converted to NDArray)
                - boundaries: Boolean mask array (will be converted to NDArray)

        Returns:
            GradientCorrelationResult instance with data from dictionary.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If correlation/p-value ranges are invalid.
        """
        return cls(
            pearson_correlation=float(data["pearson_correlation"]),
            pearson_pvalue=float(data["pearson_pvalue"]),
            spearman_correlation=float(data["spearman_correlation"]),
            spearman_pvalue=float(data["spearman_pvalue"]),
            seismic_gradient=np.array(data["seismic_gradient"]),
            boundaries=np.array(data["boundaries"], dtype=np.bool_),
        )


@dataclass(slots=True)
class BoundaryAmpsResult(StatisticalResult, FormattableModel):
    """Represents amplitude measurements at and away from boundaries.

    Inherits FormattableModel for consistent __repr__/__str__ formatting.
    """

    at_boundaries: NDArray[np.float64]
    away_from_boundaries: NDArray[np.float64]
    boundary_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate array dimensions match.

        Ensures:
        - at_boundaries count matches the number of True values in boundary_mask
        - Combined at_boundaries and away_from_boundaries covers all mask positions
        """
        at_count = len(self.at_boundaries)
        away_count = len(self.away_from_boundaries)
        mask_true_count = int(np.sum(self.boundary_mask))
        total_mask_len = len(self.boundary_mask)

        if at_count != mask_true_count:
            raise ValueError(
                f"at_boundaries length ({at_count}) must match number of True values "
                f"in boundary_mask ({mask_true_count})"
            )
        if away_count + at_count != total_mask_len:
            raise ValueError(
                f"Combined at_boundaries ({at_count}) and away_from_boundaries "
                f"({away_count}) length ({at_count + away_count}) must match "
                f"boundary_mask length ({total_mask_len})"
            )

    def is_valid(self) -> bool:
        """Check if both amplitude arrays have sufficient data."""
        return len(self.at_boundaries) > 0 and len(self.away_from_boundaries) > 0

    @property
    def amplitude_difference(self) -> float:
        """Calculate mean amplitude difference between boundaries and away."""
        if not self.is_valid():
            return np.nan
        return float(np.mean(self.at_boundaries) - np.mean(self.away_from_boundaries))

    @cached_property
    def statistics(self) -> dict[str, float]:
        """Return cached statistical comparison between boundary and non-boundary amplitudes."""
        return {
            "at_boundaries_mean": float(np.mean(self.at_boundaries)),
            "at_boundaries_std": float(np.std(self.at_boundaries)),
            "away_from_boundaries_mean": float(np.mean(self.away_from_boundaries)),
            "away_from_boundaries_std": float(np.std(self.away_from_boundaries)),
            "difference": self.amplitude_difference,
        }

    def get_stats_dict(self) -> dict[str, float]:
        """Return statistics dictionary for FormattableModel formatting.

        Provides statistics for consistent __repr__/__str__ formatting
        via FormattableModel.

        Returns:
            Dictionary with key amplitude statistics.
        """
        return self.statistics

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - at_boundaries_count: Number of amplitude samples at boundaries
            - away_from_boundaries_count: Number of amplitude samples away from boundaries
            - at_boundaries_mean: Mean amplitude value at boundaries
            - at_boundaries_std: Standard deviation of amplitudes at boundaries
            - away_from_boundaries_mean: Mean amplitude value away from boundaries
            - away_from_boundaries_std: Standard deviation of amplitudes away from boundaries
            - difference: Mean difference between boundary and non-boundary amplitudes
        """
        return {
            "at_boundaries_count": len(self.at_boundaries),
            "away_from_boundaries_count": len(self.away_from_boundaries),
            **self.statistics,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the result.

        Returns:
            String containing:
            - at_boundary_mean: Mean amplitude at boundaries
            - away_mean: Mean amplitude away from boundaries
            - diff: Mean difference between the two (rounded to 4 decimals)
        """
        stats = self.statistics
        return (
            f"BoundaryAmpsResult(at_boundary_mean={stats['at_boundaries_mean']:.4f}, "
            f"away_mean={stats['away_from_boundaries_mean']:.4f}, "
            f"diff={stats['difference']:.4f})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundaryAmpsResult:
        """Create result from dictionary representation.

        Args:
            data: Dictionary with keys:
                - at_boundaries: Array of amplitude measurements at boundaries
                - away_from_boundaries: Array of amplitude measurements away from boundaries
                - (optional) at_boundaries_count: Expected count (for validation)
                - (optional) away_from_boundaries_count: Expected count (for validation)

        Returns:
            BoundaryAmpsResult instance with reconstructed boundary_mask.
            The boundary_mask is reconstructed with True for at_boundaries indices,
            False for away_from_boundaries indices.

        Raises:
            ValueError: If array dimensions don't match or are incompatible.
            TypeError: If array data cannot be converted to numpy arrays.
        """
        at_boundaries = np.array(data["at_boundaries"])
        away_from_boundaries = np.array(data["away_from_boundaries"])

        # Reconstruct boundary_mask: True for at_boundaries, False for away
        mask_length = len(at_boundaries) + len(away_from_boundaries)
        boundary_mask = np.zeros(mask_length, dtype=np.bool_)
        boundary_mask[: len(at_boundaries)] = True

        return cls(
            at_boundaries=at_boundaries,
            away_from_boundaries=away_from_boundaries,
            boundary_mask=boundary_mask,
        )


@dataclass(slots=True)
class FaciesDiscriminationResult(StatisticalResult, FormattableModel):
    """Represents facies discrimination analysis with computed properties.

    Inherits FormattableModel for consistent __repr__/__str__ formatting.
    """

    facies_stats: dict[int, FaciesStats]
    separation_matrix: NDArray[np.float64]
    facies_amplitudes: dict[int, NDArray[np.float64]] = field(
        default_factory=lambda: cast(dict[int, NDArray[np.float64]], {})
    )
    # Order of facies labels that index rows/columns of `separation_matrix`.
    label_order: list[int] = field(default_factory=lambda: cast(list[int], []))

    def __post_init__(self) -> None:
        """Validate consistency of facies data.

        Ensures:
        - label_order is initialized from facies_stats keys if not provided
        - facies_stats count matches label_order length
        - Separation matrix matches facies count
        """
        if not self.label_order:
            self.label_order = sorted(self.facies_stats.keys())

        stats_count = len(self.facies_stats)
        order_len = len(self.label_order)

        if stats_count != order_len:
            raise ValueError(
                f"Facies data inconsistency: facies_stats has {stats_count} facies "
                f"but label_order has {order_len} labels. These must match."
            )

        # Validate separation matrix dimensions if available
        matrix_size = self.separation_matrix.size
        if matrix_size > 0:
            if self.separation_matrix.shape != (stats_count, stats_count):
                raise ValueError(
                    f"Separation matrix shape {self.separation_matrix.shape} "
                    f"must be ({stats_count}, {stats_count}) to match facies count"
                )

    def is_valid(self) -> bool:
        """Check if discrimination result has sufficient data."""
        return (
            len(self.facies_stats) > 1
            and self.separation_matrix.size > 0
            and len(self.facies_amplitudes) == len(self.label_order)
        )

    @property
    def facies_count(self) -> int:
        """Return the number of facies classes."""
        return len(self.facies_stats)

    @cached_property
    def mean_separation(self) -> float:
        """Calculate and cache mean separation between facies in the matrix."""
        if not self.is_valid():
            return np.nan
        # Get upper triangle of separation matrix (avoid diagonal)
        upper_triangle = np.triu(self.separation_matrix, k=1)
        non_zero_values = upper_triangle[upper_triangle != 0]
        return float(np.mean(non_zero_values)) if len(non_zero_values) > 0 else np.nan

    @cached_property
    def best_separated_pair(self) -> tuple[int, int, float]:
        """Return the pair of facies with best separation and the separation value."""
        max_idx = np.unravel_index(
            np.argmax(self.separation_matrix), self.separation_matrix.shape
        )
        # np.unravel_index may return numpy integer types; cast to int for
        # indexing Python lists to satisfy static analyzers and guarantee
        # compatibility across Python/numpy versions.
        ia = int(max_idx[0])
        ib = int(max_idx[1])
        facies_a = self.label_order[ia]
        facies_b = self.label_order[ib]
        separation = float(self.separation_matrix[ia, ib])
        return facies_a, facies_b, separation

    def get_stats_dict(self) -> dict[str, float]:
        """Return statistics dictionary for FormattableModel formatting.

        Provides statistics for consistent __repr__/__str__ formatting
        via FormattableModel.

        Returns:
            Dictionary with facies count and separation metrics.
        """
        if not self.is_valid():
            return {"facies_count": float(self.facies_count), "mean_separation": np.nan}
        _, _, best_sep = self.best_separated_pair
        return {
            "facies_count": float(self.facies_count),
            "best_separation": best_sep,
            "mean_separation": self.mean_separation,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - facies_count: Number of facies classes
            - best_separated_pair: Tuple of (facies_a, facies_b) with highest separation
            - best_separation: Maximum separation value between any facies pair
            - mean_separation: Mean separation across all facies pairs
            - label_order: Ordered list of facies labels (indexes separation_matrix)
            - valid: Boolean indicating if result has sufficient data
        """
        if not self.is_valid():
            return {"valid": False}
        facies_a, facies_b, sep = self.best_separated_pair
        return {
            "facies_count": self.facies_count,
            "best_separated_pair": (facies_a, facies_b),
            "best_separation": sep,
            "mean_separation": self.mean_separation,
            "label_order": self.label_order,
            "valid": True,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the result.

        Returns:
            String containing:
            - facies: Total number of facies classes
            - best_pair: Tuple of (facies_a, facies_b) with highest separation
            - separation: Maximum separation value (rounded to 4 decimals)
            - mean_separation: Mean separation across all pairs (rounded to 4 decimals)

        Returns "invalid" string if result has insufficient data.
        """
        if not self.is_valid():
            return "FaciesDiscriminationResult(invalid)"
        facies_a, facies_b, sep = self.best_separated_pair
        return (
            f"FaciesDiscriminationResult(facies={self.facies_count}, "
            f"best_pair=({facies_a}, {facies_b}), "
            f"separation={sep:.4f}, "
            f"mean_separation={self.mean_separation:.4f})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaciesDiscriminationResult:
        """Create result from dictionary representation.

        Args:
            data: Dictionary with keys:
                - facies_stats: Dictionary mapping facies IDs to FaciesStats (reconstructed)
                - separation_matrix: 2D array of separation values
                - label_order: Ordered list of facies labels (defaults to facies_stats keys)

        Returns:
            FaciesDiscriminationResult instance with reconstructed data.

        Raises:
            ValueError: If facies_stats count doesn't match label_order length.
            TypeError: If arrays cannot be reconstructed from dictionary format.
        """
        facies_stats_dict = ModelUtilities.safe_get_dict(data, "facies_stats")
        facies_stats = {
            int(k): FaciesStats.from_dict(v) for k, v in facies_stats_dict.items()
        }
        return cls(
            facies_stats=facies_stats,
            separation_matrix=np.array(data["separation_matrix"]),
            label_order=list(data.get("label_order", facies_stats.keys())),
        )


@dataclass
class InterfaceReflectionResult(StatisticalResult, FormattableModel):
    """Structured result for interface reflection analysis.

    Inherits FormattableModel for consistent __repr__/__str__ formatting.

    Attributes
    ----------
    transitions_summary
        A mapping of transition keys (e.g. "0->1") to per-transition
        statistics (as :class:`FaciesStats`) or ``None`` when no samples exist.
    interface_stats
        Raw lists of amplitudes for each transition key
    """

    transitions_summary: TransitionStatsMap = field(
        default_factory=lambda: cast(TransitionStatsMap, {})
    )
    interface_stats: TransitionArrayMap = field(
        default_factory=lambda: cast(TransitionArrayMap, {})
    )

    def __post_init__(self) -> None:
        """Validate transitions_summary and interface_stats keys match.

        Ensures that both dictionaries track the same set of transitions,
        preventing data inconsistencies where transitions_summary exists for a transition
        but raw amplitudes don't (or vice versa).
        """
        ModelUtilities.validate_matching_keys(
            self.transitions_summary,
            self.interface_stats,
            "transitions_summary",
            "interface_stats",
        )

    def is_valid(self) -> bool:
        """Check if result has valid transitions with data."""
        return bool(self.transitions_summary) and any(
            stats is not None for stats in self.transitions_summary.values()
        )

    def __contains__(self, transition: Transition) -> bool:
        """Check if a transition exists in results.

        Enables Pythonic 'in' operator: `if transition in result:`

        Args:
            transition: The transition to check.

        Returns:
            True if transition exists in transitions_summary, False otherwise.
        """
        return transition in self.transitions_summary

    @cached_property
    def transition_count(self) -> int:
        """Get the number of transitions analyzed."""
        return len(self.transitions_summary)

    @cached_property
    def valid_transitions(self) -> list[Transition]:
        """Get cached list of valid transitions with data."""
        return [
            transition
            for transition, stats in self.transitions_summary.items()
            if stats is not None and stats.count > 0
        ]

    def get_amplitudes_for_transition(
        self, transition: Transition
    ) -> NDArray[np.float64] | None:
        """Retrieve raw amplitudes for a specific transition."""
        return self.interface_stats.get(transition)

    def has_transition(self, transition: Transition) -> bool:
        """Check if a transition exists in results with valid data.

        Args:
            transition: The transition to check.

        Returns:
            True if transition exists and has valid statistics, False otherwise.
        """
        stats = self.transitions_summary.get(transition)
        return stats is not None and stats.count > 0

    def get_transitions_with_minimum_count(self, min_count: int) -> list[Transition]:
        """Get all transitions with at least the specified sample count.

        Args:
            min_count: Minimum number of samples required.

        Returns:
            List of transitions meeting the count threshold.
        """
        return [
            transition
            for transition, stats in self.transitions_summary.items()
            if stats is not None and stats.count >= min_count
        ]

    def get_stats_dict(self) -> dict[str, float]:
        """Return statistics dictionary for FormattableModel formatting.

        Provides statistics for consistent __repr__/__str__ formatting
        via FormattableModel.

        Returns:
            Dictionary with transition counts.
        """
        return {
            "transitions": float(self.transition_count),
            "valid_transitions": float(len(self.valid_transitions)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - transition_count: Total number of transitions analyzed
            - valid_transition_count: Number of transitions with valid statistical data
            - valid_transitions: List of transitions with data (as strings, e.g., "0->1")
            - valid: Boolean indicating if result has at least one valid transition
        """
        return {
            "transition_count": self.transition_count,
            "valid_transition_count": len(self.valid_transitions),
            "valid_transitions": [str(t) for t in self.valid_transitions],
            "valid": self.is_valid(),
        }

    def summary(self) -> str:
        """Return a human-readable summary of the result.

        Returns:
            String containing:
            - transitions: Total number of transitions analyzed
            - valid: Number of transitions with valid statistical data
            - valid_list: List of transition strings (e.g., "0->1")
        """
        valid_list = [str(t) for t in self.valid_transitions]
        return (
            f"InterfaceReflectionResult(transitions={self.transition_count}, "
            f"valid={len(self.valid_transitions)}, "
            f"valid_list={valid_list})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterfaceReflectionResult:
        """Create result from dictionary representation.

        Args:
            data: Dictionary with keys:
                - transitions_summary: Dictionary mapping transitions to FaciesStats
                  (reconstructed). Keys must be dicts with from_facies/to_facies keys
                - interface_stats: Dictionary mapping transitions to amplitude arrays
                  (reconstructed). Keys must be dicts with from_facies/to_facies keys

        Returns:
            InterfaceReflectionResult instance with reconstructed transition maps.

        Raises:
            ValueError: If transitions_summary and interface_stats keys don't match.
            TypeError: If transitions cannot be reconstructed from keys.
        """
        transitions_summary = ModelUtilities.reconstruct_transition_stats_map(
            data, "transitions_summary"
        )
        interface_stats = ModelUtilities.reconstruct_transition_array_map(
            data, "interface_stats"
        )

        return cls(
            transitions_summary=transitions_summary,
            interface_stats=interface_stats,
        )


@dataclass
class AvoAnalysisResult(StatisticalResult, FormattableModel):
    """Comprehensive AVO analysis result combining multiple techniques.

    Inherits FormattableModel for consistent __repr__/__str__ formatting.
    """

    gradient_corr: GradientCorrelationResult
    boundary_amps: BoundaryAmpsResult
    interface_summary: TransitionStatsMap
    interface_raw: TransitionArrayMap
    facies_disc: FaciesDiscriminationResult

    def __post_init__(self) -> None:
        """Validate consistency across all analysis components.

        Ensures that interface_summary and interface_raw maps contain
        identical transition keys, preventing data inconsistencies across
        the comprehensive analysis result.
        """
        ModelUtilities.validate_matching_keys(
            self.interface_summary,
            self.interface_raw,
            "interface_summary",
            "interface_raw",
        )

    def is_valid(self) -> bool:
        """Check if all analysis components are valid."""
        return (
            self.gradient_corr.is_valid()
            and self.boundary_amps.is_valid()
            and self.facies_disc.is_valid()
        )

    def has_interface_data(self) -> bool:
        """Check if interface analysis data exists."""
        return bool(self.interface_summary)

    @property
    def all_valid_components(self) -> list[str]:
        """Return list of valid analysis components.

        Identifies which analysis techniques produced valid results from the
        comprehensive AVO analysis. Each component is evaluated based on its
        internal validation logic.

        Returns:
            List of component names that are valid (non-empty):
            - 'gradient_correlation': Gradient correlation analysis valid
            - 'boundary_amplitudes': Boundary amplitude analysis valid
            - 'facies_discrimination': Facies discrimination analysis valid
            - 'interface_reflection': Interface reflection data present

        Examples:
            >>> result.all_valid_components
            ['gradient_correlation', 'boundary_amplitudes', 'facies_discrimination']
        """
        return ModelUtilities.build_available_results(
            {
                "gradient_correlation": self.gradient_corr.is_valid(),
                "boundary_amplitudes": self.boundary_amps.is_valid(),
                "facies_discrimination": self.facies_disc.is_valid(),
                "interface_reflection": self.has_interface_data(),
            }
        )

    @property
    def analysis_coverage(self) -> float:
        """Return percentage of analysis components that are valid."""
        valid_count = len(self.all_valid_components)
        return (valid_count / 4) * 100

    def get_stats_dict(self) -> dict[str, float]:
        """Return statistics dictionary for FormattableModel formatting.

        Provides statistics for consistent __repr__/__str__ formatting
        via FormattableModel.

        Returns:
            Dictionary with component validity and coverage metrics.
        """
        return {
            "valid_components": float(len(self.all_valid_components)),
            "coverage_pct": self.analysis_coverage,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - gradient_corr: Serialized gradient correlation result
            - boundary_amps: Serialized boundary amplitudes result
            - facies_disc: Serialized facies discrimination result
            - interface_summary_count: Number of transitions in interface analysis
            - coverage: Percentage (0-100) of valid analysis components
            - valid: Boolean indicating if all major components are valid
        """
        return {
            "gradient_corr": self.gradient_corr.to_dict(),
            "boundary_amps": self.boundary_amps.to_dict(),
            "facies_disc": self.facies_disc.to_dict(),
            "interface_summary_count": len(self.interface_summary),
            "coverage": self.analysis_coverage,
            "valid": self.is_valid(),
        }

    def summary(self) -> str:
        """Return a human-readable summary of all analysis results.

        Returns:
            String containing summaries from all four analysis components:
            - gradient_corr: GradientCorrelationResult summary
            - boundary_amps: BoundaryAmpsResult summary
            - facies_disc: FaciesDiscriminationResult summary
            - coverage: Percentage (0-100) of valid analysis components
        """
        grad_summary = self.gradient_corr.summary()
        bound_summary = self.boundary_amps.summary()
        disc_summary = self.facies_disc.summary()
        coverage_pct = f"{self.analysis_coverage:.1f}%"

        return (
            f"AvoAnalysisResult("
            f"gradient_corr={grad_summary}, "
            f"boundary_amps={bound_summary}, "
            f"facies_disc={disc_summary}, "
            f"coverage={coverage_pct})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvoAnalysisResult:
        """Create result from dictionary representation.

        Args:
            data: Dictionary with required keys:
                - gradient_corr: Dictionary for GradientCorrelationResult (reconstructed)
                - boundary_amps: Dictionary for BoundaryAmpsResult (reconstructed)
                - facies_disc: Dictionary for FaciesDiscriminationResult (reconstructed)
                - interface_summary: Dictionary mapping transitions to FaciesStats (reconstructed)
                - interface_raw: Dictionary mapping transitions to amplitude arrays (reconstructed)

        Returns:
            AvoAnalysisResult instance with all analysis components reconstructed.

        Raises:
            ValueError: If consistency checks fail (interface_summary/raw key mismatch) or
                        components are invalid.
            KeyError: If required keys are missing from input dictionary.
            TypeError: If nested objects cannot be reconstructed.
        """
        gradient_corr = GradientCorrelationResult.from_dict(data["gradient_corr"])
        boundary_amps = BoundaryAmpsResult.from_dict(data["boundary_amps"])
        facies_disc = FaciesDiscriminationResult.from_dict(data["facies_disc"])

        interface_summary = ModelUtilities.reconstruct_transition_stats_map(
            data, "interface_summary"
        )
        interface_raw = ModelUtilities.reconstruct_transition_array_map(
            data, "interface_raw"
        )

        return cls(
            gradient_corr=gradient_corr,
            boundary_amps=boundary_amps,
            interface_summary=interface_summary,
            interface_raw=interface_raw,
            facies_disc=facies_disc,
        )
