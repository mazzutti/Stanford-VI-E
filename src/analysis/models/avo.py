"""AVO (Amplitude Variation with Offset) analysis models.

This module contains models for AVO technique analysis and comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .base import ModelUtilities, ValidationConfig
from .config import Transition
from .facies import FaciesStats
from .formatters import FormattableModel

if TYPE_CHECKING:
    from .statistics import BoundaryAmpsResult, GradientCorrelationResult

# Some imports in this module occur in local scopes to avoid import cycles
# and heavier dependencies at import time (statistics-related helpers may
# be optional). These late imports are intentional; silence pylint's
# import-outside-toplevel warnings here with a brief justification.

# Type aliases for common patterns
TransitionStatsMap = dict[Transition, FaciesStats | None]
TransitionArrayMap = dict[Transition, NDArray[np.float64] | None]

__all__ = [
    "TechniqueComparison",
    "AvoStats",
    "AvoResults",
]

@dataclass
class TechniqueComparison:
    """Typed result for technique comparison summaries with validation.

    Attributes
    ----------
    avo
        Mapping of AVO metric names to numeric values (e.g. Pearson,
        Spearman).
    winner
        Name of the winning technique (e.g. "AVO").
    difference
        Numeric difference between top techniques according to the chosen
        metric.
    """

    avo: AvoStats
    winner: str
    difference: float
    # Common metric name constants for comparisons (class-level)
    # Use UPPER_CASE names for public constants; exempt from naming-style
    # checks in this class where constants improve readability.
    GRADIENT_CORRELATION: str = "gradient_correlation"
    BOUNDARY_AMPLITUDE: str = "boundary_amplitude"
    FACIES_DISCRIMINATION: str = "facies_discrimination"

    def __post_init__(self) -> None:
        """Validate comparison result."""
        if not self.winner:
            raise ValueError("winner cannot be empty")
        if self.difference < 0:
            raise ValueError("difference cannot be negative")

    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if difference is statistically significant."""
        return self.difference > threshold

    @property
    def avo_strength(self) -> float | None:
        """Return the absolute value of primary correlation strength."""
        if self.avo.pearson_correlation is not None:
            return abs(self.avo.pearson_correlation)
        if self.avo.spearman_correlation is not None:
            return abs(self.avo.spearman_correlation)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - winner: Name of the winning technique
            - difference: Numeric difference between top techniques
            - is_significant: Boolean indicating statistical significance (threshold: 0.05)
            - avo_strength: Absolute value of primary correlation strength
            - avo_stats: Serialized AVO statistics including all correlation metrics
        """
        return {
            "winner": self.winner,
            "difference": self.difference,
            "is_significant": self.is_significant(),
            "avo_strength": self.avo_strength,
            "avo_stats": self.avo.to_dict(),
        }

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"TechniqueComparison(winner={self.winner}, "
            f"difference={self.difference:.4f}, "
            f"is_significant={self.is_significant()})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TechniqueComparison:
        """Create comparison from dictionary representation.

        Args:
            data: Dictionary with required keys:
                - winner: Name of winning technique
                - difference: Numeric difference value
                - avo_stats: Dictionary with correlation statistics (pearson_correlation,
                            spearman_correlation, pearson_pvalue, spearman_pvalue)

        Returns:
            TechniqueComparison instance with values from dict.

        Raises:
            ValueError: If winner is empty, difference is negative, or avo_stats is missing.
            KeyError: If required keys are missing from input dictionary.
        """
        avo_stats = AvoStats.from_dict(data["avo_stats"])
        return cls(
            avo=avo_stats,
            winner=data["winner"],
            difference=float(data["difference"]),
        )

@dataclass
class AvoStats(FormattableModel):
    """Typed container for AVO technique statistics with validation.

    Fields are optional to support partial results. Extras may be provided in
    the ``extras`` mapping for non-standard metrics.

    Inherits formatting from FormattableModel for consistent __repr__/__str__
    implementations.
    """

    # generic container for other numeric metrics (optional)
    extras: dict[str, float] = field(default_factory=lambda: cast(dict[str, float], {}))
    pearson_correlation: float | None = None
    pearson_pvalue: float | None = None
    spearman_correlation: float | None = None
    spearman_pvalue: float | None = None

    def __post_init__(self) -> None:
        """Validate correlation and p-value ranges."""
        ModelUtilities.validate_optional_numeric_fields(
            {
                "pearson_correlation": self.pearson_correlation,
                "spearman_correlation": self.spearman_correlation,
            },
            ValidationConfig.CORRELATION_MIN,
            ValidationConfig.CORRELATION_MAX,
        )
        ModelUtilities.validate_optional_numeric_fields(
            {
                "pearson_pvalue": self.pearson_pvalue,
                "spearman_pvalue": self.spearman_pvalue,
            },
            ValidationConfig.PVALUE_MIN,
            ValidationConfig.PVALUE_MAX,
        )

    @property
    def has_data(self) -> bool:
        """Check if any statistical data is present."""
        return (
            self.pearson_correlation is not None
            or self.spearman_correlation is not None
        )

    def is_significant(
        self, alpha: float = ValidationConfig.SIGNIFICANCE_THRESHOLD
    ) -> bool:
        """Check if results are statistically significant.

        Args:
            alpha: Significance level threshold (default 0.05).

        Returns:
            True if any p-value is below alpha threshold.
        """
        if self.pearson_pvalue is not None and self.pearson_pvalue < alpha:
            return True
        if self.spearman_pvalue is not None and self.spearman_pvalue < alpha:
            return True
        return False

    @cached_property
    def strongest_correlation(self) -> tuple[str, float | None]:
        """Get cached strongest correlation method and its value."""
        if self.pearson_correlation is None and self.spearman_correlation is None:
            return "none", None

        # Compute absolute values using helper and compare
        pearson_abs = ModelUtilities.get_absolute_correlation(self.pearson_correlation)
        spearman_abs = ModelUtilities.get_absolute_correlation(
            self.spearman_correlation
        )

        # Return strongest based on absolute value
        if pearson_abs >= spearman_abs:
            return "Pearson", self.pearson_correlation
        return "Spearman", self.spearman_correlation

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - pearson_correlation: Pearson coefficient or None
            - pearson_pvalue: Pearson p-value or None
            - spearman_correlation: Spearman coefficient or None
            - spearman_pvalue: Spearman p-value or None
            - extras: Dictionary of additional metrics
        """
        return {
            "pearson_correlation": self.pearson_correlation,
            "pearson_pvalue": self.pearson_pvalue,
            "spearman_correlation": self.spearman_correlation,
            "spearman_pvalue": self.spearman_pvalue,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvoStats:
        """Create statistics from dictionary representation.

        Args:
            data: Dictionary with optional keys (missing keys are None):
                - pearson_correlation: Pearson correlation coefficient
                - pearson_pvalue: Pearson test p-value
                - spearman_correlation: Spearman correlation coefficient
                - spearman_pvalue: Spearman test p-value
                - extras: Additional metrics dictionary (defaults to {})

        Returns:
            AvoStats instance with data from dictionary or None values.

        Raises:
            ValueError: If provided values are outside valid ranges.
        """
        return cls(
            pearson_correlation=data.get("pearson_correlation"),
            pearson_pvalue=data.get("pearson_pvalue"),
            spearman_correlation=data.get("spearman_correlation"),
            spearman_pvalue=data.get("spearman_pvalue"),
            extras=ModelUtilities.safe_get_dict(data, "extras"),
        )

    def get_stats_dict(self) -> dict[str, float]:
        """Return statistics dictionary for FormattableModel formatting."""
        return {
            "pearson_correlation": self.pearson_correlation or 0.0,
            "pearson_pvalue": self.pearson_pvalue or 0.0,
            "spearman_correlation": self.spearman_correlation or 0.0,
            "spearman_pvalue": self.spearman_pvalue or 0.0,
        }

@dataclass
class AvoResults:
    """Structured container for AVO analysis results with computed properties."""

    boundary_amps: BoundaryAmpsResult | None = None
    gradient_correlation: GradientCorrelationResult | None = None
    separation_matrix: NDArray[np.float64] | None = None
    facies_amplitudes: dict[int, NDArray[np.float64]] = field(
        default_factory=lambda: cast(dict[int, NDArray[np.float64]], {})
    )
    interface_stats_summary: TransitionStatsMap = field(
        default_factory=lambda: cast(TransitionStatsMap, {})
    )

    @cached_property
    def available_results(self) -> list[str]:
        """Get list of available result components (cached)."""
        return ModelUtilities.build_available_results(
            {
                "boundary_amplitudes": self.boundary_amps is not None,
                "gradient_correlation": self.gradient_correlation is not None,
                "separation_matrix": self.separation_matrix is not None,
                "facies_amplitudes": bool(self.facies_amplitudes),
                "interface_statistics": bool(self.interface_stats_summary),
            }
        )

    def has_complete_results(self) -> bool:
        """Check if all major results are present."""
        return (
            self.boundary_amps is not None
            and self.gradient_correlation is not None
            and self.separation_matrix is not None
            and bool(self.facies_amplitudes)
        )

    def __contains__(self, transition: Transition) -> bool:
        """Check if a transition exists in interface stats summary.

        Enables Pythonic 'in' operator: `if transition in results:`

        Args:
            transition: The transition to check.

        Returns:
            True if transition exists in interface_stats_summary, False otherwise.
        """
        return transition in self.interface_stats_summary

    def has_transition(self, transition: Transition) -> bool:
        """Check if a transition exists with valid statistics.

        Args:
            transition: The transition to check.

        Returns:
            True if transition exists and has valid statistics, False otherwise.
        """
        stats = self.interface_stats_summary.get(transition)
        return stats is not None and stats.count > 0

    def get_transitions_for_facies(self, facies: int) -> list[Transition]:
        """Get all transitions involving a specific facies.

        Args:
            facies: The facies index to search for.

        Returns:
            List of transitions where facies is either source or target.
        """
        return [
            t
            for t in self.interface_stats_summary
            if facies in (t.from_facies, t.to_facies)
        ]

    @property
    def facies_count(self) -> int:
        """Get the number of facies with amplitude data."""
        return len(self.facies_amplitudes)

    @property
    def transition_count(self) -> int:
        """Get the number of interface transitions analyzed."""
        return len(self.interface_stats_summary)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - boundary_amps: Serialized boundary amplitudes result or None
            - gradient_correlation: Serialized gradient correlation result or None
            - facies_amplitudes_count: Number of facies with amplitude data
            - interface_transitions: Number of interface transitions analyzed
            - complete: Boolean indicating if all major results are present
        """
        return {
            "boundary_amps": (
                self.boundary_amps.to_dict() if self.boundary_amps else None
            ),
            "gradient_correlation": (
                self.gradient_correlation.to_dict()
                if self.gradient_correlation
                else None
            ),
            "facies_amplitudes_count": self.facies_count,
            "interface_transitions": self.transition_count,
            "complete": self.has_complete_results(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvoResults:
        """Create results from dictionary representation.

                Args:
                        data: Dictionary with optional keys:
                                - boundary_amps: Serialized BoundaryAmpsResult
                                    (reconstructed if present)
                                - gradient_correlation: Serialized GradientCorrelationResult
                                    (reconstructed if present)
                                - separation_matrix: 2D array data (reconstructed as ndarray
                                    if present)
                                - facies_amplitudes: Dictionary mapping facies IDs to amplitude
                                    arrays
                                - interface_stats_summary: Dictionary mapping transitions to
                                    facies statistics

        Returns:
            AvoResults instance with reconstructed data from dictionary.

        Raises:
            ValueError: If nested results cannot be reconstructed from dictionary format.
            TypeError: If array data is in invalid format.
        """
        from .statistics import (
            BoundaryAmpsResult,
            GradientCorrelationResult,
        )

        boundary_amps = None
        if data.get("boundary_amps"):
            boundary_amps = BoundaryAmpsResult.from_dict(data["boundary_amps"])

        gradient_correlation = None
        if data.get("gradient_correlation"):
            gradient_correlation = GradientCorrelationResult.from_dict(
                data["gradient_correlation"]
            )

        interface_stats_summary = ModelUtilities.reconstruct_transition_stats_map(
            data, "interface_stats_summary"
        )

        return cls(
            boundary_amps=boundary_amps,
            gradient_correlation=gradient_correlation,
            separation_matrix=(
                np.array(data["separation_matrix"])
                if data.get("separation_matrix")
                else None
            ),
            facies_amplitudes={
                int(k): np.array(v)
                for k, v in data.get("facies_amplitudes", {}).items()
            },
            interface_stats_summary=interface_stats_summary,
        )

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"AvoResults(facies={self.facies_count}, "
            f"transitions={self.transition_count}, "
            f"complete={self.has_complete_results()})"
        )
