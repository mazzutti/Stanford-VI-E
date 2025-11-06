"""Facies-specific data models.

This module contains models for facies-related analysis data.

Validation: Uses validators from src.analysis.validators for improved
error handling and reduced code duplication.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, cast
from functools import total_ordering

import numpy as np

from .base import ModelUtilities
from .formatters import FormattableModel
from ..validators import CountValidator, QuantileValidator, ValidationError

__all__ = [
    "FaciesStats",
]


@total_ordering
@dataclass(slots=True)
class FaciesStats(FormattableModel):
    """Per-facies amplitude statistics with computed properties and validation.
    
    Inherits formatting from FormattableModel for consistent __repr__/__str__
    implementations across all statistical model classes.
    """

    count: int = 0
    mean: float = float("nan")
    std: float = float("nan")
    median: float = float("nan")
    q25: float = float("nan")
    q75: float = float("nan")
    min: float = float("nan")
    max: float = float("nan")

    def __post_init__(self) -> None:
        """Validate statistical consistency."""
        try:
            CountValidator.validate_count(self.count, "count", allow_zero=True)
        except ValidationError as e:
            raise ValidationError(f"Invalid FaciesStats: {e}") from e

        # Empty stats or stats with missing bounds are valid
        if (
            self.count == 0
            or ModelUtilities.is_nan(self.min)
            or ModelUtilities.is_nan(self.max)
        ):
            return

        # Check if any quantile value is NaN (if so, skip ordering validation)
        quantile_values = [self.min, self.q25, self.median, self.q75, self.max]
        if any(ModelUtilities.is_nan(v) for v in quantile_values):
            return

        # All values are valid, validate min, q25, median ordering
        try:
            QuantileValidator.validate_quantile_order(self.q25, self.median, self.q75)
            # Also check min <= q25 and q75 <= max
            if self.min > self.q25:
                raise ValidationError(
                    f"min={self.min} > q25={self.q25}. Expected min <= q25."
                )
            if self.q75 > self.max:
                raise ValidationError(
                    f"q75={self.q75} > max={self.max}. Expected q75 <= max."
                )
        except ValidationError as e:
            raise ValidationError(f"Invalid quantile ordering: {e}") from e

    def is_valid(self) -> bool:
        """Check if statistics have valid data."""
        return self.count > 0 and not ModelUtilities.is_nan(self.mean)

    def is_empty(self) -> bool:
        """Check if statistics represent empty data."""
        return self.count == 0

    def __lt__(self, other: FaciesStats) -> bool:
        """Compare statistics by mean value for sorting."""
        if not ModelUtilities.check_facies_stats_type(other):
            return NotImplemented
        return self.mean < other.mean

    def __eq__(self, other: object) -> bool:
        """Check equality of statistics based on mean value.

        For consistency with __lt__, equality is based on mean value only.
        This allows FaciesStats to be properly ordered and grouped by mean.

        Accepts 'object' type per Python comparison protocol requirement (PEP 207).
        Type is narrowed after isinstance check.
        """
        if not ModelUtilities.check_facies_stats_type(other):
            return NotImplemented
        # Cast to FaciesStats after type check
        other_stats = cast(FaciesStats, other)
        # Check mean equality only (handling NaN cases)
        # This is consistent with __lt__ which only compares means
        mean_equal = (
            ModelUtilities.is_nan(self.mean) and ModelUtilities.is_nan(other_stats.mean)
        ) or abs(self.mean - other_stats.mean) < 1e-10
        return mean_equal

    def __hash__(self) -> int:
        """Return hash based on mean for use in collections.

        This is consistent with __eq__ which compares only mean values.
        """
        # Use rounded mean to allow FaciesStats to be used in sets/dicts
        mean_val = round(self.mean, 10) if not ModelUtilities.is_nan(self.mean) else 0
        return hash(mean_val)

    def get_stats_dict(self) -> Dict[str, float]:
        """Return statistics dictionary for FormattableModel formatting.
        
        Used by parent class FormattableModel for consistent __repr__/__str__.
        """
        return {
            "count": float(self.count),
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "min": self.min,
            "max": self.max,
        }

    @property
    def iqr(self) -> float:
        """Calculate interquartile range."""
        if not ModelUtilities.validate_numeric_pair(
            self.q75, self.q25, "IQR quantiles"
        ):
            return np.nan
        return self.q75 - self.q25

    @property
    def range(self) -> float:
        """Calculate the range (max - min)."""
        if not ModelUtilities.validate_numeric_pair(self.max, self.min, "range bounds"):
            return np.nan
        return self.max - self.min

    @property
    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (std/mean).

        Returns NaN if mean or std are NaN, or if mean is zero (undefined).
        """
        # Check all preconditions at once for efficiency
        if ModelUtilities.is_nan(self.mean) or ModelUtilities.is_nan(self.std):
            return np.nan
        if self.mean == 0:
            return np.nan
        return self.std / abs(self.mean)

    def to_dict(self) -> Dict[str, float]:
        """Convert statistics to dictionary for compatibility."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "min": self.min,
            "max": self.max,
        }

    @classmethod
    def from_dict(cls, stats_dict: Dict[str, float]) -> FaciesStats:
        """Create FaciesStats from dictionary.

        Args:
            stats_dict: Dictionary with statistical fields (count, mean, std, etc.).
                Missing fields default to NaN (or 0 for count).

        Returns:
            FaciesStats instance with values from dict or defaults.
        """
        # Extract float fields using mapping for cleaner, more maintainable code
        float_field_names = ["mean", "std", "median", "q25", "q75", "min", "max"]
        float_fields = {
            name: ModelUtilities.safe_float(stats_dict.get(name))
            for name in float_field_names
        }
        return cls(count=int(stats_dict.get("count", 0)), **float_fields)
