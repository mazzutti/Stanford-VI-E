"""Base utilities and abstract classes for analysis models.

This module provides common functionality for all data models used in
analysis workflows, including validation utilities and the abstract
base class for statistical results.
"""

from __future__ import annotations
from typing import (
    Dict,
    Optional,
    List,
    ClassVar,
    Any,
    Callable,
    Union,
    TYPE_CHECKING,
    cast,
)
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .facies import FaciesStats
    from .config import Transition

__all__ = [
    "ValidationConfig",
    "ModelUtilities",
    "StatisticalResult",
    "STATS_REPR_PRECISION",
    "STR_PRECISION",
    "SUMMARY_PRECISION",
    "ANALYSIS_COMPONENTS_COUNT",
]


class ValidationConfig:
    """Centralized validation configuration for statistical analysis.

    This class consolidates validation constants used throughout the module,
    making it easier to adjust thresholds and maintain consistency.
    """

    CORRELATION_MIN: ClassVar[float] = -1.0
    CORRELATION_MAX: ClassVar[float] = 1.0
    PVALUE_MIN: ClassVar[float] = 0.0
    PVALUE_MAX: ClassVar[float] = 1.0
    SIGNIFICANCE_THRESHOLD: ClassVar[float] = 0.05


# Formatting precision constants
STATS_REPR_PRECISION = 6  # FaciesStats repr output precision
STR_PRECISION = 4  # FaciesStats str output precision
SUMMARY_PRECISION = 4  # Default precision for summary string formatting (e.g., 0.0000)
ANALYSIS_COMPONENTS_COUNT = 4  # AvoAnalysisResult analysis component count


class ModelUtilities:
    """Object-oriented utilities for model validation, conversion, and computation.

    Provides stateless utility methods for common operations across model classes:
    - Type checking and NaN handling
    - Numeric conversion and validation
    - Dictionary operations
    - Statistical computations
    - Result aggregation
    """

    # ========================================================================
    # NaN and Type Checking Utilities
    # ========================================================================

    @staticmethod
    def is_nan(value: Optional[float]) -> bool:
        """Check if a float value is NaN.

        Args:
            value: The value to check (can be None).

        Returns:
            True if value is NaN or None, False otherwise.
        """
        return (
            value is None or value != value
        )  # NaN is the only value that doesn't equal itself

    @staticmethod
    def safe_float(
        value: Union[str, int, float, None], default: float = np.nan
    ) -> float:
        """Safely convert a value to float with NaN as default.

        Args:
            value: The value to convert (str, int, float, or None).
            default: Default value if conversion fails or value is None.

        Returns:
            Converted float value or default.
        """
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def check_facies_stats_type(other: object) -> bool:
        """Check if value is a FaciesStats instance for comparison operations.

        Accepts 'object' type since this is a type guard that must work with any value.

        Consolidates the isinstance check pattern used in all FaciesStats
        comparison methods (__lt__, __le__, __gt__, __ge__, __eq__) to reduce
        duplication and ensure consistent type validation.

        Args:
            other: The value to check.

        Returns:
            True if other is a FaciesStats instance, False otherwise.
        """
        from .facies import FaciesStats

        return isinstance(other, FaciesStats)

    @staticmethod
    def get_absolute_correlation(value: Optional[float]) -> float:
        """Safely get absolute value of correlation, returning -1.0 if None.

        Helper for comparing correlation magnitudes when either may be None.
        Used in AvoStats.strongest_correlation to find strongest method.

        Args:
            value: Correlation value or None.

        Returns:
            Absolute value of correlation, or -1.0 if None (ensures valid comparisons).
        """
        return abs(value) if value is not None else -1.0

    # ========================================================================
    # Numeric Validation Utilities
    # ========================================================================

    @staticmethod
    def validate_numeric_value(
        value: float,
        range_min: float,
        range_max: float,
        field_name: str,
        context: str = "",
    ) -> None:
        """Validate that a numeric value is within range with consistent error messaging.

        Consolidates repeated validation error formatting used in correlation and p-value
        validation methods, reducing duplication and ensuring consistent messaging.

        Args:
            value: The value to validate (must not be None).
            range_min: Minimum allowed value (inclusive).
            range_max: Maximum allowed value (inclusive).
            field_name: Name of field being validated (e.g., "Pearson correlation").
            context: Additional context about valid range (e.g., "normalized between -1 and 1").

        Raises:
            ValueError: If value is not within [range_min, range_max].
        """
        # Allow NaN for cases where computation is impossible (e.g., constant data)
        if np.isnan(float(value)):
            return

        if not (range_min <= value <= range_max):
            context_str = f" ({context})" if context else ""
            raise ValueError(
                f"{field_name} must be in [{range_min}, {range_max}]{context_str}, got {value}"
            )

    @staticmethod
    def validate_in_range(
        value: Optional[float],
        range_min: float,
        range_max: float,
        name: str,
        allow_none: bool = False,
    ) -> None:
        """Validate that value is within the specified range.

        Args:
            value: The value to validate.
            range_min: Minimum allowed value (inclusive).
            range_max: Maximum allowed value (inclusive).
            name: Name of the value for error messages.
            allow_none: If True, None values pass validation.

        Raises:
            ValueError: If value is outside the allowed range.
        """
        if value is None:
            if not allow_none:
                raise ValueError(f"{name} cannot be None")
            return
        if not (range_min <= value <= range_max):
            raise ValueError(
                f"{name} must be in [{range_min}, {range_max}], got {value}"
            )

    @staticmethod
    def validate_optional_numeric_fields(
        fields: Dict[str, Optional[float]],
        range_min: float,
        range_max: float,
    ) -> None:
        """Validate multiple optional numeric fields with identical range constraints.

        Consolidates the pattern of validating multiple optional fields with the same
        range constraints (e.g., AvoStats validation of Pearson and Spearman fields).

        Args:
            fields: Dictionary mapping field names to values to validate.
            range_min: Minimum allowed value (inclusive).
            range_max: Maximum allowed value (inclusive).

        Raises:
            ValueError: If any value is outside the allowed range.
        """
        for field_name, value in fields.items():
            ModelUtilities.validate_in_range(
                value,
                range_min,
                range_max,
                field_name,
                allow_none=True,
            )

    @staticmethod
    def validate_matching_keys(
        dict1: Dict[Any, Any],
        dict2: Dict[Any, Any],
        dict1_name: str = "dict1",
        dict2_name: str = "dict2",
    ) -> None:
        """Validate that two dictionaries have identical key sets.

        Consolidates repeated validation logic used in multiple __post_init__ methods
        that ensure corresponding dictionaries maintain consistency (e.g., summary and
        raw data maps must have the same transitions).

        Args:
            dict1: First dictionary to compare.
            dict2: Second dictionary to compare.
            dict1_name: Name of first dict for error messages.
            dict2_name: Name of second dict for error messages.

        Raises:
            ValueError: If key sets don't match, with detailed reporting of missing keys.
        """
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())

        if keys1 != keys2:
            missing_in_dict2 = keys1 - keys2
            missing_in_dict1 = keys2 - keys1
            error_parts: List[str] = []
            if missing_in_dict2:
                error_parts.append(
                    f"keys in {dict1_name} but not {dict2_name}: {len(missing_in_dict2)}"
                )
            if missing_in_dict1:
                error_parts.append(
                    f"keys in {dict2_name} but not {dict1_name}: {len(missing_in_dict1)}"
                )
            raise ValueError(
                f"{dict1_name} and {dict2_name} must have identical keys. "
                f"Issues: {'; '.join(error_parts)}"
            )

    @staticmethod
    def validate_numeric_pair(
        val1: Optional[float],
        val2: Optional[float],
        name: Optional[str] = None,
    ) -> bool:
        """Validate that both values in a numeric pair are not NaN.

        Consolidates repeated NaN checking pattern used in range/IQR properties
        that validate a pair of quantile values before performing computation.

        Args:
            val1: First value to check (e.g., q75).
            val2: Second value to check (e.g., q25).
            name: Optional name for the pair (used for logging/error messages).

        Returns:
            False if either value is NaN, True otherwise (both are valid).
        """
        return not (ModelUtilities.is_nan(val1) or ModelUtilities.is_nan(val2))

    # ========================================================================
    # Dictionary and Data Conversion Utilities
    # ========================================================================

    @staticmethod
    def safe_get_dict(
        data: Dict[str, Any],
        key: str,
        default_factory: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Safely extract and return a dictionary from another dict, with default.

        Consolidates the pattern of `.get("key", {})` used in multiple from_dict()
        methods to reduce duplication and improve clarity.

        Args:
            data: The source dictionary to extract from.
            key: The key to look up.
            default_factory: Callable to create default (default: dict).

        Returns:
            The value at data[key] if it exists and is a dict, else default.
        """
        if default_factory is None:
            # use a typed empty dict factory to avoid partially unknown dict types
            def _default_factory() -> Dict[str, Any]:
                return {}

            default_factory = _default_factory

        value = data.get(key, default_factory())
        return (
            cast(Dict[str, Any], value)
            if isinstance(value, dict)
            else default_factory()
        )

    # ========================================================================
    # Statistical Computation Utilities
    # ========================================================================

    @staticmethod
    def compute_array_stats(arr: NDArray[np.float64]) -> Dict[str, float]:
        """Compute standard statistical measures for a numeric array.

        Consolidates the pattern of computing min, max, mean, std across
        multiple result classes to reduce duplication.

        Args:
            arr: Numeric array to compute statistics for.

        Returns:
            Dictionary with keys: min, max, mean, std (all as floats).
        """
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    @staticmethod
    def build_available_results(
        conditions: Dict[str, bool],
    ) -> List[str]:
        """Build list of available result names from condition checks.

        Consolidates the pattern of checking multiple conditions and appending
        corresponding names to a result list (used in AvoResults and AvoAnalysisResult).

        Args:
            conditions: Dictionary mapping result names to boolean conditions.

        Returns:
            List of result names where condition is True.
        """
        return [name for name, is_available in conditions.items() if is_available]

    @staticmethod
    def reconstruct_transition_stats_map(
        data: Dict[str, Any], key: str
    ) -> Dict["Transition", Optional["FaciesStats"]]:
        """Reconstruct a transition-keyed statistics map from dictionary data.

        Helper method to consolidate duplicate logic in from_dict() methods that
        reconstruct transition-keyed dictionaries mapping to FaciesStats objects.

        Args:
            data: Source dictionary containing the key.
            key: Dictionary key to extract (e.g., "interface_summary").

        Returns:
            Dictionary mapping Transition instances to FaciesStats (or None).
        """
        from .config import Transition
        from .facies import FaciesStats

        result: Dict["Transition", Optional["FaciesStats"]] = {}
        for transition_data, stats_dict in ModelUtilities.safe_get_dict(
            data, key
        ).items():
            transition = Transition.from_string_key(transition_data)
            result[transition] = (
                FaciesStats.from_dict(stats_dict) if stats_dict else None
            )
        return result

    @staticmethod
    def reconstruct_transition_array_map(
        data: Dict[str, Any], key: str
    ) -> Dict["Transition", Optional[NDArray[np.float64]]]:
        """Reconstruct a transition-keyed amplitude array map from dictionary data.

        Helper method to consolidate duplicate logic in from_dict() methods that
        reconstruct transition-keyed dictionaries mapping to numpy arrays.

        Args:
            data: Source dictionary containing the key.
            key: Dictionary key to extract (e.g., "interface_raw").

        Returns:
            Dictionary mapping Transition instances to numpy arrays (or None).
        """
        from .config import Transition

        result: Dict["Transition", Optional[NDArray[np.float64]]] = {}
        for transition_data, amps in ModelUtilities.safe_get_dict(data, key).items():
            transition = Transition.from_string_key(transition_data)
            result[transition] = np.array(amps) if amps else None
        return result


class StatisticalResult(ABC):
    """Abstract base class for statistical result models.

    Provides common interface for all statistical analysis results.
    """

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if the result contains valid statistical data."""
        pass

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of the result."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation.

        Subclasses should override for custom serialization.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement to_dict()"
        )
