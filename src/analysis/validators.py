"""Validation utilities for analysis models.

This module provides reusable validators for common data validation patterns
found across the analysis package. It reduces code duplication and provides
consistent error messages and behavior.

Validators handle:
    - Correlation coefficients (range [-1, 1])
    - P-values (range [0, 1])
    - Generic range validation
    - Count validation (non-negative integers)
    - Array shape validation
    - Quantile ordering

Design Principles:
    - Single responsibility per validator
    - Clear, actionable error messages
    - Type-safe with proper type hints
    - Composable and reusable across models
    - Extensive logging for debugging

Example Usage:
    >>> from src.analysis.validators import RangeValidator, CountValidator
    >>>
    >>> # Validate correlation
    >>> RangeValidator.validate_correlation(0.95)  # OK
    >>> RangeValidator.validate_correlation(1.5)   # Raises ValidationError
    >>>
    >>> # Validate p-value
    >>> RangeValidator.validate_pvalue(0.05)       # OK
    >>> RangeValidator.validate_pvalue(1.5)        # Raises ValidationError
    >>>
    >>> # Validate count
    >>> CountValidator.validate_count(100)         # OK
    >>> CountValidator.validate_count(-1)          # Raises ValidationError
    >>>
    >>> # Custom range
    >>> RangeValidator.validate_range(
    ...     value=50,
    ...     min_val=0,
    ...     max_val=100,
    ...     name="percentage"
    ... )
"""

import logging

from src.analysis.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "RangeValidator",
    "CountValidator",
    "QuantileValidator",
    "ValidationError",
]


class RangeValidator:
    """Validates numeric values fall within expected ranges.

    Provides common range validation methods for correlation values,
    p-values, and generic numeric ranges.
    """

    @staticmethod
    def validate_correlation(
        value: float,
        name: str = "correlation",
        *,
        allow_nan: bool = False,
    ) -> None:
        """Validate correlation coefficient is in [-1, 1].

        Parameters
        ----------
        value : float
            The correlation value to validate.
        name : str, default="correlation"
            Name for error messages (e.g., "pearson_correlation").
        allow_nan : bool, default=False
            If True, NaN values are accepted. Otherwise raises error.

        Raises
        ------
        ValidationError
            If value is outside [-1, 1] or NaN when not allowed.

        Examples
        --------
        >>> RangeValidator.validate_correlation(0.95)
        >>> RangeValidator.validate_correlation(0.95, name="spearman_r")
        >>> RangeValidator.validate_correlation(float('nan'), allow_nan=True)
        """
        import math

        if math.isnan(value):
            if not allow_nan:
                raise ValidationError(
                    f"{name} is NaN, which is not allowed. "
                    "Pass allow_nan=True to permit NaN values."
                )
            logger.debug(f"{name} is NaN (allowed)")
            return

        if not (-1.0 <= value <= 1.0):
            raise ValidationError(
                f"{name}={value} is outside valid range [-1, 1]. "
                "Correlation coefficients must be between -1 and 1."
            )
        logger.debug(f"{name}={value} is valid")

    @staticmethod
    def validate_pvalue(
        value: float,
        name: str = "p_value",
        *,
        allow_nan: bool = False,
    ) -> None:
        """Validate p-value is in [0, 1].

        Parameters
        ----------
        value : float
            The p-value to validate.
        name : str, default="p_value"
            Name for error messages (e.g., "pearson_pvalue").
        allow_nan : bool, default=False
            If True, NaN values are accepted.

        Raises
        ------
        ValidationError
            If value is outside [0, 1] or NaN when not allowed.

        Examples
        --------
        >>> RangeValidator.validate_pvalue(0.05)
        >>> RangeValidator.validate_pvalue(0.05, name="spearman_pvalue")
        """
        import math

        if math.isnan(value):
            if not allow_nan:
                raise ValidationError(f"{name} is NaN, which is not allowed.")
            logger.debug(f"{name} is NaN (allowed)")
            return

        if not (0.0 <= value <= 1.0):
            raise ValidationError(
                f"{name}={value} is outside valid range [0, 1]. "
                "P-values must be between 0 and 1."
            )
        logger.debug(f"{name}={value} is valid")

    @staticmethod
    def validate_range(
        value: float,
        min_val: float,
        max_val: float,
        name: str,
        *,
        allow_nan: bool = False,
        include_endpoints: bool = True,
    ) -> None:
        """Validate numeric value is within specified range.

        Parameters
        ----------
        value : float
            The value to validate.
        min_val : float
            Minimum acceptable value.
        max_val : float
            Maximum acceptable value.
        name : str
            Name for error messages (e.g., "threshold").
        allow_nan : bool, default=False
            If True, NaN values are accepted.
        include_endpoints : bool, default=True
            If True, range is [min, max]. If False, range is (min, max).

        Raises
        ------
        ValidationError
            If value is outside the specified range.

        Examples
        --------
        >>> # Percentage validation
        >>> RangeValidator.validate_range(
        ...     value=50, min_val=0, max_val=100, name="percentage"
        ... )

        >>> # Open interval (0, 100) - not including endpoints
        >>> RangeValidator.validate_range(
        ...     value=50,
        ...     min_val=0,
        ...     max_val=100,
        ...     name="ratio",
        ...     include_endpoints=False
        ... )
        """
        import math

        if math.isnan(value):
            if not allow_nan:
                raise ValidationError(f"{name} is NaN, which is not allowed.")
            logger.debug(f"{name} is NaN (allowed)")
            return

        if include_endpoints:
            valid = min_val <= value <= max_val
            range_str = f"[{min_val}, {max_val}]"
        else:
            valid = min_val < value < max_val
            range_str = f"({min_val}, {max_val})"

        if not valid:
            raise ValidationError(f"{name}={value} is outside valid range {range_str}.")
        logger.debug(f"{name}={value} is valid")

    @staticmethod
    def validate_probability(
        value: float,
        name: str = "probability",
        *,
        allow_nan: bool = False,
    ) -> None:
        """Validate value is valid probability [0, 1].

        Alias for validate_pvalue() with a different default name.

        Parameters
        ----------
        value : float
            Probability value to validate.
        name : str, default="probability"
            Name for error messages.
        allow_nan : bool, default=False
            If True, NaN values are accepted.

        Examples
        --------
        >>> RangeValidator.validate_probability(0.75)
        """
        RangeValidator.validate_pvalue(value, name=name, allow_nan=allow_nan)


class CountValidator:
    """Validates count-like values (non-negative integers)."""

    @staticmethod
    def validate_count(
        value: int,
        name: str = "count",
        *,
        allow_zero: bool = True,
    ) -> None:
        """Validate count is non-negative integer.

        Parameters
        ----------
        value : int
            Count value to validate.
        name : str, default="count"
            Name for error messages (e.g., "sample_count").
        allow_zero : bool, default=True
            If False, value must be > 0.

        Raises
        ------
        ValidationError
            If value is negative or zero (when allow_zero=False).

        Examples
        --------
        >>> CountValidator.validate_count(100)  # OK
        >>> CountValidator.validate_count(0)    # OK (allow_zero=True by default)
        >>> CountValidator.validate_count(0, allow_zero=False)  # Error
        >>> CountValidator.validate_count(-1)   # Error
        """
        if not isinstance(value, int):
            raise ValidationError(
                f"{name} must be an integer, got {type(value).__name__}"
            )

        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")

        if value == 0 and not allow_zero:
            raise ValidationError(f"{name} must be greater than 0, got {value}")

        logger.debug(f"{name}={value} is valid")

    @staticmethod
    def validate_positive_count(value: int, name: str = "count") -> None:
        """Validate count is positive (> 0).

        Convenience method, equivalent to validate_count(..., allow_zero=False).

        Parameters
        ----------
        value : int
            Count value to validate.
        name : str, default="count"
            Name for error messages.

        Raises
        ------
        ValidationError
            If value is not positive.

        Examples
        --------
        >>> CountValidator.validate_positive_count(100)
        >>> CountValidator.validate_positive_count(0)  # Error: must be > 0
        """
        CountValidator.validate_count(value, name=name, allow_zero=False)


class QuantileValidator:
    """Validates quantile-related values."""

    @staticmethod
    def validate_quantile(
        value: float,
        name: str = "quantile",
    ) -> None:
        """Validate quantile value is in [0, 1].

        Parameters
        ----------
        value : float
            Quantile value to validate.
        name : str, default="quantile"
            Name for error messages (e.g., "q25").

        Raises
        ------
        ValidationError
            If value is outside [0, 1].

        Examples
        --------
        >>> QuantileValidator.validate_quantile(0.25)   # OK
        >>> QuantileValidator.validate_quantile(0.75)   # OK
        >>> QuantileValidator.validate_quantile(1.5)    # Error
        """
        RangeValidator.validate_range(
            value=value,
            min_val=0.0,
            max_val=1.0,
            name=name,
        )

    @staticmethod
    def validate_quantile_order(
        q25: float,
        q50: float,
        q75: float,
        *,
        allow_equal: bool = True,
    ) -> None:
        """Validate quantiles are in correct order: q25 <= q50 <= q75.

        Parameters
        ----------
        q25 : float
            25th percentile value.
        q50 : float
            50th percentile (median) value.
        q75 : float
            75th percentile value.
        allow_equal : bool, default=True
            If True, equal values are allowed (<=). If False, requires <.

        Raises
        ------
        ValidationError
            If quantiles are not in correct order.

        Examples
        --------
        >>> QuantileValidator.validate_quantile_order(10, 15, 20)
        >>> QuantileValidator.validate_quantile_order(10, 10, 10)  # OK by default
        >>> QuantileValidator.validate_quantile_order(
        ...     10, 10, 10, allow_equal=False
        ... )  # Error: must be strictly increasing
        """
        if allow_equal:
            if not (q25 <= q50 <= q75):
                raise ValidationError(
                    f"Quantiles not in order: q25={q25} <= q50={q50} <= q75={q75}. "
                    "Expected q25 <= q50 <= q75."
                )
        else:
            if not (q25 < q50 < q75):
                raise ValidationError(
                    f"Quantiles not strictly increasing: q25={q25}, q50={q50}, q75={q75}. "
                    "Expected q25 < q50 < q75."
                )

        logger.debug(f"Quantile order valid: {q25} <= {q50} <= {q75}")
