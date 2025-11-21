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
from abc import ABC, abstractmethod
from typing import Any

from src.analysis.exceptions import ValidationError
from src.analysis.validation_result import ValidationResult
from src.core import validation as core_validation

logger = logging.getLogger(__name__)

__all__ = [
    "Validator",
    "RangeValidator",
    "CountValidator",
    "QuantileValidator",
    "ValidatorStrategy",
    "CompositeValidator",
    "ValidationError",
]


class Validator(ABC):
    """Abstract base class for all data validators.

    Defines the common interface for validators and provides shared
    validation infrastructure. All concrete validators should inherit
    from this class and implement the validate() method.

    This ensures consistent validation patterns across the codebase and
    makes it easy to extend validation capabilities.
    """

    @abstractmethod
    def validate(self, value: Any, name: str = "value") -> None:
        """Validate a value and raise ValidationError if invalid.

        Parameters
        ----------
        value
            The value to validate (type depends on validator).
        name : str, default="value"
            Name for error messages.
        **kwargs
            Additional validation parameters (validator-specific).

        Raises
        ------
        ValidationError
            If value is invalid according to validator rules.
        """
        raise NotImplementedError()

    @staticmethod
    def _format_error_message(
        name: str,
        actual: Any,
        requirement: str,
    ) -> str:
        """Format a consistent validation error message.

        Parameters
        ----------
        name : str
            Field/value name.
        actual : any
            The actual value that failed validation.
        requirement : str
            Description of what was required.

        Returns
        -------
        str
            Formatted error message.
        """
        return f"Invalid {name}: {actual} (requirement: {requirement})"

    # This module focuses on clear validation primitives; small helper classes
    # are intentionally compact to reduce boilerplate across analyzers.


class RangeValidator(Validator):
    """Validates numeric values fall within expected ranges.

    This implementation delegates to ``src.core.validation.RangeValidator``
    to avoid duplicating logic. Any ``core_validation.ValidationError`` is
    translated into the analysis-level ``ValidationError``.
    """

    def validate(
        self,
        value: float,
        name: str = "value",
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """Validate a numeric value is within specified range (delegate)."""
        actual_min = min_val if min_val is not None else float("-inf")
        actual_max = max_val if max_val is not None else float("inf")
        try:
            core_validation.RangeValidator.validate_range(
                float(value), actual_min, actual_max, name
            )
        except core_validation.ValidationError as exc:  # pragma: no cover - translation
            raise ValidationError(str(exc)) from exc

    @staticmethod
    def validate_correlation(
        value: float,
        name: str = "correlation",
        *,
        allow_nan: bool = False,
    ) -> None:
        """Validate a correlation coefficient is within [-1, 1]."""
        try:
            core_validation.RangeValidator.validate_correlation(
                value, name=name, allow_nan=allow_nan
            )
        except core_validation.ValidationError as exc:  # pragma: no cover
            raise ValidationError(str(exc)) from exc

    @staticmethod
    def validate_pvalue(
        value: float, name: str = "p_value", *, allow_nan: bool = False
    ) -> None:
        """Validate a p-value is within [0, 1]."""
        try:
            core_validation.RangeValidator.validate_pvalue(
                value, name=name, allow_nan=allow_nan
            )
        except core_validation.ValidationError as exc:  # pragma: no cover
            raise ValidationError(str(exc)) from exc

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
        # This helper intentionally accepts multiple configuration flags
        # (allow_nan, include_endpoints). Keep the explicit signature for
        # clarity and silence pylint's argument-count warning for this
        # delegating adaptor.

        """Validate a numeric value falls between ``min_val`` and ``max_val``.

        This delegates to the core validation implementation and translates
        any core-level ValidationError into the analysis-level
        ``ValidationError``.
        """
        try:
            core_validation.RangeValidator.validate_range(
                value,
                min_val,
                max_val,
                name,
                allow_nan=allow_nan,
                include_endpoints=include_endpoints,
            )
        except core_validation.ValidationError as exc:  # pragma: no cover
            raise ValidationError(str(exc)) from exc

    @staticmethod
    def validate_probability(
        value: float, name: str = "probability", *, allow_nan: bool = False
    ) -> None:
        """Validate a probability value is within [0, 1]."""
        try:
            core_validation.RangeValidator.validate_probability(
                value, name=name, allow_nan=allow_nan
            )
        except core_validation.ValidationError as exc:  # pragma: no cover
            raise ValidationError(str(exc)) from exc


class CountValidator(Validator):
    """Validates count-like values (non-negative integers)."""

    def validate(
        self,
        value: int,
        name: str = "value",
        allow_zero: bool = True,
    ) -> None:
        """Validate a count value (non-negative integer).

        Parameters
        ----------
        value : int
            Count value to validate.
        name : str, default="value"
            Name for error messages.
        allow_zero : bool, default=True
            If True, zero is accepted. If False, value must be > 0.

        Raises
        ------
        ValidationError
            If value is negative or is zero when not allowed.
        """
        self.validate_count(value, name, allow_zero=allow_zero)

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
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")

        if value == 0 and not allow_zero:
            raise ValidationError(f"{name} must be greater than 0, got {value}")

        logger.debug("%s=%s is valid", name, value)

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


class QuantileValidator(Validator):
    """Validates quantile-related values."""

    def validate(
        self,
        value: float,
        name: str = "value",
    ) -> None:
        """Validate a quantile value [0, 1].

        Parameters
        ----------
        value : float
            Quantile value to validate.
        name : str, default="value"
            Name for error messages.

        Raises
        ------
        ValidationError
            If value is outside [0, 1].
        """
        self.validate_quantile(value, name)

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
            if not q25 <= q50 <= q75:
                raise ValidationError(
                    f"Quantiles not in order: q25={q25} <= q50={q50} <= q75={q75}. "
                    "Expected q25 <= q50 <= q75."
                )
        else:
            if not q25 < q50 < q75:
                raise ValidationError(
                    f"Quantiles not strictly increasing: q25={q25}, q50={q50}, q75={q75}. "
                    "Expected q25 < q50 < q75."
                )

        logger.debug("Quantile order valid: %s <= %s <= %s", q25, q50, q75)


# ============================================================================
# Strategy Pattern Validators (composable validation chains)
# ============================================================================


class ValidatorStrategy(ABC):
    """Abstract base for validation logic.

    Validators implement specific validation rules that can be composed
    into validation pipelines through the Strategy pattern.

    Examples
    --------
    A custom validator:

    >>> class PositiveValidator(ValidatorStrategy):
    ...     def validate(self, data):
    ...         if (data < 0).any():
    ...             from src.analysis.processors.config import ValidationResult
    ...             return ValidationResult(
    ...                 is_valid=False,
    ...                 error_message="Data contains negative values"
    ...             )
    ...         return ValidationResult(is_valid=True)
    ...
    ...     def describe(self) -> str:
    ...         return "All values must be positive"
    """

    @abstractmethod
    def validate(self, data: Any) -> Any:
        """Execute validation logic.

        Parameters
        ----------
        data
            Data to validate.

        Returns
        -------
        ValidationResult
            Validation result with success status and error message.
        """

    @abstractmethod
    def describe(self) -> str:
        """Get human-readable description of this validation.

        Returns
        -------
        str
            Description suitable for logging or documentation.
        """


class CompositeValidator(ValidatorStrategy):
    """Compose multiple validators in a validation pipeline.

    Allows combining multiple validators with AND/OR logic to build
    flexible validation chains. All validators are executed and results
    are combined according to the composition mode.

    Parameters
    ----------
    *validators
        Variable number of validators to compose.
    mode
        How to combine results: 'all' (default) requires all validators pass,
        'any' requires at least one validator pass.

    Examples
    --------
    Chain multiple validators:

    >>> pipeline = CompositeValidator(
    ...     ShapeValidator(expected_shape=(10, 10)),
    ...     RangeValidator(min=0, max=100),
    ...     NonNaNValidator(),
    ...     mode='all'
    ... )
    >>> result = pipeline.validate(data)
    >>> if result.is_valid:
    ...     print("All validations passed")

    Alternative validators:

    >>> backup_pipeline = CompositeValidator(
    ...     FastValidator(),
    ...     ThoroughValidator(),
    ...     mode='any'
    ... )
    """

    def __init__(
        self,
        *validators: ValidatorStrategy,
        mode: str = "all",
    ) -> None:
        """Initialize composite validator.

        Parameters
        ----------
        validators
            Validators to compose.
        mode
            'all' (AND logic) or 'any' (OR logic).

        Raises
        ------
        ValueError
            If mode is not 'all' or 'any'.
        """
        if mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', got {mode}")
        if not validators:
            raise ValueError("At least one validator must be provided")

        self.validators = list(validators)
        self.mode = mode

    def validate(self, data: Any) -> Any:
        """Execute all validators and combine results.

        Parameters
        ----------
        data
            Data to validate.

        Returns
        -------
        ValidationResult
            Combined validation result.
        """

        results = [v.validate(data) for v in self.validators]

        if self.mode == "all":
            return self._combine_all(results)
        return self._combine_any(results)

    def _combine_all(self, results: Any) -> Any:
        """Combine results with AND logic (all must pass).

        Returns failure on first failed validation.
        """

        for result in results:
            if not result.is_valid:
                return result

        # All passed - combine data and counts
        return ValidationResult(
            is_valid=True,
            arr1=results[0].arr1 if results else None,
            arr2=results[0].arr2 if results else None,
            n_removed=sum(r.n_removed for r in results),
        )

    def _combine_any(self, results: Any) -> Any:
        """Combine results with OR logic (at least one must pass).

        Returns success if any validation passes.
        """

        for result in results:
            if result.is_valid:
                return result

        # All failed - return first failure
        return results[0] if results else ValidationResult(is_valid=False)

    def describe(self) -> str:
        """Get description of all validators."""
        mode_word = "All of" if self.mode == "all" else "Any of"
        descriptions = [f"  - {v.describe()}" for v in self.validators]
        return f"{mode_word}:\n" + "\n".join(descriptions)
