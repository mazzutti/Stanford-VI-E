"""Central registry for reusable validators.

This module centralizes validators used across multiple config classes,
eliminating duplicate validation code. Uses the Registry pattern to
provide a single source of truth for common validation logic.

Example:
    >>> ValidatorRegistry.validate_angle_range([0, 30, 60, 90])
    >>> ValidatorRegistry.validate_probability(0.75)
"""

from __future__ import annotations
from typing import Callable, Sequence, ClassVar, Dict, Any
import logging

logger = logging.getLogger(__name__)

__all__ = ["ValidatorRegistry"]


class ValidatorRegistry:
    """Central registry for common validation logic.

    Consolidates validators used across multiple config classes,
    eliminating duplicate validation code and providing a single
    point of extension for validation behavior.
    """

    _validators: ClassVar[Dict[str, Callable[..., None]]] = {}
    """Dictionary of registered validators by name"""

    @classmethod
    def register(cls, name: str, validator: Callable[..., None]) -> None:
        """Register a named validator.

        Args:
            name: Unique name for the validator
            validator: Callable that validates and raises ValueError if invalid
        """
        cls._validators[name] = validator
        logger.debug(f"Registered validator: {name}")

    @classmethod
    def get(cls, name: str) -> Callable[..., None]:
        """Get a validator by name.

        Args:
            name: Validator name to retrieve

        Returns:
            Registered validator function

        Raises:
            KeyError: If validator not registered
        """
        if name not in cls._validators:
            raise KeyError(f"Validator '{name}' not registered")
        return cls._validators[name]

    @classmethod
    def validate_angle_range(cls, angles: Sequence[float] | None) -> None:
        """Validate angles are in [0, 90] degrees.

        Args:
            angles: Sequence of angle values

        Raises:
            ValueError: If angles is empty or contains values outside [0, 90]
        """
        if angles is None or len(angles) == 0:
            raise ValueError("angles must not be empty")

        invalid = [a for a in angles if not (0 <= a <= 90)]
        if invalid:
            raise ValueError(f"All angles must be in [0, 90] degrees, got: {invalid}")

    @classmethod
    def validate_probability(cls, prob: float, name: str = "probability") -> None:
        """Validate probability is in [0, 1].

        Args:
            prob: Probability value to validate
            name: Name of field (for error messages)

        Raises:
            ValueError: If probability is not in [0, 1]
        """
        if not (0 <= prob <= 1):
            raise ValueError(f"{name} must be in [0, 1], got {prob}")

    @classmethod
    def validate_positive(cls, value: float, name: str = "value") -> None:
        """Validate value is positive (> 0).

        Args:
            value: Float value to validate
            name: Name of field (for error messages)

        Raises:
            ValueError: If value <= 0
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @classmethod
    def validate_non_negative(cls, value: float, name: str = "value") -> None:
        """Validate value is non-negative (>= 0).

        Args:
            value: Float value to validate
            name: Name of field (for error messages)

        Raises:
            ValueError: If value < 0
        """
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    @classmethod
    def validate_in_range(
        cls, value: float, min_val: float, max_val: float, name: str = "value"
    ) -> None:
        """Validate value is in [min_val, max_val].

        Args:
            value: Float value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            name: Name of field (for error messages)

        Raises:
            ValueError: If value not in range
        """
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")

    @classmethod
    def validate_exclusive_range(
        cls, value: float, min_val: float, max_val: float, name: str = "value"
    ) -> None:
        """Validate value is in (min_val, max_val) - exclusive bounds.

        Args:
            value: Float value to validate
            min_val: Minimum allowed value (exclusive)
            max_val: Maximum allowed value (exclusive)
            name: Name of field (for error messages)

        Raises:
            ValueError: If value not in exclusive range
        """
        if not (min_val < value < max_val):
            raise ValueError(f"{name} must be in ({min_val}, {max_val}), got {value}")

    @classmethod
    def validate_non_empty(cls, seq: Sequence[Any], name: str = "sequence") -> None:
        """Validate sequence is not empty.

        Args:
            seq: Sequence to validate
            name: Name of field (for error messages)

        Raises:
            ValueError: If sequence is empty
        """
        if not seq:
            raise ValueError(f"{name} must not be empty")

    @classmethod
    def validate_length(
        cls, seq: Sequence[Any], expected_len: int, name: str = "sequence"
    ) -> None:
        """Validate sequence has exact length.

        Args:
            seq: Sequence to validate
            expected_len: Expected length
            name: Name of field (for error messages)

        Raises:
            ValueError: If sequence length doesn't match
        """
        if len(seq) != expected_len:
            raise ValueError(f"{name} must have length {expected_len}, got {len(seq)}")

    @classmethod
    def validate_length_between(
        cls, seq: Sequence[Any], min_len: int, max_len: int, name: str = "sequence"
    ) -> None:
        """Validate sequence length is in range.

        Args:
            seq: Sequence to validate
            min_len: Minimum length (inclusive)
            max_len: Maximum length (inclusive)
            name: Name of field (for error messages)

        Raises:
            ValueError: If sequence length not in range
        """
        if not (min_len <= len(seq) <= max_len):
            raise ValueError(
                f"{name} length must be in [{min_len}, {max_len}], " f"got {len(seq)}"
            )
