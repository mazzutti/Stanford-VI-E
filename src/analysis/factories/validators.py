"""Type validation utilities for builder pattern.

This module provides type validation helpers that handle regular types.

Exception classes (BuilderValidationError, BuilderFrozenError) have been moved
to src.analysis.exceptions for centralized exception handling.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class TypeValidator:
    """Validates values against expected types.

    This class handles validation of regular types.

    Example Usage:
        >>> validator = TypeValidator()
        >>> validator.validate(42, int, "count")  # passes
        >>> validator.validate(None, int, "count")  # passes (None is always OK)
        >>> validator.validate("not int", int, "count")  # raises TypeError
    """

    @staticmethod
    def validate(value: object, expected_type: object, field_name: str) -> None:
        """Validate value against expected type.

        Parameters
        ----------
        value
            The value to validate.
        expected_type
            The expected type.
        field_name
            Name of the field being validated (for error messages).

        Raises
        ------
        TypeError
            If value doesn't match expected_type.
        """
        if value is None:
            return  # None is always acceptable

        # Handle Callable type specially
        if expected_type is Callable or expected_type == Callable:
            if not callable(value):
                raise TypeError(
                    f"Expected callable for '{field_name}', got {type(value).__name__}"
                )
            return

        # Handle type check
        if expected_type is type:
            if not isinstance(value, type):
                raise TypeError(
                    f"Expected type for '{field_name}', got {type(value).__name__}"
                )
            return

        # For regular types, use isinstance
        if isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected {expected_type.__name__} for '{field_name}', "
                    f"got {type(value).__name__}"
                )
            return
