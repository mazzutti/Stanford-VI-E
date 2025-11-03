"""Type validation utilities for builder pattern.

This module provides type validation helpers that handle both regular types
and Protocol types with duck-typing fallback.
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def validate_type(value: object, expected_type: object, field_name: str) -> None:
    """Validate value against expected type with proper Protocol handling.

    This function handles validation of regular types and Protocol types.
    For Protocol types, it uses duck-typing validation via hasattr checks
    instead of isinstance, since Protocols don't support isinstance directly.

    Parameters
    ----------
    value
        The value to validate.
    expected_type
        The expected type (could be a regular type or Protocol).
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
        try:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected {expected_type.__name__} for '{field_name}', "
                    f"got {type(value).__name__}"
                )
            return
        except TypeError:
            # Fallback for edge cases
            logger.warning(
                f"Could not validate type for '{field_name}' against "
                f"{expected_type}. Using duck-typing validation."
            )

    # For Protocol types (which don't work with isinstance)
    # Use duck-typing: check if the value has the expected attributes/methods
    try:
        # Try to get the Protocol's expected attributes
        if hasattr(expected_type, "__protocol_attrs__"):
            required_attrs = getattr(expected_type, "__protocol_attrs__", [])
            missing_attrs = [
                attr for attr in required_attrs if not hasattr(value, attr)
            ]
            if missing_attrs:
                logger.warning(
                    f"Value for '{field_name}' may not implement all Protocol methods: "
                    f"missing {missing_attrs}"
                )
        else:
            # Generic warning for unknown Protocol-like types
            logger.debug(
                f"Skipping strict validation for '{field_name}' "
                f"(assumed Protocol type)"
            )
    except Exception as e:
        logger.debug(f"Could not validate Protocol type for '{field_name}': {e}")


class BuilderValidationError(ValueError):
    """Raised when builder validation fails with specific details."""

    def __init__(self, message: str, missing_deps: Optional[list[str]] = None) -> None:
        """Initialize validation error.

        Parameters
        ----------
        message
            Error message describing what failed.
        missing_deps
            List of missing dependencies (optional).
        """
        super().__init__(message)
        self.missing_deps = missing_deps or []


class BuilderFrozenError(RuntimeError):
    """Raised when attempting to modify a frozen builder."""

    pass
