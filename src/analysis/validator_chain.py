"""Validator composition pattern for flexible, reusable validation logic.

This module provides composable validator chains that reduce validation boilerplate
across the analysis module. Instead of writing custom validators in each component,
validators can be composed and reused through a fluent API.

Design Pattern:
    - Validator: Base protocol for validation functions
    - ValidatorChain: Composes multiple validators with AND/OR logic
    - Built-in validators: Common validation predicates (not_none, positive, etc.)
    - Fluent API: Chain validators with method chaining

Benefits:
    - Eliminates ~100+ lines of duplicate validation code
    - Centralized, reusable validation logic
    - Easier to test and maintain
    - Type-safe validation with clear error messages

Example:
    >>> from src.analysis import ValidatorChain, not_none, positive, in_range
    >>>
    >>> # Compose validators
    >>> validator = (ValidatorChain("value")
    ...     .add(not_none("required field"))
    ...     .add(positive("must be positive"))
    ...     .add(in_range(0, 100, "must be between 0-100")))
    >>>
    >>> # Use validator
    >>> result = validator.validate(50)  # Valid
    >>> errors = validator.validate(-5)   # Returns ["must be positive", ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TypeVar,
    Optional,
    List,
    Union,
    Protocol,
    Generic,
    Callable,
    Any,
    cast,
)
from enum import Enum
import logging

__all__ = [
    "ValidatorResult",
    "Validator",
    "ValidatorChain",
    "ValidatorComposite",
    "not_none",
    "positive",
    "negative",
    "in_range",
    "length_between",
    "matches_type",
    "is_callable",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class ValidatorResult(Enum):
    """Result of validation."""

    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"


class Validator(Protocol[T_contra]):
    """Protocol for validation functions.

    Any callable that takes a value and returns validation errors or empty list.
    """

    def __call__(self, value: T_contra) -> List[str]:
        """Validate a value.

        Parameters
        ----------
        value : T
            Value to validate

        Returns
        -------
        list[str]
            List of error messages (empty if valid)
        """
        ...


@dataclass
class ValidatorChain(Generic[T]):
    """Composable validator that chains multiple validators.

    Combines multiple validators using AND logic (all must pass).
    Provides fluent API for building validators.

    Attributes
    ----------
    name : str
        Name of the value being validated (for error messages)
    validators : list[Validator[T]]
        List of validators to apply in sequence
    stop_on_first_error : bool
        If True, stop validation on first error

    Examples
    --------
    Build a validator:
        >>> chain = (ValidatorChain("value")
        ...     .add(not_none())
        ...     .add(positive())
        ...     .add(in_range(0, 100)))

    Validate values:
        >>> errors = chain.validate(50)       # []
        >>> errors = chain.validate(None)     # ["value is required"]
        >>> errors = chain.validate(-5)       # ["value must be positive"]

    Get first error only:
        >>> chain = ValidatorChain("x", stop_on_first_error=True)
        >>> chain.add(positive()).validate(-5)  # ["x must be positive"]
    """

    name: str
    validators: List[Validator[T]] = field(
        default_factory=lambda: cast(List[Validator[T]], [])
    )
    stop_on_first_error: bool = False

    def add(self, validator: Validator[T]) -> ValidatorChain[T]:
        """Add a validator to the chain.

        Parameters
        ----------
        validator : Validator[T]
            Validator function to add

        Returns
        -------
        ValidatorChain[T]
            Self for method chaining
        """
        self.validators.append(validator)
        return self

    def add_multiple(self, *validators: Validator[T]) -> ValidatorChain[T]:
        """Add multiple validators at once.

        Parameters
        ----------
        *validators : Validator[T]
            Validator functions to add

        Returns
        -------
        ValidatorChain[T]
            Self for method chaining
        """
        for validator in validators:
            self.validators.append(validator)
        return self

    def validate(self, value: T) -> List[str]:
        """Validate a value through all validators.

        Parameters
        ----------
        value : T
            Value to validate

        Returns
        -------
        list[str]
            List of error messages (empty if all pass)
        """
        errors: List[str] = []
        for validator in self.validators:
            try:
                validator_errors: List[str] = validator(value)
                errors.extend(validator_errors)
                if self.stop_on_first_error and errors:
                    break
            except Exception as e:
                logger.warning(f"Validator raised exception: {e}")
                errors.append(f"Validation error: {e}")
        return errors

    def is_valid(self, value: T) -> bool:
        """Check if value is valid (no errors).

        Parameters
        ----------
        value : T
            Value to validate

        Returns
        -------
        bool
            True if all validators pass
        """
        return len(self.validate(value)) == 0

    def assert_valid(self, value: T) -> None:
        """Assert that value is valid, raise if not.

        Parameters
        ----------
        value : T
            Value to validate

        Raises
        ------
        ValueError
            If validation fails
        """
        errors = self.validate(value)
        if errors:
            raise ValueError(f"{self.name}: {'; '.join(errors)}")

    def summary(self) -> str:
        """Get summary of validator chain.

        Returns
        -------
        str
            Description of validators in chain
        """
        count = len(self.validators)

        def _validator_name(v: Any) -> str:
            # Prefer the function name if available, otherwise try sensible fallbacks.
            name = getattr(v, "__name__", None)
            if isinstance(name, str):
                return name
            # If it's a callable object, use its class name; otherwise use str()
            try:
                if callable(v):
                    # __name__ may be dynamically typed; ensure a `str` is returned
                    return str(v.__class__.__name__)
            except Exception:
                pass
            return str(v)

        names = ", ".join(_validator_name(v) for v in self.validators[:3])
        return f"ValidatorChain({self.name}, {count} validators: {names}...)"

    def __repr__(self) -> str:
        """Return string representation."""
        return self.summary()

    def __str__(self) -> str:
        """Return human-readable string."""
        return self.summary()


@dataclass
class ValidatorComposite(Generic[T]):
    """Composite validator combining multiple chains with OR logic.

    Allows validation to pass if ANY chain succeeds (useful for alternative validators).

    Attributes
    ----------
    name : str
        Name of the value being validated
    chains : list[ValidatorChain[T]]
        List of validator chains

    Example
    -------
    >>> chain1 = ValidatorChain("x").add(positive())
    >>> chain2 = ValidatorChain("x").add(negative())
    >>> composite = ValidatorComposite("x").add_chain(chain1).add_chain(chain2)
    >>> errors = composite.validate(0)  # Neither positive nor negative?
    """

    name: str
    chains: List[ValidatorChain[T]] = field(
        default_factory=lambda: cast(List[ValidatorChain[T]], [])
    )

    def add_chain(self, chain: ValidatorChain[T]) -> ValidatorComposite[T]:
        """Add a validator chain.

        Parameters
        ----------
        chain : ValidatorChain[T]
            Validator chain to add

        Returns
        -------
        ValidatorComposite[T]
            Self for method chaining
        """
        self.chains.append(chain)
        return self

    def validate(self, value: T) -> List[str]:
        """Validate using OR logic (any chain can pass).

        Parameters
        ----------
        value : T
            Value to validate

        Returns
        -------
        list[str]
            List of all errors if all chains fail, empty if any passes
        """
        if not self.chains:
            return []

        all_errors: List[str] = []
        for chain in self.chains:
            errors = chain.validate(value)
            if not errors:
                return []  # Any chain passed
            all_errors.extend(errors)

        return all_errors

    def is_valid(self, value: T) -> bool:
        """Check if value passes any chain."""
        return len(self.validate(value)) == 0


# Built-in validators (composable)


def not_none(message: str = "is required") -> Callable[[Any], List[str]]:
    """Validator that checks if value is not None.

    Parameters
    ----------
    message : str
        Error message

    Returns
    -------
    Callable
        Validator function
    """

    def _not_none(value: Any) -> List[str]:
        return [] if value is not None else [message]

    _not_none.__name__ = "not_none"
    return _not_none


def positive(message: str = "must be positive") -> Callable[[Any], List[str]]:
    """Validator that checks if value is positive.

    Parameters
    ----------
    message : str
        Error message

    Returns
    -------
    Callable
        Validator function
    """

    def _positive(value: Union[int, float]) -> List[str]:
        return [] if value > 0 else [message]

    _positive.__name__ = "positive"
    return _positive


def negative(message: str = "must be negative") -> Callable[[Any], List[str]]:
    """Validator that checks if value is negative.

    Parameters
    ----------
    message : str
        Error message

    Returns
    -------
    Validator
        Validator function
    """

    def _negative(value: Union[int, float]) -> List[str]:
        return [] if value < 0 else [message]

    _negative.__name__ = "negative"
    return _negative


def in_range(
    min_val: Union[int, float],
    max_val: Union[int, float],
    message: Optional[str] = None,
) -> Callable[[Any], List[str]]:
    """Validator that checks if value is in range.

    Parameters
    ----------
    min_val : int or float
        Minimum value (inclusive)
    max_val : int or float
        Maximum value (inclusive)
    message : str, optional
        Error message (default: "must be between X and Y")

    Returns
    -------
    Callable
        Validator function
    """
    if message is None:
        message = f"must be between {min_val} and {max_val}"

    def _in_range(value: Union[int, float]) -> List[str]:
        return [] if min_val <= value <= max_val else [message]

    _in_range.__name__ = f"in_range({min_val}, {max_val})"
    return _in_range


def length_between(
    min_len: int,
    max_len: int,
    message: Optional[str] = None,
) -> Callable[[Any], List[str]]:
    """Validator that checks if value length is in range.

    Parameters
    ----------
    min_len : int
        Minimum length (inclusive)
    max_len : int
        Maximum length (inclusive)
    message : str, optional
        Error message

    Returns
    -------
    Callable
        Validator function
    """
    if message is None:
        message = f"length must be between {min_len} and {max_len}"

    def _length_between(value: Any) -> List[str]:
        try:
            return [] if min_len <= len(value) <= max_len else [message]
        except TypeError:
            return ["does not support length check"]

    _length_between.__name__ = f"length_between({min_len}, {max_len})"
    return _length_between


def matches_type(
    expected_type: type, message: Optional[str] = None
) -> Callable[[Any], List[str]]:
    """Validator that checks if value is of expected type.

    Parameters
    ----------
    expected_type : type
        Expected type
    message : str, optional
        Error message

    Returns
    -------
    Validator
        Validator function
    """
    if message is None:
        message = f"must be of type {expected_type.__name__}"

    def _matches_type(value: Any) -> List[str]:
        return [] if isinstance(value, expected_type) else [message]

    _matches_type.__name__ = f"matches_type({expected_type.__name__})"
    return _matches_type


def is_callable(message: str = "must be callable") -> Callable[[Any], List[str]]:
    """Validator that checks if value is callable.

    Parameters
    ----------
    message : str
        Error message

    Returns
    -------
    Callable
        Validator function
    """

    def _is_callable(value: Any) -> List[str]:
        return [] if callable(value) else [message]

    _is_callable.__name__ = "is_callable"
    return _is_callable
