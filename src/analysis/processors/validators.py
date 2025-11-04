"""Validation protocols and utilities for processor inputs."""

import logging
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple, TypeVar, TYPE_CHECKING

from numpy.typing import NDArray

from .exceptions import ValidationError
from .types import ArrayNamePair

if TYPE_CHECKING:
    from src.analysis.domain.enum import Domain

logger = logging.getLogger(__name__)

__all__ = [
    "Validatable",
    "ValidationHelpers",
    "ArrayValidator",
    "DomainValidator",
    "PathValidator",
    "_ValidationErrors",
]

T = TypeVar("T")  # TypeVar for generic return type in validate_or_return


class Validatable(Protocol):
    """Protocol for objects that can be validated.

    Defines a contract for objects that implement validation logic,
    enabling structural typing for validation-aware code.
    """

    def validate(self) -> bool:
        """Validate the object's state. Returns True if valid, False otherwise."""
        ...

    def assert_valid(self) -> None:
        """Assert object is valid, raise ValidationError if not."""
        ...


class ValidationHelpers:
    """Helper class for common validation patterns.

    Consolidates repeated validation logic to follow DRY principle
    and reduce boilerplate in validation code.
    """

    @staticmethod
    def validate_or_return(
        condition: bool,
        error_msg: str,
        default_value: Optional[T] = None,
        log_level: str = "warning",
    ) -> Optional[T]:
        """Validate condition or return default with logging.

        Parameters
        ----------
        condition : bool
            Condition to validate.
        error_msg : str
            Error message to log if validation fails.
        default_value : T, optional
            Default value to return on validation failure.
        log_level : str, optional
            Logging level ('warning', 'error', 'debug'). Default: 'warning'.

        Returns
        -------
        T or None
            Returns default_value if condition is False, None otherwise.
        """
        if not condition:
            log_func = getattr(logger, log_level, logger.warning)
            log_func(error_msg)
            return default_value
        return None

    @staticmethod
    def ensure_valid_arrays(
        *arrays_with_names: ArrayNamePair,
    ) -> None:
        """Ensure all arrays are valid 3D arrays.

        Parameters
        ----------
        *arrays_with_names
            Variable number of (array, name) tuples to validate.

        Raises
        ------
        ValidationError
            If any array is not 3D or is empty.
        """
        for arr, name in arrays_with_names:
            if arr.ndim != 3:
                raise ValidationError(
                    f"Array '{name}' must be 3D, got {arr.ndim}D with shape {arr.shape}"
                )
            if arr.size == 0:
                raise ValidationError(
                    f"Array '{name}' cannot be empty (shape: {arr.shape})"
                )


class _ValidationErrors:
    """Centralized validation error messages for consistency and maintainability."""

    @staticmethod
    def invalid_dimensions(
        name: str, actual_ndim: int, actual_shape: tuple[int, ...]
    ) -> str:
        """Error message for invalid array dimensions."""
        return (
            f"Array '{name}' must be 3-dimensional, got {actual_ndim}D with shape {actual_shape}. "
            f"Suggested fix: ensure input is a cube (i, j, k) not an image or vector."
        )

    @staticmethod
    def empty_array(name: str, shape: tuple[int, ...]) -> str:
        """Error message for empty array."""
        return (
            f"Array '{name}' cannot be empty (shape: {shape}). "
            f"Suggested fix: verify input data contains valid measurements."
        )

    @staticmethod
    def invalid_parameter(param_name: str, value: int) -> str:
        """Error message for invalid parameter value."""
        return (
            f"Parameter '{param_name}' must be non-negative, got {value}. "
            f"Suggested fix: use a positive integer or zero."
        )


class ArrayValidator:
    """Centralized validation logic for array inputs to processors.

    Provides consistent, reusable validation for 3D cube arrays and parameters.
    """

    @staticmethod
    def validate_3d_array(arr: NDArray[Any], name: str = "array") -> None:
        """Validate that input is a non-empty 3D numpy array.

        Shared validation logic for 3D cube inputs. Provides consistent error
        messages across all processor methods.

        Parameters
        ----------
        arr : numpy.ndarray
            Array to validate.
        name : str, optional
            Name of the array for error messages (default: "array").

        Raises
        ------
        TypeError
            If array is not a numpy array.
        ValueError
            If array is not 3D or is empty.

        Examples
        --------
        >>> ArrayValidator.validate_3d_array(np.zeros((10, 10, 20)), "seismic_cube")
        """
        import numpy as np

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
        if arr.ndim != 3:
            raise ValueError(
                _ValidationErrors.invalid_dimensions(name, arr.ndim, arr.shape)
            )
        if arr.size == 0:
            raise ValueError(_ValidationErrors.empty_array(name, arr.shape))

    @staticmethod
    def validate_3d_arrays(*arrays_with_names: ArrayNamePair) -> None:
        """Validate multiple 3D arrays efficiently in a single call.

        Consolidates repeated validate_3d_array() calls to reduce boilerplate
        when validating multiple cube inputs.

        Parameters
        ----------
        *arrays_with_names
            Variable number of (array, name) tuples to validate.

        Raises
        ------
        ValueError
            If any array is not 3D or is empty.

        Examples
        --------
        >>> ArrayValidator.validate_3d_arrays(
        ...     (seismic_cube, "seismic_cube"),
        ...     (facies_cube, "facies_cube")
        ... )
        """
        for arr, name in arrays_with_names:
            ArrayValidator.validate_3d_array(arr, name)

    @staticmethod
    def validate_positive_parameter(value: int, param_name: str) -> None:
        """Validate that a parameter is a non-negative integer.

        Helper for validating dilation window and window size parameters.

        Parameters
        ----------
        value : int
            Parameter value to validate.
        param_name : str
            Name of the parameter for error messages.

        Raises
        ------
        ValueError
            If value is negative. Error includes suggestion to fix the issue.

        Examples
        --------
        >>> ArrayValidator.validate_positive_parameter(2, "dilation_window")
        >>> ArrayValidator.validate_positive_parameter(-1, "window_size")  # Raises ValueError
        """
        if value < 0:
            raise ValueError(_ValidationErrors.invalid_parameter(param_name, value))


class DomainValidator:
    """Centralized validation for Domain enums.

    Provides consistent validation for Domain enum values used throughout
    the analysis pipeline. This consolidates domain validation logic and
    eliminates duplication across modules.
    """

    VALID_DOMAINS: frozenset[str] = frozenset(["depth", "time"])
    """Set of valid domain string values."""

    @staticmethod
    def validate_domain(
        domain: "Domain", valid_domains: Optional[set["Domain"]] = None
    ) -> "Domain":
        """Validate and return a Domain enum value.

        Validates that the provided domain is a supported value. Accepts either
        a Domain enum or delegates to the Domain enum's validation.

        Parameters
        ----------
        domain : Domain
            Domain enum instance to validate (e.g., Domain.DEPTH, Domain.TIME).
        valid_domains : set, optional
            Set of valid Domain values. If None, uses standard DEPTH and TIME.

        Returns
        -------
        Domain
            The validated domain enum.

        Raises
        ------
        TypeError
            If domain is not a Domain enum instance.
        ValueError
            If domain is not in the set of valid domains.

        Examples
        --------
        >>> from src.analysis.domain.enum import Domain
        >>> DomainValidator.validate_domain(Domain.DEPTH)
        <Domain.DEPTH: 'depth'>

        >>> DomainValidator.validate_domain("depth")  # Raises TypeError
        TypeError: Expected Domain enum, got str...
        """
        # Import here to avoid circular dependencies
        from src.analysis.domain.enum import Domain

        if not isinstance(domain, Domain):
            raise TypeError(
                f"Expected Domain enum, got {type(domain).__name__}. "
                f"Use Domain.DEPTH or Domain.TIME"
            )

        if valid_domains is None:
            valid_domains = {Domain.DEPTH, Domain.TIME}

        if domain not in valid_domains:
            domain_names = ", ".join(d.value for d in valid_domains)
            raise ValueError(
                f"Unsupported domain: {domain}. Valid options: {domain_names}"
            )

        return domain


class PathValidator:
    """Centralized validation for file paths and directories.

    Provides consistent path validation across the analysis module,
    consolidating repeated path validation logic.
    """

    @staticmethod
    def validate_cache_dir(cache_dir: str) -> Path:
        """Validate and return cache directory as a Path object.

        Validates that cache_dir is a non-empty string and returns it as
        a Path object for consistent path handling.

        Parameters
        ----------
        cache_dir : str
            Path to cache directory (can be relative or absolute).

        Returns
        -------
        Path
            Validated Path object.

        Raises
        ------
        ValueError
            If cache_dir is empty, not a string, or whitespace-only.

        Examples
        --------
        >>> PathValidator.validate_cache_dir(".cache")
        PosixPath('.cache')

        >>> PathValidator.validate_cache_dir("")  # Raises ValueError
        ValueError: cache_dir must be a non-empty string...
        """
        if not isinstance(cache_dir, str) or not cache_dir.strip():
            raise ValueError(
                f"cache_dir must be a non-empty string, got: {repr(cache_dir)}"
            )
        return Path(cache_dir)
