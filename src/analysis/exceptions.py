"""Exception hierarchy for analysis module.

This module provides structured exception types for the analysis package,
enabling more precise error handling and better error reporting throughout
the codebase.

Exception Hierarchy:
    AnalysisException (base)
    ├── CacheError
    │   ├── CacheLoadingError
    │   ├── CacheSelectionError
    │   └── CacheExtractionError
    ├── ValidationError
    ├── DomainError
    └── ProcessingError

Usage:
    >>> try:
    ...     loader.load_full_stack(path)
    ... except CacheLoadingError as e:
    ...     logger.error("Failed to load cache: %s", e)
    ... except CacheSelectionError as e:
    ...     logger.warning("Cache file not found: %s", e)
"""


class AnalysisException(Exception):
    """Base exception for all analysis module errors.

    All exceptions raised by the analysis module inherit from this class,
    allowing callers to catch all analysis-specific errors with a single
    except clause.
    """


class CacheError(AnalysisException):
    """Base class for cache-related errors."""


class CacheLoadingError(CacheError):
    """Raised when cache file cannot be loaded.

    Indicates an error occurred while loading or parsing a cache file,
    such as corrupted data, permission issues, or format errors.

    Examples:
        - File cannot be read (permission denied)
        - File format is corrupted
        - NumPy cannot parse the file
    """


class CacheSelectionError(CacheError):
    """Raised when no suitable cache file can be found.

    Indicates that the cache selection strategy could not locate a cache
    file matching the requested domain and criteria.

    Examples:
        - Domain has no corresponding cache file
        - Cache directory doesn't exist
        - Search strategy found no matches
    """


class CacheExtractionError(CacheError):
    """Raised when data cannot be extracted from cache archive.

    Indicates an error occurred while extracting data from an NPZ archive,
    such as missing expected keys or incompatible data formats.

    Examples:
        - Expected 'full_stack' key not found
        - Archive is empty
        - Archive structure is unexpected
    """


class ValidationError(AnalysisException):
    """Raised when data validation fails.

    Indicates that input data does not meet required criteria or constraints,
    such as correlation values outside [-1, 1] or p-values outside [0, 1].

    Examples:
        - Correlation value > 1.0
        - P-value < 0 or > 1
        - Array shape mismatch
    """


class DomainError(AnalysisException):
    """Raised for domain-related errors.

    Indicates an error related to seismic domain handling (time/depth),
    such as invalid domain selection or domain conversion failures.

    Examples:
        - Invalid domain specification
        - Unsupported domain type
        - Domain conversion failed
    """


class ProcessingError(AnalysisException):
    """Raised when data processing fails.

    Indicates an error occurred during analysis processing, such as
    computation failures or incompatible data formats.

    Examples:
        - AVO computation failed
        - Correlation computation failed
        - Gradient analysis failed
    """


class ConfigurationError(AnalysisException):
    """Raised for configuration-related errors.

    Indicates an error in the configuration of analysis components,
    such as invalid parameters or missing required settings.

    Examples:
        - Invalid cache size
        - Invalid sharding configuration
        - Missing required processor
    """


class BuilderValidationError(ValueError):
    """Raised when builder validation fails.

    Indicates that a value provided to the builder does not meet
    type or constraint requirements.

    Examples:
        - Incorrect type for processor
        - Invalid configuration parameter
        - Missing required dependency
    """

    def __init__(self, message: str, missing_deps: list[str] | None = None) -> None:
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
    """Raised when attempting to modify a frozen builder.

    Indicates an attempt to modify a builder after it has been
    frozen (locked) to prevent further changes.

    Examples:
        - Setting processor after freeze()
        - Setting config after freeze()
    """


class ComputationError(AnalysisException):
    """Raised when computational operations fail.

    Base class for computation-related errors (alignment, detection,
    extraction, interpolation).

    Examples:
        - Alignment quality insufficient
        - Feature detection failed
        - Amplitude extraction failed
        - Interpolation produced invalid values
    """


class AlignmentError(ComputationError):
    """Raised when cube alignment fails."""


class DetectionError(ComputationError):
    """Raised when boundary detection fails."""


class ExtractionError(ComputationError):
    """Raised when amplitude extraction fails."""


class InterpolationError(ComputationError):
    """Raised when interpolation fails."""


class StateError(AnalysisException):
    """Raised when object state is invalid for operation.

    Indicates an operation was attempted on an object in an invalid state,
    or an invalid state transition was requested.

    Examples:
        - Processor not initialized
        - Invalid state transition
        - Required setup not completed
    """


class ExceptionContextBuilder:
    """Builder for creating AnalysisExceptions with proper error context.

    This class helps convert low-level exceptions (e.g., OSError, ValueError)
    into module-specific exceptions while preserving the original error
    information and maintaining proper exception chaining.

    Example Usage:
        >>> try:
        ...     data = np.load("cache.npz")
        ... except (OSError, ValueError) as e:
        ...     raise ExceptionContextBuilder(e).build(
        ...         CacheLoadingError, "loading NPZ archive"
        ...     )
    """

    def __init__(self, original_error: Exception) -> None:
        """Initialize with the original exception.

        Parameters
        ----------
        original_error : Exception
            The exception that occurred and should be contextualized.
        """
        self._original_error = original_error
        self._include_cause = True

    def with_cause(self, include_cause: bool) -> "ExceptionContextBuilder":
        """Configure whether to attach original error as __cause__.

        Parameters
        ----------
        include_cause : bool
            If True, sets original error as __cause__ for exception chaining.

        Returns
        -------
        ExceptionContextBuilder
            Self for method chaining.
        """
        self._include_cause = include_cause
        return self

    def build(
        self,
        error_type: type[AnalysisException],
        message: str,
    ) -> AnalysisException:
        """Build and return the contextualized exception.

        Parameters
        ----------
        error_type : type[AnalysisException]
            The target exception type to instantiate.
        message : str
            A description of what was being attempted when the error occurred.

        Returns
        -------
        AnalysisException
            Configured exception ready to raise.
        """
        error = error_type(message)
        if self._include_cause:
            error.__cause__ = self._original_error
        return error
