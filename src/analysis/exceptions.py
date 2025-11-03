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
    ...     logger.error(f"Failed to load cache: {e}")
    ... except CacheSelectionError as e:
    ...     logger.warning(f"Cache file not found: {e}")
"""


class AnalysisException(Exception):
    """Base exception for all analysis module errors.

    All exceptions raised by the analysis module inherit from this class,
    allowing callers to catch all analysis-specific errors with a single
    except clause.
    """

    pass


class CacheError(AnalysisException):
    """Base class for cache-related errors."""

    pass


class CacheLoadingError(CacheError):
    """Raised when cache file cannot be loaded.

    Indicates an error occurred while loading or parsing a cache file,
    such as corrupted data, permission issues, or format errors.

    Examples:
        - File cannot be read (permission denied)
        - File format is corrupted
        - NumPy cannot parse the file
    """

    pass


class CacheSelectionError(CacheError):
    """Raised when no suitable cache file can be found.

    Indicates that the cache selection strategy could not locate a cache
    file matching the requested domain and criteria.

    Examples:
        - Domain has no corresponding cache file
        - Cache directory doesn't exist
        - Search strategy found no matches
    """

    pass


class CacheExtractionError(CacheError):
    """Raised when data cannot be extracted from cache archive.

    Indicates an error occurred while extracting data from an NPZ archive,
    such as missing expected keys or incompatible data formats.

    Examples:
        - Expected 'full_stack' key not found
        - Archive is empty
        - Archive structure is unexpected
    """

    pass


class ValidationError(AnalysisException):
    """Raised when data validation fails.

    Indicates that input data does not meet required criteria or constraints,
    such as correlation values outside [-1, 1] or p-values outside [0, 1].

    Examples:
        - Correlation value > 1.0
        - P-value < 0 or > 1
        - Array shape mismatch
    """

    pass


class DomainError(AnalysisException):
    """Raised for domain-related errors.

    Indicates an error related to seismic domain handling (time/depth),
    such as invalid domain selection or domain conversion failures.

    Examples:
        - Invalid domain specification
        - Unsupported domain type
        - Domain conversion failed
    """

    pass


class ProcessingError(AnalysisException):
    """Raised when data processing fails.

    Indicates an error occurred during analysis processing, such as
    computation failures or incompatible data formats.

    Examples:
        - AVO computation failed
        - Correlation computation failed
        - Gradient analysis failed
    """

    pass


class ConfigurationError(AnalysisException):
    """Raised for configuration-related errors.

    Indicates an error in the configuration of analysis components,
    such as invalid parameters or missing required settings.

    Examples:
        - Invalid cache size
        - Invalid sharding configuration
        - Missing required processor
    """

    pass


def reraise_with_context(
    original_error: Exception,
    error_type: type[AnalysisException],
    message: str,
    *,
    include_cause: bool = True,
) -> AnalysisException:
    """Convert an exception to an AnalysisException with context.

    This utility function helps convert low-level exceptions (e.g., OSError,
    ValueError) into module-specific exceptions while preserving the
    original error information.

    Parameters
    ----------
    original_error : Exception
        The original exception that occurred.
    error_type : type[AnalysisException]
        The target exception type to raise.
    message : str
        A description of what was being attempted when the error occurred.
    include_cause : bool, default=True
        If True, set the original error as __cause__ (enables traceback with `from`).

    Returns
    -------
    AnalysisException
        Configured exception ready to raise.

    Examples
    --------
    >>> try:
    ...     data = np.load("cache.npz")
    ... except (OSError, ValueError) as e:
    ...     raise reraise_with_context(
    ...         e, CacheLoadingError, "loading NPZ archive"
    ...     ) from e
    """
    error = error_type(message)
    if include_cause:
        error.__cause__ = original_error
    return error
