"""Core exceptions for the processing module.

Custom exception hierarchy for processing-related errors.
"""

__all__ = [
    "ProcessingError",
    "ResamplingError",
    "ValidationError",
    "CacheError",
    "ConfigurationError",
]

class ProcessingError(Exception):
    """Base exception for processing module errors."""

class ResamplingError(ProcessingError):
    """Errors during resampling operations."""

class ValidationError(ProcessingError):
    """Data or model validation errors."""

class CacheError(ProcessingError):
    """Errors in cache operations."""

class ConfigurationError(ProcessingError):
    """Configuration or initialization errors."""
