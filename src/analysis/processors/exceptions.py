"""Custom exceptions for processor operations."""


class ProcessorError(Exception):
    """Base exception for processor operations."""

    pass


class ValidationError(ProcessorError):
    """Raised when array validation fails."""

    pass


class CorrelationError(ProcessorError):
    """Raised when correlation computation fails."""

    pass


class ReshapeError(ProcessorError):
    """Raised when array reshaping fails."""

    pass


__all__ = [
    "ProcessorError",
    "ValidationError",
    "CorrelationError",
    "ReshapeError",
]
