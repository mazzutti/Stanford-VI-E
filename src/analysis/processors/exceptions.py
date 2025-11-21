"""Custom exceptions for processor operations."""


class ProcessorError(Exception):
    """Base exception for processor operations."""


class ValidationError(ProcessorError):
    """Raised when array validation fails."""


class CorrelationError(ProcessorError):
    """Raised when correlation computation fails."""


class ReshapeError(ProcessorError):
    """Raised when array reshaping fails."""


__all__ = [
    "ProcessorError",
    "ValidationError",
    "CorrelationError",
    "ReshapeError",
]
