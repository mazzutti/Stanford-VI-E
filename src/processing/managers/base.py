"""Base manager class with common functionality."""

from abc import abstractmethod
import logging
from pathlib import Path
from typing import Any


from src.processing.core.abstracts import Manager


__all__ = ["BaseManager"]


class BaseManager(Manager):
    """Base class for all managers with common functionality.

    Provides consistent initialization, logging, and error handling.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize manager.

        Args:
            logger: Optional logger instance, defaults to class logger
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def clear(
        self,
        patterns: list[str] | None = None,
        cache_dir: Path | None = None,
        prefix: str = "",
    ) -> int:
        """Clear managed resources."""
        pass

    @abstractmethod
    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: list[str] | None = None,
        prefix: str = "",
    ) -> None:
        """Print summary of managed resources."""
        pass

    def _log_info(self, message: str, *args: Any) -> None:
        """Log info message."""
        self.logger.info(message, *args)

    def _log_warning(self, message: str, *args: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, *args)

    def _log_error(self, message: str, *args: Any) -> None:
        """Log error message."""
        self.logger.error(message, *args)
