"""Generic resource manager with strategy pattern for operations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, Protocol, Optional, List
import logging

from src.processing.managers.base import BaseManager

__all__ = [
    "ClearStrategy",
    "SummarizeStrategy",
    "NoOpClearStrategy",
    "NoOpSummarizeStrategy",
    "ResourceManager",
]

T = TypeVar("T")


class ClearStrategy(Protocol):
    """Strategy for clearing resources."""

    def clear(self, resource_dir: Path, patterns: Optional[List[str]] = None) -> int:
        """Clear resources matching criteria.

        Args:
            resource_dir: Directory containing resources
            patterns: Optional patterns to match

        Returns:
            Number of resources cleared
        """
        ...


class SummarizeStrategy(Protocol):
    """Strategy for summarizing resources."""

    def summarize(self, resource_dir: Path, keys: Optional[List[str]] = None) -> None:
        """Print summary of resources.

        Args:
            resource_dir: Directory containing resources
            keys: Optional keys to filter by
        """
        ...


class NoOpClearStrategy:
    """No-operation clear strategy."""

    def clear(self, resource_dir: Path, patterns: Optional[List[str]] = None) -> int:
        """Do nothing."""
        return 0


class NoOpSummarizeStrategy:
    """No-operation summarize strategy."""

    def summarize(self, resource_dir: Path, keys: Optional[List[str]] = None) -> None:
        """Do nothing."""
        pass


class ResourceManager(BaseManager, Generic[T]):
    """Generic resource manager with pluggable strategies.

    Eliminates duplicate code across CacheManager, FileManager, etc.
    by using composition and strategy pattern.
    """

    def __init__(
        self,
        resource_dir: Path,
        clear_strategy: Optional[ClearStrategy] = None,
        summarize_strategy: Optional[SummarizeStrategy] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize resource manager.

        Args:
            resource_dir: Base directory for resources
            clear_strategy: Strategy for clearing operations
            summarize_strategy: Strategy for summarizing operations
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.resource_dir = resource_dir
        self._clear_strategy = clear_strategy or NoOpClearStrategy()
        self._summarize_strategy = summarize_strategy or NoOpSummarizeStrategy()

    def clear(
        self,
        patterns: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        """Clear resources using configured strategy.

        Args:
            patterns: Optional patterns to match
            cache_dir: Override resource directory
            prefix: Prefix for log messages

        Returns:
            Number of resources cleared
        """
        target = cache_dir or self.resource_dir
        removed = self._clear_strategy.clear(target, patterns)
        self.logger.info(f"{prefix}Removed {removed} resources from {target}")
        return removed

    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: Optional[List[str]] = None,
        prefix: str = "",
    ) -> None:
        """Summarize resources using configured strategy.

        Args:
            cache_dir: Directory to summarize (string for backward compatibility)
            keys: Optional keys to filter by
            prefix: Prefix for log messages
        """
        target = Path(cache_dir) if cache_dir else self.resource_dir
        self._summarize_strategy.summarize(target, keys)

    def _log_info(self, message: str, *args) -> None:
        """Log info message."""
        self.logger.info(message, *args)

    def _log_warning(self, message: str, *args) -> None:
        """Log warning message."""
        self.logger.warning(message, *args)

    def _log_error(self, message: str, *args) -> None:
        """Log error message."""
        self.logger.error(message, *args)
