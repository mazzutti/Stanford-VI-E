"""Generic resource manager with strategy pattern for operations."""

import logging
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

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
    """Strategy for clearing resources.

    Protocol used to define the interface for clear strategies. This
    intentionally contains a single method and is exempt from the
    "too-few-public-methods" check.
    """

    def clear(self, resource_dir: Path, patterns: list[str] | None = None) -> int:
        """Clear resources matching criteria.

        Args:
            resource_dir: Directory containing resources
            patterns: Optional patterns to match

        Returns:
            Number of resources cleared
        """
        raise NotImplementedError()

class SummarizeStrategy(Protocol):
    """Strategy for summarizing resources.

    Protocol used to define summarization strategies and intentionally
    contains a single method.
    """

    def summarize(self, resource_dir: Path, keys: list[str] | None = None) -> None:
        """Print summary of resources.

        Args:
            resource_dir: Directory containing resources
            keys: Optional keys to filter by
        """
        raise NotImplementedError()

class NoOpClearStrategy:
    """No-operation clear strategy.

    Lightweight implementation used as a default strategy when none is
    provided by callers; intentionally minimal.
    """

    def clear(self, resource_dir: Path, patterns: list[str] | None = None) -> int:
        """Do nothing."""
        _ = (resource_dir, patterns)
        return 0

class NoOpSummarizeStrategy:
    """No-operation summarize strategy.

    Lightweight implementation used as a default strategy when none is
    provided by callers; intentionally minimal.
    """

    def summarize(self, resource_dir: Path, keys: list[str] | None = None) -> None:
        """Do nothing."""
        _ = (resource_dir, keys)

class ResourceManager(BaseManager, Generic[T]):
    """Generic resource manager with pluggable strategies.

    Eliminates duplicate code across CacheManager, FileManager, etc.
    by using composition and strategy pattern.
    """

    def __init__(
        self,
        resource_dir: Path,
        clear_strategy: ClearStrategy | None = None,
        summarize_strategy: SummarizeStrategy | None = None,
        logger: logging.Logger | None = None,
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
        patterns: list[str] | None = None,
        cache_dir: Path | None = None,
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
        self.logger.info("%sRemoved %s resources from %s", prefix, removed, target)
        return removed

    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: list[str] | None = None,
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

    def _log_info(self, message: str, *args: Any) -> None:
        """Log info message."""
        self.logger.info(message, *args)

    def _log_warning(self, message: str, *args: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, *args)

    def _log_error(self, message: str, *args: Any) -> None:
        """Log error message."""
        self.logger.error(message, *args)
