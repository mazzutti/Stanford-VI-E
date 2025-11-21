"""Process management utilities with simplified API."""

import logging
from pathlib import Path

from src.processing.managers.base import BaseManager
from src.processing.managers.cache import CacheManager
from src.processing.managers.file import FileManager

__all__ = ["ProcessManager", "ManagerHub"]


class ProcessManager(BaseManager):
    """Facade for process-related utilities with simpler API.

    Composes CacheManager and FileManager to provide unified process management,
    without duplicating method names.

    Attributes:
        cache: CacheManager instance
        files: FileManager instance
    """

    def __init__(
        self,
        cache_manager: CacheManager | None = None,
        file_manager: FileManager | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize process manager with dependencies.

        Args:
            cache_manager: Cache manager instance (auto-created if None)
            file_manager: File manager instance (auto-created if None)
            logger: Logger instance
        """
        super().__init__(logger=logger)
        self.cache = cache_manager or CacheManager(logger=self.logger)
        self.files = file_manager or FileManager(logger=self.logger)

    def clear(
        self,
        patterns: list[str] | None = None,
        cache_dir: Path | None = None,
        prefix: str = "",
    ) -> int:
        """Clear cache files matching patterns or using size-based pruning.

        Args:
            patterns: Optional glob patterns to match
            cache_dir: Cache directory (defaults to ".cache")
            prefix: Prefix for log messages

        Returns:
            Number of files removed
        """
        return self.cache.clear(patterns=patterns, cache_dir=cache_dir, prefix=prefix)

    def open_file(self, filepath: str, prefix: str = "") -> bool:
        """Open a file using the FileManager.

        Args:
            filepath: Path to file to open
            description: Optional description of file
            prefix: Prefix for log messages

        Returns:
            True if file opened successfully
        """
        return self.files.open(filepath=filepath, prefix=prefix)

    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: list[str] | None = None,
        prefix: str = "",
    ) -> None:
        """Print summary of cache files.

        Args:
            cache_dir: Cache directory to summarize
            keys: Optional keys to filter cache entries
            prefix: Prefix for log messages
        """
        return self.cache.summarize(cache_dir=cache_dir, keys=keys, prefix=prefix)

    # Protocol-compatible aliases for AnalysisCommon
    def clear_cache(
        self,
        patterns: list[str] | None = None,
        cache_dir: Path | None = None,
        prefix: str = "",
    ) -> int:
        """Alias for clear() to match ProcessManagerProtocol."""
        return self.clear(patterns=patterns, cache_dir=cache_dir, prefix=prefix)

    def summarize_cache_files(
        self,
        cache_dir: str = ".cache",
        keys: list[str] | None = None,
        prefix: str = "",
    ) -> None:
        """Alias for summarize() to match ProcessManagerProtocol."""
        return self.summarize(cache_dir=cache_dir, keys=keys, prefix=prefix)


class ManagerHub(BaseManager):
    """Unified facade for all resource managers.

    Provides single entry point for managing cache, files, and processes.

    Attributes:
        cache: CacheManager instance
        files: FileManager instance
        processes: ProcessManager instance
    """

    def __init__(
        self,
        cache_manager: CacheManager | None = None,
        file_manager: FileManager | None = None,
        process_manager: ProcessManager | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize manager hub with dependencies.

        Args:
            cache_manager: Cache manager (auto-created if None)
            file_manager: File manager (auto-created if None)
            process_manager: Process manager (auto-created if None)
            logger: Logger instance
        """
        super().__init__(logger=logger)
        self.cache = cache_manager or CacheManager(logger=self.logger)
        self.files = file_manager or FileManager(logger=self.logger)
        self.processes = process_manager or ProcessManager(
            cache_manager=self.cache, file_manager=self.files, logger=self.logger
        )

    def clear(
        self,
        patterns: list[str] | None = None,
        cache_dir: Path | None = None,
        prefix: str = "",
    ) -> int:
        """Clear all managed resources.

        Args:
            patterns: Glob patterns to match
            cache_dir: Cache directory
            prefix: Log message prefix

        Returns:
            Total number of resources cleared
        """
        return self.cache.clear(patterns=patterns, cache_dir=cache_dir, prefix=prefix)

    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: list[str] | None = None,
        prefix: str = "",
    ) -> None:
        """Print summary of all managed resources.

        Args:
            cache_dir: Cache directory to summarize
            keys: Optional cache entry keys to filter
            prefix: Log message prefix
        """
        self._log_info("Manager Hub Summary")
        self._log_info("=" * 50)
        self.cache.summarize(cache_dir=cache_dir, keys=keys, prefix=prefix)
