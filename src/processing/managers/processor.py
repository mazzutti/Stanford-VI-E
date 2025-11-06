"""Process management utilities."""

from pathlib import Path
import logging
from typing import List, Optional

from src.processing.managers.base import BaseManager
from src.processing.managers.cache import CacheManager
from src.processing.managers.file import FileManager

__all__ = ["ProcessManager", "ManagerHub"]


class ProcessManager(BaseManager):
    """High-level facade for process-related utilities.

    Provides a convenient API surface that composes CacheManager and FileManager
    for easier testing and dependency injection.
    """

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        file_manager: Optional[FileManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger=logger)
        self.cache_manager = cache_manager or CacheManager(logger=self.logger)
        self.file_manager = file_manager or FileManager(logger=self.logger)

    def clear(
        self,
        patterns: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        """Clear cache files. Delegates to CacheManager."""
        return self.cache_manager.clear(
            patterns=patterns, cache_dir=cache_dir, prefix=prefix
        )

    def clear_cache(
        self,
        patterns: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        """Clear cache files. Delegates to CacheManager."""
        return self.cache_manager.clear(
            patterns=patterns, cache_dir=cache_dir, prefix=prefix
        )

    def open_file(
        self, filepath: str, description: Optional[str] = None, prefix: str = ""
    ) -> bool:
        """Open a file. Delegates to FileManager."""
        return self.file_manager.open(
            filepath=filepath, description=description, prefix=prefix
        )

    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: Optional[List[str]] = None,
        prefix: str = "",
    ) -> None:
        """Summarize cache files. Delegates to CacheManager."""
        return self.cache_manager.summarize(
            cache_dir=cache_dir, keys=keys, prefix=prefix
        )

    def summarize_cache_files(
        self,
        cache_dir: str = ".cache",
        keys: Optional[List[str]] = None,
        prefix: str = "",
    ) -> None:
        """Summarize cache files. Delegates to CacheManager."""
        return self.cache_manager.summarize(
            cache_dir=cache_dir, keys=keys, prefix=prefix
        )


class ManagerHub(BaseManager):
    """Unified interface for all managers.

    Provides single entry point for managing all resources (cache, files, processes).
    """

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        file_manager: Optional[FileManager] = None,
        process_manager: Optional[ProcessManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger=logger)
        self.cache = cache_manager or CacheManager(logger=self.logger)
        self.files = file_manager or FileManager(logger=self.logger)
        self.processes = process_manager or ProcessManager(logger=self.logger)

    def clear(self, *args, **kwargs) -> int:
        """Clear all managed resources.

        Returns:
            Total number of resources cleared
        """
        total = 0
        total += self.cache.clear(*args, **kwargs)
        total += self.processes.cache_manager.clear(*args, **kwargs)
        return total

    def summarize(self, *args, **kwargs) -> None:
        """Print summary for all managers."""
        self._log_info("Manager Hub Summary")
        self._log_info("-" * 40)
        self.cache.summarize(*args, **kwargs)
        # Files don't have meaningful summary
        # Processes can add summary as needed
