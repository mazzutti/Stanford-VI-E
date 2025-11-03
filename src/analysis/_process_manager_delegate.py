"""ProcessManager delegation methods for AnalysisCommon.

This module provides the delegation interface between AnalysisCommon singleton
and the underlying ProcessManager, including convenience methods for common
operations like cache management and file handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.utils.constants import CACHE_DIR_DEFAULT
from src.utils.types import ProcessManagerProtocol


__all__ = ["ProcessManagerDelegate"]


class ProcessManagerDelegate:
    """Mixin providing ProcessManager delegation methods.

    This class encapsulates all delegation methods that forward calls
    to the underlying ProcessManager instance. It's designed to be
    mixed into AnalysisCommon to keep the core singleton class focused
    on lifecycle management while these methods handle ProcessManager
    interaction.

    Usage:
        This mixin is inherited by AnalysisCommon. Callers access
        delegation methods directly on AnalysisCommon instances.

    Example:
        ac = AnalysisCommon.instance(manager)
        ac.clear_cache(patterns=["*.pkl"])
        ac.open_file("output.txt")
    """

    _proc_manager: ProcessManagerProtocol
    is_initialized: bool
    _REQUIRED_METHODS: tuple[str, ...]

    def clear_cache(
        self,
        patterns: Optional[list[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        """Clear process caches via the configured process manager.

        Delegates to :meth:`src.processing.process.ProcessManager.clear_cache`.

        Args:
            patterns: Optional list of glob patterns to match cache files.
            cache_dir: Optional path to cache directory. Defaults to CACHE_DIR_DEFAULT.
            prefix: Optional prefix for logging/identification.

        Returns:
            Number of cache files cleared.

        Raises:
            Any exceptions raised by the underlying process manager.
        """
        return self._proc_manager.clear_cache(
            patterns=patterns, cache_dir=cache_dir, prefix=prefix
        )

    def open_file(
        self, filepath: str, description: Optional[str] = None, prefix: str = ""
    ) -> bool:
        """Open a file using the configured process manager helper.

        Delegates to :meth:`src.processing.process.ProcessManager.open_file`.

        Args:
            filepath: Path to the file to open.
            description: Optional description for logging/identification.
            prefix: Optional prefix for logging/identification.

        Returns:
            True if file was successfully opened, False otherwise.

        Raises:
            Any exceptions raised by the underlying process manager or OS.
        """
        return self._proc_manager.open_file(
            filepath=filepath, description=description, prefix=prefix
        )

    def summarize_cache_files(
        self,
        cache_dir: str = CACHE_DIR_DEFAULT,
        keys: Optional[list[str]] = None,
        prefix: str = "",
    ) -> None:
        """Return a summary of cache files using the process manager.

        Delegates to :meth:`src.processing.process.ProcessManager.summarize_cache_files`.

        Args:
            cache_dir: Path to cache directory. Defaults to CACHE_DIR_DEFAULT.
            keys: Optional list of cache keys to include in summary.
            prefix: Optional prefix for logging/identification.

        Raises:
            Any exceptions raised by the underlying process manager.
        """
        self._proc_manager.summarize_cache_files(
            cache_dir=cache_dir, keys=keys, prefix=prefix
        )

    def __call__(self, method_name: str, *args: object, **kwargs: object) -> object:
        """Delegate method calls to the underlying process manager.

        This allows direct invocation of ProcessManager methods via the singleton.

        Args:
            method_name: Name of the method to call on the ProcessManager.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            Result from the delegated ProcessManager method.

        Raises:
            AttributeError: If the method doesn't exist on the ProcessManager.
            TypeError: If singleton is not initialized.
        """
        if not self.is_initialized:
            raise TypeError(
                "Cannot invoke methods before initialization. "
                "Call AnalysisCommon.instance(proc_manager) first."
            )
        if not hasattr(self._proc_manager, method_name):
            raise AttributeError(
                f"ProcessManager has no method '{method_name}'. "
                f"Available methods: {self._REQUIRED_METHODS}"
            )
        method = getattr(self._proc_manager, method_name)
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            "Invoking ProcessManager.%s with args=%s, kwargs=%s",
            method_name,
            args,
            kwargs,
        )
        return method(*args, **kwargs)
