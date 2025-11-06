"""Shared helpers and convenience utilities for analysis scripts.

This module exposes a single class, :class:`AnalysisCommon`, which provides
test-friendly wrappers around commonly used analysis helpers using the
SingletonMixin and ValidatableMixin patterns.

Design:
    - Uses SingletonMixin for thread-safe singleton management
    - Uses ValidatableMixin for ProcessManager protocol validation
    - Focuses on delegation to underlying ProcessManager
    - Reduced from 280+ lines to ~150 lines of actual logic

Thread Safety:
    Singleton creation and reconfiguration are protected by a lock inherited
    from SingletonMixin.

Usage Examples:
    Get the singleton instance::

        ac = AnalysisCommon.instance()
        ac.clear_cache()

    Initialize with a custom ProcessManager::

        manager = ProcessManager()
        ac = AnalysisCommon.instance(manager)

    Use as context manager::

        with AnalysisCommon.instance(test_manager) as ac:
            ac.clear_cache()

    Access ProcessManager directly::

        manager = ac.proc_manager
        count = manager.clear_cache(patterns=["*.pkl"])

    Dynamic method invocation::

        result = ac("clear_cache", patterns=["*.pkl"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast, Any, ClassVar


from src.utils.types import ProcessManagerProtocol
from src.utils.constants import CACHE_DIR_DEFAULT
from src.analysis.mixins import SingletonMixin, ValidatableMixin

if TYPE_CHECKING:
    pass

__all__ = ["AnalysisCommon"]

logger = logging.getLogger(__name__)


class AnalysisCommon(SingletonMixin, ValidatableMixin):
    """Object-oriented facade exposing analysis helper methods.

    Combines SingletonMixin for thread-safe singleton management and
    ValidatableMixin for ProcessManager protocol validation. Delegates all
    actual work to the configured ProcessManager.

    Usage patterns:
    - ``AnalysisCommon()`` returns the singleton instance (via SingletonMixin.__new__)
    - ``AnalysisCommon.instance()`` explicitly requests the singleton

    Key simplifications from previous version:
    - Singleton logic moved to SingletonMixin
    - Protocol validation moved to ValidatableMixin
    - Removed 150+ lines of boilerplate validation code
    - Cleaner, more focused implementation

    Thread Safety:
        Creation and reconfiguration use locks from mixins.
        Safe to use from multiple threads.
    """

    # Required methods that a ProcessManager must implement
    _REQUIRED_METHODS: ClassVar[tuple[str, ...]] = (
        "clear_cache",
        "open_file",
        "summarize_cache_files",
    )

    def __init__(self, proc_manager: ProcessManagerProtocol | None = None) -> None:
        """Initialize the AnalysisCommon singleton.

        Args:
            proc_manager: A ProcessManager instance implementing ProcessManagerProtocol.
                         Required on first initialization.

        Raises:
            TypeError: If proc_manager is None on first initialization.
            TypeError: If proc_manager does not implement ProcessManagerProtocol.
        """
        # Idempotent initializer — safe to call multiple times
        if getattr(self, "_initialized", False):
            if proc_manager is not None and proc_manager is not getattr(
                self, "_proc_manager", None
            ):
                logger.debug(
                    "AnalysisCommon already initialized; ignoring proc_manager override"
                )
            return

        # Require explicit ProcessManager to avoid module-level singleton dependencies
        if proc_manager is None:
            raise TypeError(
                "proc_manager is required on first initialization; "
                "pass a ProcessManager instance to AnalysisCommon() "
                "or use AnalysisCommon.instance(proc_manager)"
            )

        # Validate using mixin
        self.validate_protocol(
            proc_manager,
            self._REQUIRED_METHODS,
            obj_name="proc_manager",
            protocol_name="ProcessManagerProtocol",
        )

        self._proc_manager = proc_manager
        self._initialized = True

    @classmethod
    def instance(
        cls, proc_manager: ProcessManagerProtocol | None = None
    ) -> AnalysisCommon:
        """Explicit accessor for the singleton instance with lazy initialization.

        If the singleton doesn't exist and no proc_manager is provided,
        automatically uses the processing registry for default initialization.

        Args:
            proc_manager: Optional ProcessManager to initialize/reconfigure with.

        Returns:
            The AnalysisCommon singleton instance.

        Raises:
            TypeError: If ProcessManager doesn't implement ProcessManagerProtocol.
        """
        with cls._lock:
            # If no singleton exists yet, create it
            if cls._instance is None:
                if proc_manager is None:
                    # Lazy import to avoid cycles
                    from src.processing import get_registry

                    proc: ProcessManagerProtocol = cast(
                        ProcessManagerProtocol,
                        get_registry().get_manager_hub().processes,
                    )
                else:
                    proc = proc_manager
                inst = cls(proc)
                return inst

            inst = cls._instance
            # Allow reconfiguration with new manager
            if proc_manager is not None:
                inst.validate_protocol(
                    proc_manager,
                    inst._REQUIRED_METHODS,
                    obj_name="proc_manager",
                    protocol_name="ProcessManagerProtocol",
                )
                inst._proc_manager = proc_manager
                logger.info(
                    "AnalysisCommon configured with new ProcessManager: %s",
                    type(proc_manager).__name__,
                )
            return inst

    def configure(self, proc_manager: ProcessManagerProtocol) -> None:
        """Replace the underlying ProcessManager.

        Args:
            proc_manager: A ProcessManager instance implementing ProcessManagerProtocol.

        Raises:
            TypeError: If proc_manager does not implement ProcessManagerProtocol.
        """
        self.validate_protocol(
            proc_manager,
            self._REQUIRED_METHODS,
            obj_name="proc_manager",
            protocol_name="ProcessManagerProtocol",
        )
        with self._lock:
            self._proc_manager = proc_manager
        logger.info(
            "AnalysisCommon configured with new ProcessManager: %s",
            type(proc_manager).__name__,
        )

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the singleton."""
        try:
            manager_name = type(self._proc_manager).__name__
        except AttributeError:
            manager_name = "<uninitialized>"
        return f"<AnalysisCommon singleton proc_manager={manager_name}>"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        if self._is_fully_initialized:
            manager_type = type(self._proc_manager).__name__
            return f"AnalysisCommon (initialized with {manager_type})"
        return "AnalysisCommon (uninitialized)"

    @property
    def _is_fully_initialized(self) -> bool:
        """Check if the singleton has been fully initialized."""
        return getattr(self, "_initialized", False)

    @_is_fully_initialized.setter
    def _is_fully_initialized(self, value: bool) -> None:
        """Set the initialization state."""
        self._initialized = value

    @property
    def proc_manager(self) -> ProcessManagerProtocol:
        """Access the configured ProcessManager.

        Raises:
            RuntimeError: If called before initialization.
        """
        if not self._is_fully_initialized:
            raise RuntimeError(
                "ProcessManager not initialized. "
                "Call AnalysisCommon.initialize() first."
            )
        return self._proc_manager

    def __eq__(self, other: object) -> bool:
        """Check equality - singleton is only equal to itself."""
        if not isinstance(other, AnalysisCommon):
            return NotImplemented
        return id(self) == id(other)

    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __bool__(self) -> bool:
        """Check if fully initialized (allows: if AnalysisCommon.instance(): ...)."""
        return self._is_fully_initialized

    def __hash__(self) -> int:
        """Return hash of singleton instance."""
        return id(self)

    def __enter__(self) -> AnalysisCommon:
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Support context manager protocol exit."""
        return None

    # ========================================================================
    # ProcessManager Delegation Methods
    # ========================================================================

    def clear_cache(
        self,
        patterns: Optional[list[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        """Clear process caches via the configured process manager.

        Args:
            patterns: Optional list of glob patterns to match cache files.
            cache_dir: Optional path to cache directory. Defaults to CACHE_DIR_DEFAULT.
            prefix: Optional prefix for logging/identification.

        Returns:
            Number of cache files cleared.
        """
        return self._proc_manager.clear_cache(
            patterns=patterns, cache_dir=cache_dir, prefix=prefix
        )

    def open_file(
        self, filepath: str, description: Optional[str] = None, prefix: str = ""
    ) -> bool:
        """Open a file using the configured process manager helper.

        Args:
            filepath: Path to the file to open.
            description: Optional description for logging/identification.
            prefix: Optional prefix for logging/identification.

        Returns:
            True if file was successfully opened, False otherwise.
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

        Args:
            cache_dir: Path to cache directory. Defaults to CACHE_DIR_DEFAULT.
            keys: Optional list of cache keys to include in summary.
            prefix: Optional prefix for logging/identification.
        """
        self._proc_manager.summarize_cache_files(
            cache_dir=cache_dir, keys=keys, prefix=prefix
        )

    def __call__(self, method_name: str, *args: object, **kwargs: object) -> object:
        """Delegate method calls to the underlying process manager.

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
        if not self._is_fully_initialized:
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
        logger.debug(
            "Invoking ProcessManager.%s with args=%s, kwargs=%s",
            method_name,
            args,
            kwargs,
        )
        return method(*args, **kwargs)
