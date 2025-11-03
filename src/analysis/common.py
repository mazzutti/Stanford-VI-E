"""Shared helpers and convenience utilities for analysis scripts.

This module exposes a single class, :class:`AnalysisCommon`, which provides
small, test-friendly wrappers around commonly used analysis helpers.

The module intentionally avoids module-level singletons or procedural
functions — call ``AnalysisCommon()`` (it is a singleton) or use
``AnalysisCommon.instance()`` to obtain the shared instance.

Thread Safety:
    Singleton creation and reconfiguration are protected by a lock to ensure
    thread-safe initialization and runtime reconfigurations.

Usage Examples:
    Get the singleton instance with automatic lazy initialization::

        ac = AnalysisCommon.instance()
        ac.clear_cache()

    Initialize with a custom ProcessManager::

        from src.processing.process import ProcessManager
        manager = ProcessManager()
        ac = AnalysisCommon.instance(manager)

    Use context manager for temporary manager swaps::

        with AnalysisCommon.instance(test_manager) as ac:
            ac.clear_cache()

    Access the underlying ProcessManager directly::

        manager = ac.proc_manager
        count = manager.clear_cache(patterns=["*.pkl"])

    Dynamic method invocation::

        result = ac("clear_cache", patterns=["*.pkl"])
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional, cast

from src.utils.types import ProcessManagerProtocol
from src.analysis._util_imports import Path, os, sys, time, shutil
from src.analysis._process_manager_delegate import ProcessManagerDelegate

if TYPE_CHECKING:
    pass

__all__ = ["Path", "os", "sys", "time", "shutil", "AnalysisCommon"]


class AnalysisCommon(ProcessManagerDelegate):
    """Object-oriented facade exposing analysis helper methods.

    Usage patterns:
    - ``AnalysisCommon()`` always returns the same singleton instance.
    - ``AnalysisCommon.instance()`` returns the same instance (explicit).

    The class owns a small mapping of helper callables which delegates to the
    underlying ``process_manager``. Keeping these as attributes makes it
    simple to inject test doubles during unit tests.

    Thread Safety:
        Creation and reconfiguration of the singleton are protected by a lock.
        Safe to use from multiple threads.
    """

    # Singleton storage
    _instance: Optional[AnalysisCommon] = None
    # Lock used to guard singleton creation and reconfiguration
    _lock: threading.RLock = threading.RLock()
    # Logger for the class
    _logger: logging.Logger = logging.getLogger(__name__)
    # Required methods that a ProcessManager must implement
    _REQUIRED_METHODS: tuple[str, ...] = (
        "clear_cache",
        "open_file",
        "summarize_cache_files",
    )
    # Cached frozenset of required methods for faster lookups
    _REQUIRED_METHODS_SET: frozenset[str] = frozenset(_REQUIRED_METHODS)

    def __new__(
        cls: type[AnalysisCommon], *args: object, **kwargs: object
    ) -> AnalysisCommon:
        """Create or return the singleton instance in a thread-safe manner."""
        # Use the lock to ensure thread-safe singleton creation
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    @staticmethod
    def _validate_proc_manager(proc_manager: ProcessManagerProtocol) -> None:
        """Validate that proc_manager implements required methods.

        Args:
            proc_manager: The ProcessManager instance to validate.

        Raises:
            TypeError: If proc_manager is missing required methods.
        """
        missing_methods = [
            m for m in AnalysisCommon._REQUIRED_METHODS if not hasattr(proc_manager, m)
        ]
        if missing_methods:
            raise TypeError(
                f"proc_manager must implement ProcessManagerProtocol; "
                f"missing methods: {missing_methods}. "
                f"Expected methods: {AnalysisCommon._REQUIRED_METHODS}"
            )
        AnalysisCommon._logger.debug(
            "ProcessManager validated successfully: %s",
            type(proc_manager).__name__,
        )

    @staticmethod
    def _assert_initialized(instance: Optional[AnalysisCommon]) -> None:
        """Assert that the singleton has been initialized.

        Args:
            instance: The instance to check.

        Raises:
            TypeError: If instance is not initialized.
        """
        if not instance or not getattr(instance, "_initialized", False):
            raise TypeError(
                "AnalysisCommon not yet initialized. "
                "Call AnalysisCommon.instance(proc_manager) first."
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
        # Idempotent initializer — safe to call multiple times.
        if getattr(self, "_initialized", False):
            # If caller supplied a different manager after initialization,
            # ignore but log for visibility.
            if proc_manager is not None and proc_manager is not getattr(
                self, "_proc_manager", None
            ):
                AnalysisCommon._logger.debug(
                    "AnalysisCommon already initialized; ignoring proc_manager override"
                )
            return

        # Require an explicit ProcessManager to avoid import-time reliance on
        # module-level singletons. Callers must provide a manager instance.
        if proc_manager is None:
            raise TypeError(
                "proc_manager is required on first initialization; "
                "pass a ProcessManager instance to AnalysisCommon() "
                "or use AnalysisCommon.instance(proc_manager)"
            )

        # Validate proc_manager has all required methods
        self._validate_proc_manager(proc_manager)

        self._proc_manager = proc_manager
        self._initialized = True

    @classmethod
    def instance(
        cls, proc_manager: ProcessManagerProtocol | None = None
    ) -> AnalysisCommon:
        """Explicit accessor for the singleton instance.

        Equivalent to calling ``AnalysisCommon()`` but reads clearer at call
        sites that explicitly request the shared instance.

        This method implements lazy initialization: if the singleton doesn't
        exist and no proc_manager is provided, it will automatically call
        ``get_process_manager()`` from the processing module to initialize with
        a default ProcessManager.

        Args:
            proc_manager: Optional ProcessManager to initialize or reconfigure with.
                         If None and singleton doesn't exist, uses get_process_manager().

        Returns:
            The AnalysisCommon singleton instance.

        Raises:
            TypeError: If the ProcessManager does not implement ProcessManagerProtocol.
        """
        # Use lock to ensure thread-safe singleton creation and configuration
        with cls._lock:
            # If no singleton exists yet, construct it with a provided
            # proc_manager or fall back to the module-level default
            if cls._instance is None:
                if proc_manager is None:
                    # Import lazily to avoid import-time cycles
                    from src.processing.process import get_process_manager

                    proc: ProcessManagerProtocol = cast(
                        ProcessManagerProtocol, get_process_manager()
                    )
                else:
                    proc = proc_manager
                inst = cls(proc)
                return inst

            inst = cls._instance
            if proc_manager is not None:
                # Allow callers to replace the manager even if the singleton
                # was already constructed previously.
                # Validate the protocol without relying on isinstance
                cls._validate_proc_manager(proc_manager)
                inst._proc_manager = proc_manager
                AnalysisCommon._logger.info(
                    "AnalysisCommon configured with new ProcessManager: %s",
                    type(proc_manager).__name__,
                )
            return inst

    @classmethod
    def create_with_manager(
        cls, proc_manager: ProcessManagerProtocol
    ) -> AnalysisCommon:
        """Factory method to create/configure the singleton with a specific manager.

        This is an alternative to ``instance(proc_manager)`` that explicitly
        shows intent to initialize with a particular manager.

        Args:
            proc_manager: ProcessManager instance implementing ProcessManagerProtocol.

        Returns:
            The AnalysisCommon singleton instance configured with the given manager.

        Raises:
            TypeError: If proc_manager does not implement ProcessManagerProtocol.

        Example::

            from src.processing.process import ProcessManager
            manager = ProcessManager()
            ac = AnalysisCommon.create_with_manager(manager)
        """
        return cls.instance(proc_manager)

    def configure(self, proc_manager: ProcessManagerProtocol) -> None:
        """Replace the underlying ProcessManager and rebuild helpers.

        This allows tests or runtime code to swap in a different manager
        instance after the singleton has been created.

        Args:
            proc_manager: A ProcessManager instance implementing ProcessManagerProtocol.

        Raises:
            TypeError: If proc_manager does not implement ProcessManagerProtocol.

        Thread Safety:
            This method is protected by a lock to ensure safe reconfiguration
            from multiple threads.
        """
        # Validate the protocol without relying on isinstance (which may not work
        # correctly with Protocol types). Check for required methods instead.
        self._validate_proc_manager(proc_manager)

        # Guard reconfiguration with a lock to ensure thread-safety
        with type(self)._lock:
            self._proc_manager = proc_manager

        AnalysisCommon._logger.info(
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
        """Return a human-readable string representation of the singleton.

        Returns:
            A formatted string with initialization status and manager type.
        """
        if self.is_initialized:
            manager_type = type(self._proc_manager).__name__
            return f"AnalysisCommon (initialized with {manager_type})"
        return "AnalysisCommon (uninitialized)"

    @property
    def is_initialized(self) -> bool:
        """Check if the singleton has been fully initialized.

        Returns:
            True if proc_manager has been set, False otherwise.
        """
        return getattr(self, "_initialized", False)

    @is_initialized.setter
    def is_initialized(self, value: bool) -> None:
        """Set the initialization state.

        Parameters:
            value: The initialization state to set.
        """
        self._initialized = value

    @property
    def proc_manager(self) -> ProcessManagerProtocol:
        """Access the configured ProcessManager.

        Returns:
            The underlying ProcessManager instance.

        Raises:
            AttributeError: If accessed before initialization.
        """
        if not self.is_initialized:
            raise AttributeError(
                "ProcessManager not yet initialized. "
                "Call AnalysisCommon.instance(proc_manager) first."
            )
        return self._proc_manager

    def __eq__(self, other: object) -> bool:
        """Check equality with another object.

        Since AnalysisCommon is a singleton, it is only equal to itself.

        Args:
            other: The object to compare with.

        Returns:
            True if other is the same singleton instance, False otherwise.
        """
        if not isinstance(other, AnalysisCommon):
            return NotImplemented
        # All instances are the same singleton
        return id(self) == id(other)

    def __ne__(self, other: object) -> bool:
        """Check inequality with another object.

        Args:
            other: The object to compare with.

        Returns:
            True if other is not the same singleton instance, False otherwise.
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __bool__(self) -> bool:
        """Check if the singleton is fully initialized.

        Returns:
            True if initialized, False otherwise.

        Note:
            This allows Pythonic checks like: if AnalysisCommon.instance(): ...
        """
        return self.is_initialized

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing purposes.

        This forcibly clears the singleton and reinitializes it.
        Use with caution - only intended for test cleanup.

        Thread Safety:
            This method is protected by a lock to ensure thread-safe reset.

        Warning:
            This is primarily for testing. Production code should not call this method.
        """
        with cls._lock:
            cls._instance = None
            AnalysisCommon._logger.debug("AnalysisCommon singleton reset")

    @classmethod
    def temporary_manager(cls, temp_manager: ProcessManagerProtocol) -> AnalysisCommon:
        """Context manager for temporarily swapping the ProcessManager.

        This is a convenience factory for creating a context manager that
        temporarily replaces the ProcessManager and restores it on exit.

        Args:
            temp_manager: Temporary ProcessManager to use within the context.

        Returns:
            The AnalysisCommon singleton configured with the temporary manager.

        Example::

            original_manager = AnalysisCommon.instance().proc_manager
            with AnalysisCommon.temporary_manager(test_manager) as ac:
                ac.clear_cache()  # Uses test_manager
            # Restored to original_manager here
        """
        original = cls.instance()
        cls.instance(temp_manager)
        return original

    def __hash__(self) -> int:
        """Return hash of the singleton instance.

        Since this is a singleton, all instances are identical and hash to
        the same value, allowing use in sets and as dict keys.

        Returns:
            Hash value based on object identity.
        """
        return id(self)

    def __enter__(self) -> AnalysisCommon:
        """Support context manager protocol for temporary configuration swaps.

        Example:
            with AnalysisCommon.instance(test_manager) as ac:
                # Use temporary manager
                ac.clear_cache()
        """
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Support context manager protocol exit.

        Returns:
            None to propagate any exceptions that occurred in the with block.
        """
        # No cleanup needed, but implementing for completeness
        return None
