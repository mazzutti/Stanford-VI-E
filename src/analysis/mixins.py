"""Reusable mixin classes for analysis components.

This module provides composable mixins that implement common patterns used
throughout the analysis module, reducing boilerplate and improving code reuse.

Patterns:
    - SingletonMixin: Thread-safe singleton pattern with optional reconfiguration
    - ValidatableMixin: Protocol validation for dependency injection
    - ConfigurableMixin: Standard configuration management with validation
    - StateTrackingMixin: Lifecycle state management

Examples:
    Thread-safe singleton with dependency injection:

    >>> class MyService(SingletonMixin, ValidatableMixin):
    ...     _REQUIRED_METHODS = ("method_a", "method_b")
    ...
    ...     def __init__(self, dependency):
    ...         self.validate_protocol(dependency)
    ...         self.dependency = dependency

    Configurable analyzer:

    >>> class MyAnalyzer(ConfigurableMixin):
    ...     def configure(self, config: MyConfig) -> None:
    ...         self.validate_config(config, MyConfig)
    ...         self._config = config
"""

from __future__ import annotations

import logging
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "SingletonMixin",
    "ValidatableMixin",
    "ConfigurableMixin",
    "StateTrackingMixin",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")
T_Config = TypeVar("T_Config")


class SingletonMixin:
    """Mixin providing thread-safe singleton pattern with optional reconfiguration.

    Provides automatic singleton creation and management while allowing
    subclasses to define initialization logic cleanly. Supports optional
    reconfiguration for testing and runtime updates.

    Thread Safety:
        All singleton creation and modification is protected by a lock to
        ensure safe usage from multiple threads.

    Example:
        >>> class MyService(SingletonMixin):
        ...     def __init__(self, config):
        ...         self.config = config
        ...
        >>> service1 = MyService(config1)
        >>> service2 = MyService(config1)  # Same instance
        >>> assert service1 is service2
    """

    _instance: ClassVar[Optional[Any]] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __new__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Create or return the singleton instance in a thread-safe manner."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (primarily for testing).

        After reset, the next instantiation will create a new instance.
        Thread-safe.
        """
        with cls._lock:
            cls._instance = None
            logger.debug(f"Reset singleton {cls.__name__}")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if singleton has been initialized.

        Returns:
            True if singleton exists, False otherwise.
        """
        with cls._lock:
            return cls._instance is not None

    @classmethod
    def get_instance(cls: Type[T]) -> Optional[T]:
        """Get the singleton instance without creating it.

        Returns:
            The singleton instance if initialized, None otherwise.
        """
        with cls._lock:
            return cls._instance


class ValidatableMixin:
    """Mixin providing protocol validation for dependency injection.

    Enables duck-typing validation without relying on isinstance(), which
    doesn't work reliably with Protocol types. Subclasses define required
    methods via `_REQUIRED_METHODS` class variable.

    Example:
        >>> class ProcessManager(ValidatableMixin):
        ...     _REQUIRED_METHODS = ("run", "cleanup")
        ...
        >>> class Service(ValidatableMixin):
        ...     def use_manager(self, manager):
        ...         self.validate_protocol(manager, ProcessManager)
        ...         # manager is now guaranteed to have run() and cleanup()
    """

    _REQUIRED_METHODS: ClassVar[Sequence[str]] = ()

    @staticmethod
    def validate_protocol(
        obj: Any,
        required_methods: Sequence[str],
        obj_name: str = "object",
        protocol_name: str = "Protocol",
    ) -> None:
        """Validate that object implements required methods.

        Args:
            obj: Object to validate.
            required_methods: Method names that must exist on obj.
            obj_name: Name for error messages (e.g., "proc_manager").
            protocol_name: Protocol name for error messages.

        Raises:
            TypeError: If obj is missing any required methods.
        """
        missing = [m for m in required_methods if not hasattr(obj, m)]
        if missing:
            raise TypeError(
                f"{obj_name} must implement {protocol_name}; "
                f"missing methods: {missing}. "
                f"Expected: {required_methods}"
            )
        logger.debug(f"{obj_name} validated successfully: {type(obj).__name__}")

    def validate_self(self) -> None:
        """Validate that this instance implements its required methods.

        Raises:
            TypeError: If instance is missing any required methods.
        """
        if not self._REQUIRED_METHODS:
            return
        self.validate_protocol(
            self,
            self._REQUIRED_METHODS,
            obj_name=f"{self.__class__.__name__} instance",
            protocol_name=self.__class__.__name__,
        )


class ConfigurableMixin(Generic[T_Config]):
    """Mixin for standard configuration management with validation.

    Provides a consistent pattern for configuration management across
    analyzers, services, and other configurable components. Supports
    type checking and validation hooks.

    Example:
        >>> @dataclass
        ... class MyConfig:
        ...     value: int
        ...
        >>> class MyAnalyzer(ConfigurableMixin[MyConfig]):
        ...     _config: Optional[MyConfig] = None
        ...
        ...     def configure(self, config: MyConfig) -> None:
        ...         self._config = config
        ...
        ...     def get_configuration(self) -> MyConfig:
        ...         if self._config is None:
        ...             raise RuntimeError("Not configured")
        ...         return self._config
    """

    _config: Optional[T_Config] = None

    @staticmethod
    def validate_config_type(
        config: Any, expected_type: Type[T_Config], config_name: str = "config"
    ) -> None:
        """Validate that config is of expected type.

        Args:
            config: Configuration object to validate.
            expected_type: Expected type (not used for duck-typing).
            config_name: Name for error messages.

        Raises:
            TypeError: If config is not the expected type.
        """
        if not isinstance(config, expected_type):
            raise TypeError(
                f"{config_name} must be {expected_type.__name__}, "
                f"got {type(config).__name__}"
            )
        logger.debug(f"Config validated: {type(config).__name__}")

    def get_configuration(self) -> Optional[T_Config]:
        """Get the current configuration.

        Returns:
            Current configuration object or None if not configured.
        """
        return self._config


class StateTrackingMixin:
    """Mixin for tracking component lifecycle state.

    Provides standard state management with validation, useful for ensuring
    components are properly initialized before use.

    States:
        - created: Instance created but not initialized
        - initializing: Initialization in progress
        - ready: Fully initialized and ready for use
        - failed: Initialization failed
        - stopped: Component has been stopped

    Example:
        >>> class MyComponent(StateTrackingMixin):
        ...     def __init__(self):
        ...         self.mark_state("created")
        ...
        ...     def initialize(self):
        ...         self.mark_state("initializing")
        ...         try:
        ...             # Do work
        ...             self.mark_state("ready")
        ...         except Exception:
        ...             self.mark_state("failed")
    """

    _state: str = "created"
    _state_lock: ClassVar[threading.RLock] = threading.RLock()

    def mark_state(self, new_state: str) -> None:
        """Mark component as entering a new state.

        Args:
            new_state: The state to transition to.
        """
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            logger.debug(f"{self.__class__.__name__}: {old_state} -> {new_state}")

    def get_state(self) -> str:
        """Get the current state.

        Returns:
            Current state name.
        """
        with self._state_lock:
            return self._state

    def is_ready(self) -> bool:
        """Check if component is in ready state.

        Returns:
            True if state is "ready", False otherwise.
        """
        return self.get_state() == "ready"

    def assert_ready(self, operation: str = "operation") -> None:
        """Assert that component is ready or raise RuntimeError.

        Args:
            operation: Description of operation for error message.

        Raises:
            RuntimeError: If component is not in ready state.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Cannot perform {operation}: "
                f"{self.__class__.__name__} is {self.get_state()}"
            )
