"""Base interface for all domain-specific analyzers.

This module defines the AnalyzerInterface that all domain analyzers must implement,
enabling polymorphic usage and consistent orchestration of different analysis types.

Design Patterns:
    - Interface/Protocol: Clear contract for all analyzers
    - Template Method: Standardized lifecycle (validate → analyze)
    - Dependency Injection: Configurations injected, not created
    - Generic Types: Type-safe configuration and result handling
    - Composition: Use mixins for cross-cutting concerns (see src.analysis.mixins)

Key Simplifications:
    - Removed duplicate name property/method (use 'name' property only)
    - Removed separate get_name() method
    - Merged lifecycle checking with is_ready()
    - Clear separation: validate_inputs() checks data, is_ready() checks component state

Available Mixins (for optional use):
    - SingletonMixin: Thread-safe singleton pattern
    - ValidatableMixin: Protocol validation for dependency injection
    - ConfigurableMixin[T]: Generic configuration management
    - StateTrackingMixin: Component lifecycle state tracking

Example Implementation (basic analyzer):
    >>> from dataclasses import dataclass
    >>> from src.analysis.base import AnalyzerInterface, AnalysisConfig
    >>>
    >>> @dataclass
    ... class MyConfig(AnalysisConfig):
    ...     timeout: float = 60.0
    >>>
    >>> class MyAnalyzer(AnalyzerInterface[MyConfig, dict]):
    ...     def __init__(self):
    ...         self._config: Optional[MyConfig] = None
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_domain"
    ...
    ...     def validate_inputs(self, **kwargs) -> bool:
    ...         return "data" in kwargs
    ...
    ...     def analyze(self, **kwargs) -> dict:
    ...         return {"result": "success"}
    ...
    ...     def configure(self, config: MyConfig) -> None:
    ...         self._config = config
    ...
    ...     def get_configuration(self) -> MyConfig:
    ...         if self._config is None:
    ...             raise RuntimeError("Not configured")
    ...         return self._config
    ...
    ...     def is_ready(self) -> bool:
    ...         return self._config is not None

Example Usage (polymorphic):
    >>> analyzers: list[AnalyzerInterface] = [MyAnalyzer(), ...]
    >>> for analyzer in analyzers:
    ...     if analyzer.is_ready() and analyzer.validate_inputs(data=None):
    ...         result = analyzer.analyze(data=None)

Example with Mixins (advanced singleton):
    >>> from src.analysis.mixins import SingletonMixin
    >>> class MyService(AnalyzerInterface, SingletonMixin):
    ...     pass
    >>>
    >>> s1 = MyService()
    >>> s2 = MyService()
    >>> assert s1 is s2  # Same instance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

__all__ = [
    "AnalyzerInterface",
    "AnalysisConfig",
]

T_Config = TypeVar("T_Config")  # Analyzer-specific configuration type
T_Result = TypeVar("T_Result")  # Analyzer-specific result type


class AnalysisConfig(ABC):
    """Base class for analyzer-specific configuration objects.

    Provides a common interface for accessing configuration properties
    across different analyzer types. Subclasses should use @dataclass
    decorator for automatic to_dict() behavior.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class MyConfig(AnalysisConfig):
        ...     cache_dir: str = ".cache"
        ...     timeout: float = 60.0
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation.

        For dataclass configs, returns asdict(self).
        Other implementations can override for custom serialization.

        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary for serialization.
        """
        if is_dataclass(self):
            return asdict(self)  # type: ignore
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement to_dict() "
            "or be decorated with @dataclass"
        )


class AnalyzerInterface(ABC, Generic[T_Config, T_Result]):
    """Unified interface for all domain-specific analyzers.

    This abstract base class defines the minimal contract that all analyzers
    must implement. Subclasses focus on domain-specific logic while the
    interface ensures consistent lifecycle management.

    Key Methods:
        - name: Identify the analyzer domain
        - validate_inputs(): Check data suitability (returns bool, doesn't throw)
        - analyze(): Execute analysis (assumes validate_inputs() already called)
        - configure()/get_configuration(): Configuration management
        - is_ready(): Check component state (dependencies initialized, configured, etc.)

    Type Parameters
    ---------------
    T_Config
        Type of configuration object specific to this analyzer.
    T_Result
        Type of result object returned by analyze().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifying this analyzer's domain.

        Examples: 'facies', 'rock_physics', 'seismic_attributes'

        Returns
        -------
        str
            Unique, lowercase identifier for this analyzer domain.
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs: Any) -> bool:
        """Validate that input data is suitable for analysis.

        Checks for required parameters, data shapes, value ranges, and
        domain-specific constraints. Should return bool without raising
        exceptions for invalid inputs.

        Parameters
        ----------
        **kwargs : Any
            Input parameters to validate.

        Returns
        -------
        bool
            True if inputs are valid and analysis can proceed, False otherwise.

        Notes
        -----
        - Must not modify input data
        - Return False for invalid inputs, don't raise
        - Only raise exceptions for unexpected/fatal errors
        """
        pass

    @abstractmethod
    def analyze(self, **kwargs: Any) -> T_Result:
        """Execute the analysis pipeline.

        Performs domain-specific analysis. Assumes validate_inputs() has
        already been called and returned True. Results should include
        metadata about execution (time, success, etc.).

        Parameters
        ----------
        **kwargs : Any
            Domain-specific input parameters and data.

        Returns
        -------
        T_Result
            Analysis results specific to this analyzer's domain.

        Raises
        ------
        ValueError
            If inputs are invalid (caller should check validate_inputs first).
        RuntimeError
            If analysis fails during execution.
        """
        pass

    @abstractmethod
    def configure(self, config: T_Config) -> None:
        """Update the analyzer configuration.

        Parameters
        ----------
        config : T_Config
            New configuration object to use.

        Raises
        ------
        ValueError
            If configuration is invalid.
        TypeError
            If config is not the expected type.
        """
        pass

    @abstractmethod
    def get_configuration(self) -> T_Config:
        """Get the current analyzer configuration.

        Returns
        -------
        T_Config
            Current configuration object for this analyzer.
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if analyzer is ready to execute analysis.

        Verifies that dependencies, configuration, and resources are
        properly initialized. Call before analyze() to ensure safety.

        Returns
        -------
        bool
            True if analyzer has all required state, False otherwise.
        """
        pass
