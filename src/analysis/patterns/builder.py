"""Builder Pattern Implementation for Analyzer Configuration

This module provides a fluent builder interface for creating and configuring
complex analysis components, making configuration intuitive and chainable.

Patterns Used:
  - Builder: Fluent API for complex object construction
  - Fluent Interface: Method chaining for readability

Example:
    >>> from src.analysis.patterns.builder import FaciesAnalyzerBuilder
    >>>
    >>> analyzer = (FaciesAnalyzerBuilder()
    ...     .with_transitions([Transition(1, 2), Transition(2, 3)])
    ...     .with_boundary_detection(min_change=0.05)
    ...     .with_logger()
    ...     .build())
    >>>
    >>> result = analyzer.run(data)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod
import logging

from src.analysis.models.config import (
    FaciesCorrelationConfig,
    Transition,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisBuilderBase",
    "FaciesAnalyzerBuilder",
    "ProcessorChainBuilder",
]


class AnalysisBuilderBase(ABC):
    """Abstract base class for analysis builders.

    Provides common builder functionality including validation and
    component registration.
    """

    def __init__(self):
        """Initialize the builder"""
        self._components: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self._validators: List[callable] = []

    @abstractmethod
    def build(self) -> Any:
        """Build and return the configured component.

        Returns:
            The configured component instance
        """
        pass

    def validate(self) -> bool:
        """Run all validators on current configuration.

        Returns:
            True if all validators pass, False otherwise

        Raises:
            ValueError: If validation fails
        """
        for validator in self._validators:
            if not validator(self._config):
                raise ValueError(f"Validation failed: {validator.__name__}")
        return True

    def reset(self) -> AnalysisBuilderBase:
        """Reset builder to initial state.

        Returns:
            Self for chaining
        """
        self._components.clear()
        self._config.clear()
        logger.debug(f"Reset {self.__class__.__name__}")
        return self


class FaciesAnalyzerBuilder(AnalysisBuilderBase):
    """Fluent builder for FaciesCorrelationAnalyzer configuration.

    Simplifies creation of complex analyzer configurations through
    method chaining.
    """

    def __init__(self):
        """Initialize facies analyzer builder"""
        super().__init__()
        self._config["transitions"] = []
        self._config["boundary_config"] = {}
        self._config["processors"] = {}
        self._config["validators"] = {}
        self._config["use_logger"] = False

    def with_transitions(self, transitions: List[Transition]) -> FaciesAnalyzerBuilder:
        """Set transitions for correlation analysis.

        Args:
            transitions: List of Transition objects

        Returns:
            Self for chaining
        """
        if not transitions:
            raise ValueError("At least one transition required")

        self._config["transitions"] = transitions
        logger.debug(f"Added {len(transitions)} transitions")
        return self

    def with_boundary_detection(
        self,
        min_change: float = 0.05,
        window_size: int = 3,
        use_gradient: bool = True,
    ) -> FaciesAnalyzerBuilder:
        """Configure boundary detection.

        Args:
            min_change: Minimum change threshold
            window_size: Window size for change detection
            use_gradient: Whether to use gradient-based detection

        Returns:
            Self for chaining
        """
        if min_change <= 0 or min_change >= 1:
            raise ValueError("min_change must be between 0 and 1")

        self._config["boundary_config"] = {
            "min_change": min_change,
            "window_size": window_size,
            "use_gradient": use_gradient,
        }
        logger.debug(
            f"Configured boundary detection: "
            f"min_change={min_change}, window_size={window_size}"
        )
        return self

    def with_processor(
        self,
        name: str,
        processor: Any,
    ) -> FaciesAnalyzerBuilder:
        """Add a data processor to the chain.

        Args:
            name: Processor identifier
            processor: Processor instance

        Returns:
            Self for chaining
        """
        if not processor:
            raise ValueError(f"Processor '{name}' cannot be None")

        self._config["processors"][name] = processor
        logger.debug(f"Added processor: {name}")
        return self

    def with_validator(
        self,
        name: str,
        validator: Any,
    ) -> FaciesAnalyzerBuilder:
        """Add a data validator to the chain.

        Args:
            name: Validator identifier
            validator: Validator instance

        Returns:
            Self for chaining
        """
        if not validator:
            raise ValueError(f"Validator '{name}' cannot be None")

        self._config["validators"][name] = validator
        logger.debug(f"Added validator: {name}")
        return self

    def with_logger(self, enabled: bool = True) -> FaciesAnalyzerBuilder:
        """Enable/disable execution logging.

        Args:
            enabled: Whether logging is enabled

        Returns:
            Self for chaining
        """
        self._config["use_logger"] = enabled
        logger.debug(f"Logging enabled: {enabled}")
        return self

    def with_cache(self, enabled: bool = True) -> FaciesAnalyzerBuilder:
        """Enable/disable result caching.

        Args:
            enabled: Whether caching is enabled

        Returns:
            Self for chaining
        """
        self._config["use_cache"] = enabled
        logger.debug(f"Caching enabled: {enabled}")
        return self

    def with_config_object(
        self,
        config: FaciesCorrelationConfig,
    ) -> FaciesAnalyzerBuilder:
        """Set the entire configuration object.

        Args:
            config: FaciesCorrelationConfig instance

        Returns:
            Self for chaining
        """
        if not config:
            raise ValueError("Config cannot be None")

        self._components["config"] = config
        logger.debug("Set custom configuration object")
        return self

    def build(self) -> Any:
        """Build and return configured FaciesCorrelationAnalyzer.

        Returns:
            FaciesCorrelationAnalyzer instance

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        self.validate()

        # Import here to avoid circular imports
        from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer
        from src.analysis.facies.config import FaciesAnalysisConfig

        # Use provided config or create new one
        if "config" not in self._components:
            config = FaciesAnalysisConfig()
        else:
            config = self._components["config"]

        # Create analyzer
        analyzer = FaciesCorrelationAnalyzer(config=config)

        # Add processors if analyzer has this capability
        if hasattr(analyzer, "processors"):
            for name, processor in self._config["processors"].items():
                analyzer.processors[name] = processor

        # Add validators if analyzer has this capability
        if hasattr(analyzer, "validators"):
            for name, validator in self._config["validators"].items():
                analyzer.validators[name] = validator

        logger.info(
            f"Built FaciesCorrelationAnalyzer with "
            f"{len(self._config['transitions'])} transitions, "
            f"{len(self._config['processors'])} processors, "
            f"{len(self._config['validators'])} validators"
        )

        return analyzer


class ProcessorChainBuilder(AnalysisBuilderBase):
    """Fluent builder for processor chain configuration.

    Allows building chains of data processors with flexible ordering
    and configuration.
    """

    def __init__(self):
        """Initialize processor chain builder"""
        super().__init__()
        self._config["processors"] = []

    def add_processor(
        self,
        processor: Any,
        name: Optional[str] = None,
    ) -> ProcessorChainBuilder:
        """Add a processor to the chain.

        Args:
            processor: Processor instance
            name: Optional name for the processor

        Returns:
            Self for chaining
        """
        if not processor:
            raise ValueError("Processor cannot be None")

        proc_name = name or processor.__class__.__name__
        self._config["processors"].append((proc_name, processor))
        logger.debug(f"Added processor to chain: {proc_name}")
        return self

    def add_validator(
        self,
        validator: Any,
        name: Optional[str] = None,
    ) -> ProcessorChainBuilder:
        """Add a validator to the chain.

        Args:
            validator: Validator instance
            name: Optional name for the validator

        Returns:
            Self for chaining
        """
        if not validator:
            raise ValueError("Validator cannot be None")

        val_name = name or validator.__class__.__name__
        if "validators" not in self._config:
            self._config["validators"] = []

        self._config["validators"].append((val_name, validator))
        logger.debug(f"Added validator to chain: {val_name}")
        return self

    def with_error_handling(self, enabled: bool = True) -> ProcessorChainBuilder:
        """Enable/disable error handling in chain.

        Args:
            enabled: Whether error handling is enabled

        Returns:
            Self for chaining
        """
        self._config["error_handling"] = enabled
        logger.debug(f"Error handling enabled: {enabled}")
        return self

    def build(self) -> List[tuple]:
        """Build and return processor chain.

        Returns:
            List of (name, processor) tuples in execution order
        """
        if not self._config["processors"]:
            raise ValueError("Chain must have at least one processor")

        logger.info(
            f"Built processor chain with "
            f"{len(self._config['processors'])} processors, "
            f"error_handling={self._config.get('error_handling', False)}"
        )

        return self._config["processors"]
