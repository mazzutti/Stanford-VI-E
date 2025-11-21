"""Factory for creating pipeline stages with fluent builder pattern.

Eliminates boilerplate in stage creation and provides consistent configuration.
"""

import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from src.analysis.pipelines.orchestrator import PipelineStage

logger = logging.getLogger(__name__)

# TypeVar names include an underscore for readability across pipeline modules.
# Keep conventional TypeVar naming; disable the style warning here.
T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")

__all__ = [
    "StageFactory",
    "StageBuilder",
]

# Lightweight pipeline stage builders and factories; intentionally concise
# to keep pipeline wiring straightforward.

class StageBuilder(PipelineStage[Any, Any]):
    """Convenience base class for building stages with common patterns."""

    def __init__(self, name: str):
        self._name = name
        self._precondition: Callable[[Any], bool] | None = None

    @property
    def name(self) -> str:
        return self._name

    def can_execute(self, input_data: Any) -> bool:
        """Check precondition if set, otherwise allow execution."""
        if self._precondition is None:
            return True
        try:
            return self._precondition(input_data)
        except Exception as e:
            # Precondition predicates may raise; treat as False and log.
            logger.warning("Precondition check failed for %s: %s", self.name, e)
            return False

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute stage (implemented by subclass)."""

    def with_precondition(self, predicate: Callable[[Any], bool]) -> "StageBuilder":
        """Add a precondition for execution."""
        self._precondition = predicate
        return self

class StageFactory:
    """Factory for creating stages with fluent configuration."""

    @staticmethod
    def create_simple(
        name: str,
        execute_fn: Callable[[Any], Any],
        can_execute_fn: Callable[[Any], bool] | None = None,
    ) -> "PipelineStage[Any, Any]":
        """Create a simple stage from functions.

        Parameters
        ----------
        name : str
            Stage name
        execute_fn : Callable
            Function that executes the stage
        can_execute_fn : Callable, optional
            Function that checks if stage can execute

        Returns
        -------
        PipelineStage
            Simple stage wrapping the functions
        """

        class SimpleStage(PipelineStage[Any, Any]):
            """A simple pipeline stage wrapper around provided functions."""

            @property
            def name(self) -> str:
                return name

            def can_execute(self, input_data: Any) -> bool:
                if can_execute_fn is None:
                    return True
                try:
                    return can_execute_fn(input_data)
                except Exception as e:
                    # Predicate may raise at runtime; swallow and treat as False.
                    logger.warning("Precondition failed for %s: %s", name, e)
                    return False

            def execute(self, input_data: Any) -> Any:
                return execute_fn(input_data)

        return SimpleStage()

    @staticmethod
    def create_validator(
        name: str,
        validator_fn: Callable[[Any], bool],
        error_msg: str = "",
    ) -> "PipelineStage[Any, Any]":
        """Create a validation stage.

        Parameters
        ----------
        name : str
            Stage name
        validator_fn : Callable
            Function that validates input
        error_msg : str, optional
            Error message if validation fails

        Returns
        -------
        PipelineStage
            Validation stage
        """

        class ValidatorStage(PipelineStage[Any, Any]):
            """A pipeline stage that validates input using `validator_fn`."""

            @property
            def name(self) -> str:
                return name

            def can_execute(self, input_data: Any) -> bool:
                return True

            def execute(self, input_data: Any) -> Any:
                if not validator_fn(input_data):
                    msg = error_msg or f"Validation failed in {name}"
                    raise ValueError(msg)
                return input_data

        return ValidatorStage()

    @staticmethod
    def create_transformer(
        name: str, transform_fn: Callable[[Any], Any]
    ) -> "PipelineStage[Any, Any]":
        """Create a transformation stage.

        Parameters
        ----------
        name : str
            Stage name
        transform_fn : Callable
            Function that transforms input to output

        Returns
        -------
        PipelineStage
            Transformation stage
        """

        class TransformerStage(PipelineStage[Any, Any]):
            """A pipeline stage that applies `transform_fn` to input data."""

            @property
            def name(self) -> str:
                return name

            def can_execute(self, input_data: Any) -> bool:
                return input_data is not None

            def execute(self, input_data: Any) -> Any:
                return transform_fn(input_data)

        return TransformerStage()
