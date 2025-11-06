"""Factory for creating pipeline stages with fluent builder pattern.

Eliminates boilerplate in stage creation and provides consistent configuration.
"""

from typing import Callable, Any, Optional, TypeVar
from abc import abstractmethod
import logging

from src.analysis.pipelines.orchestrator import PipelineStage

logger = logging.getLogger(__name__)

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")

__all__ = [
    "StageFactory",
    "StageBuilder",
]


class StageBuilder(PipelineStage):
    """Convenience base class for building stages with common patterns."""

    def __init__(self, name: str):
        self._name = name
        self._precondition: Optional[Callable[[Any], bool]] = None

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
            logger.warning(f"Precondition check failed for {self.name}: {e}")
            return False

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute stage (implemented by subclass)."""
        pass

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
        can_execute_fn: Optional[Callable[[Any], bool]] = None,
    ) -> PipelineStage:
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

        class SimpleStage(PipelineStage):
            @property
            def stage_name(self) -> str:
                return name

            def can_execute(self, input_data: Any) -> bool:
                if can_execute_fn is None:
                    return True
                try:
                    return can_execute_fn(input_data)
                except Exception as e:
                    logger.warning(f"Precondition failed for {name}: {e}")
                    return False

            def execute(self, input_data: Any) -> Any:
                return execute_fn(input_data)

        return SimpleStage()

    @staticmethod
    def create_validator(
        name: str,
        validator_fn: Callable[[Any], bool],
        error_msg: str = "",
    ) -> PipelineStage:
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

        class ValidatorStage(PipelineStage):
            @property
            def stage_name(self) -> str:
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
    ) -> PipelineStage:
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

        class TransformerStage(PipelineStage):
            @property
            def stage_name(self) -> str:
                return name

            def can_execute(self, input_data: Any) -> bool:
                return input_data is not None

            def execute(self, input_data: Any) -> Any:
                return transform_fn(input_data)

        return TransformerStage()
