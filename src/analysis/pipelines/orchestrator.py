"""Pipeline and workflow orchestration for composable analysis stages.

This module provides:
- PipelineStage: Abstract interface for analysis pipeline stages
- Pipeline: Composable orchestrator for executing stages in sequence
- ConditionalStage: Stage that executes based on predicates
- StageResult: Structured result from stage execution

Pattern: Pipeline/Chain of Responsibility
- Composable stages that can be reused across analyzers
- Support for conditional execution paths
- Built-in support for stage composition and transformation
- Integration with EventEmitter for observability
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    cast,
)
import logging
from datetime import datetime

__all__ = [
    "StageResult",
    "PipelineStage",
    "Pipeline",
    "ConditionalStage",
    "ParallelPipeline",
]

logger = logging.getLogger(__name__)

T_In = TypeVar("T_In")  # Input type for stage
T_Out = TypeVar("T_Out")  # Output type for stage


@dataclass
class StageResult(Generic[T_Out]):
    """Result from executing a pipeline stage.

    Attributes
    ----------
    stage_name : str
        Name of the stage that produced this result.
    success : bool
        Whether stage execution succeeded.
    output : Optional[T_Out]
        Stage output (None if failed).
    error : Optional[Exception]
        Exception if execution failed.
    duration_ms : float
        How long stage took to execute (milliseconds).
    metadata : Dict[str, Any]
        Additional stage metadata (items processed, memory used, etc).
    """

    stage_name: str
    success: bool
    output: Optional[T_Out] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=lambda: cast(Dict[str, Any], {}))

    def __str__(self) -> str:
        """Return human-readable representation.

        Returns
        -------
        str
            Stage result summary.
        """
        status = "✓" if self.success else "✗"
        return f"{status} {self.stage_name} ({self.duration_ms:.1f}ms)"


class PipelineStage(ABC, Generic[T_In, T_Out]):
    """Abstract interface for a single pipeline stage.

    Stages are composable units of work that transform input into output.
    They are designed to be:
    - Reusable across different pipelines
    - Composable with other stages
    - Observable for logging/metrics
    - Testable in isolation

    Type Parameters
    ---------------
    T_In
        Type of input to this stage.
    T_Out
        Type of output from this stage.

    Examples
    --------
    Concrete stage implementation:

    >>> class PreprocessStage(PipelineStage[RawData, ProcessedData]):
    ...     @property
    ...     def name(self) -> str:
    ...         return "preprocess"
    ...
    ...     def can_execute(self, input_data: RawData) -> bool:
    ...         return input_data is not None
    ...
    ...     def execute(self, input_data: RawData) -> ProcessedData:
    ...         # Do preprocessing
    ...         return ProcessedData(...)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this pipeline stage.

        Used for logging, debugging, and result tracking.

        Returns
        -------
        str
            Unique stage name.
        """
        pass

    @abstractmethod
    def can_execute(self, input_data: T_In) -> bool:
        """Check if this stage can execute with given input.

        Allows conditional stage execution based on input state.
        Used by Pipeline to decide whether to run this stage.

        Parameters
        ----------
        input_data : T_In
            Input to check.

        Returns
        -------
        bool
            True if stage can execute, False to skip stage.
        """
        pass

    @abstractmethod
    def execute(self, input_data: T_In) -> T_Out:
        """Execute this stage on input data.

        Parameters
        ----------
        input_data : T_In
            Input data to process.

        Returns
        -------
        T_Out
            Stage output.

        Raises
        ------
        Exception
            If stage execution fails. Pipeline will catch and wrap in StageResult.
        """
        pass

    def __call__(self, input_data: T_In) -> T_Out:
        """Allow stage to be called as function.

        Parameters
        ----------
        input_data : T_In
            Input to process.

        Returns
        -------
        T_Out
            Processed output.
        """
        return self.execute(input_data)


class Pipeline(Generic[T_In, T_Out]):
    """Composable pipeline of analysis stages.

    Orchestrates execution of multiple stages in sequence, with support for:
    - Conditional stage execution
    - Result tracking and retrieval
    - Error handling and reporting
    - Event emission for observability

    Type Parameters
    ---------------
    T_In
        Type of input to first stage.
    T_Out
        Type of output from last stage.

    Examples
    --------
    Create and use a pipeline:

    >>> pipeline = Pipeline[RawData, FinalResult]("analysis_pipeline")
    >>> pipeline.add_stage(PreprocessStage())
    >>> pipeline.add_stage(AnalysisStage())
    >>> pipeline.add_stage(PostprocessStage())
    >>>
    >>> result = pipeline.execute(raw_data)
    >>> preprocessing_result = pipeline.get_stage_result("preprocess")
    """

    def __init__(self, name: str):
        """Initialize a new pipeline.

        Parameters
        ----------
        name : str
            Name of this pipeline (for logging).
        """
        self.name = name
        self._stages: List["PipelineStage[Any, Any]"] = []
        self._results: Dict[str, "StageResult[Any]"] = {}

    def add_stage(self, stage: "PipelineStage[Any, Any]") -> "Pipeline[Any, Any]":
        """Add a stage to the pipeline (fluent API).

        Parameters
        ----------
        stage : PipelineStage
            Stage to add to end of pipeline.

        Returns
        -------
        Pipeline
            Self for method chaining.

        Examples
        --------
        >>> pipeline = Pipeline("my_pipeline")
        >>> pipeline.add_stage(Stage1()).add_stage(Stage2()).add_stage(Stage3())
        """
        self._stages.append(stage)
        logger.debug(f"Added stage '{stage.name}' to pipeline '{self.name}'")
        return self

    def execute(self, input_data: T_In) -> T_Out:
        """Execute all stages in sequence.

        Processes input through each stage that can execute, tracking results.
        If a stage fails, execution stops and error is raised.

        Parameters
        ----------
        input_data : T_In
            Input to first stage.

        Returns
        -------
        T_Out
            Output from final stage.

        Raises
        ------
        RuntimeError
            If any stage fails during execution.

        Examples
        --------
        >>> pipeline = Pipeline("preprocess")
        >>> pipeline.add_stage(LoadStage())
        >>> pipeline.add_stage(ValidateStage())
        >>> pipeline.add_stage(FilterStage())
        >>> result = pipeline.execute(raw_data)
        """
        current = input_data
        logger.info(f"Starting pipeline execution: {self.name}")

        for stage in self._stages:
            # ensure start_time is always defined before any potential exception
            start_time = datetime.now()
            try:

                # Check if stage can execute with current data
                if not stage.can_execute(current):
                    logger.debug(f"Skipping stage '{stage.name}' (precondition failed)")
                    continue

                # Execute stage
                logger.debug(f"Executing stage: {stage.name}")
                output = stage.execute(current)

                # Record result
                duration = (datetime.now() - start_time).total_seconds() * 1000
                result: StageResult[Any] = StageResult(
                    stage_name=stage.name,
                    success=True,
                    output=output,
                    duration_ms=duration,
                )
                self._results[stage.name] = result
                logger.debug(f"Stage '{stage.name}' completed in {duration:.1f}ms")

                current = output

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                result = StageResult(
                    stage_name=stage.name,
                    success=False,
                    error=e,
                    duration_ms=duration,
                )
                self._results[stage.name] = result
                logger.error(f"Stage '{stage.name}' failed: {e}")
                raise RuntimeError(
                    f"Pipeline '{self.name}' failed at stage '{stage.name}': {e}"
                ) from e

        logger.info(f"Pipeline '{self.name}' completed successfully")
        return cast(T_Out, current)

    def get_stage_result(self, stage_name: str) -> Optional["StageResult[Any]"]:
        """Get result from a specific stage.

        Parameters
        ----------
        stage_name : str
            Name of stage to retrieve result for.

        Returns
        -------
        Optional[StageResult]
            Result if stage was executed, None otherwise.
        """
        return self._results.get(stage_name)

    def get_all_results(self) -> Dict[str, "StageResult[Any]"]:
        """Get results from all executed stages.

        Returns
        -------
        Dict[str, StageResult]
            Mapping of stage names to their results.
        """
        return dict(self._results)

    def get_stage_output(self, stage_name: str) -> Optional[Any]:
        """Get output data from a specific stage.

        Parameters
        ----------
        stage_name : str
            Name of stage to retrieve output from.

        Returns
        -------
        Optional[Any]
            Stage output if stage succeeded, None otherwise.
        """
        result = self._results.get(stage_name)
        return result.output if result and result.success else None

    def get_execution_summary(self) -> str:
        """Get human-readable summary of pipeline execution.

        Returns
        -------
        str
            Summary showing each stage's status and timing.

        Examples
        --------
        >>> print(pipeline.get_execution_summary())
        Pipeline: analysis (3 stages)
        ✓ preprocess (12.3ms)
        ✓ analysis (245.1ms)
        ✓ postprocess (8.7ms)
        Total: 266.1ms
        """
        if not self._results:
            return f"Pipeline '{self.name}': No stages executed"

        lines = [f"Pipeline: {self.name} ({len(self._results)} stages)"]
        total_time = 0.0

        for result in self._results.values():
            lines.append(f"  {result}")
            total_time += result.duration_ms

        lines.append(f"  Total: {total_time:.1f}ms")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation of pipeline.

        Returns
        -------
        str
            Representation showing name and stage count.
        """
        return f"Pipeline(name='{self.name}', stages={len(self._stages)})"


class ConditionalStage(PipelineStage[T_In, T_Out]):
    """Pipeline stage that executes based on a predicate function.

    Wraps another stage with a condition that must be true for
    stage execution. If condition is false, stage is skipped.

    Examples
    --------
    >>> stage = ConditionalStage(
    ...     inner_stage=AnalysisStage(),
    ...     condition=lambda data: data.has_property("attribute_x"),
    ...     name="conditional_analysis"
    ... )
    >>> # Stage only executes if data has property "attribute_x"
    """

    def __init__(
        self,
        inner_stage: PipelineStage[T_In, T_Out],
        condition: Callable[[T_In], bool],
        name: Optional[str] = None,
    ):
        """Initialize conditional stage.

        Parameters
        ----------
        inner_stage : PipelineStage
            Stage to conditionally execute.
        condition : Callable
            Predicate function that determines execution.
        name : str, optional
            Stage name (uses inner_stage name if not provided).
        """
        self._inner_stage = inner_stage
        self._condition = condition
        self._name = name or f"conditional[{inner_stage.name}]"

    @property
    def name(self) -> str:
        """Return stage name."""
        return self._name

    def can_execute(self, input_data: T_In) -> bool:
        """Check if condition is met and inner stage can execute.

        Parameters
        ----------
        input_data : T_In
            Input data to check.

        Returns
        -------
        bool
            True if condition is true and inner stage can execute.
        """
        try:
            condition_met = self._condition(input_data)
            inner_can_execute = self._inner_stage.can_execute(input_data)
            return condition_met and inner_can_execute
        except Exception as e:
            logger.warning(f"Condition check failed for {self.name}: {e}")
            return False

    def execute(self, input_data: T_In) -> T_Out:
        """Execute inner stage (after condition is verified).

        Parameters
        ----------
        input_data : T_In
            Input data to process.

        Returns
        -------
        T_Out
            Output from inner stage.
        """
        return self._inner_stage.execute(input_data)


class ParallelPipeline:
    """Execute multiple pipelines in parallel.

    NOT YET IMPLEMENTED - placeholder for future concurrent execution.

    When implemented, will support:
    - Running multiple pipelines concurrently
    - Gathering results from all pipelines
    - Error handling across parallel executions
    """

    def __init__(self, name: str):
        """Initialize parallel pipeline orchestrator.

        Parameters
        ----------
        name : str
            Name of this parallel pipeline.
        """
        self.name = name
        self._pipelines: List["Pipeline[Any, Any]"] = []

    def add_pipeline(self, pipeline: "Pipeline[Any, Any]") -> "ParallelPipeline":
        """Add a pipeline to execute in parallel.

        Parameters
        ----------
        pipeline : Pipeline
            Pipeline to execute concurrently.

        Returns
        -------
        ParallelPipeline
            Self for method chaining.
        """
        self._pipelines.append(pipeline)
        return self

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute all pipelines in parallel.

        Parameters
        ----------
        input_data : Any
            Input to all pipelines.

        Returns
        -------
        Dict[str, Any]
            Mapping of pipeline names to results.
        """
        results: Dict[str, Any] = {}
        for pipeline in self._pipelines:
            results[pipeline.name] = pipeline.execute(input_data)
        return results
