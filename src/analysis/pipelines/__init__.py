"""Analysis pipelines subpackage for workflow orchestration.

This subpackage contains high-level analysis pipelines and workflow orchestration.

Public API:
    - Pipeline, PipelineStage, StageResult: Pipeline orchestration
    - SeismogramAnalyzer: Main class for seismogram analysis workflows

Example:
    >>> from src.analysis.pipelines import SeismogramAnalyzer, Pipeline
    >>> analyzer = SeismogramAnalyzer()
    >>> pipeline = Pipeline("my_analysis")
    >>> results = analyzer.run(...)
"""

from .orchestrator import (
    ConditionalStage,
    ParallelPipeline,
    Pipeline,
    PipelineStage,
    StageResult,
)
from .seismograms import SeismogramAnalyzer

__all__ = [
    "SeismogramAnalyzer",
    "Pipeline",
    "PipelineStage",
    "StageResult",
    "ConditionalStage",
    "ParallelPipeline",
]
