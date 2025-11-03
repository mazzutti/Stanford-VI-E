"""Analysis pipelines subpackage for workflow orchestration.

This subpackage contains high-level analysis pipelines and workflow orchestration.

Public API:
    - SeismogramAnalyzer: Main class for seismogram analysis workflows

Example:
    >>> from src.analysis.pipelines import SeismogramAnalyzer
    >>> analyzer = SeismogramAnalyzer()
    >>> results = analyzer.run(...)
"""

from .seismograms import (
    SeismogramAnalyzer,
)

__all__ = [
    "SeismogramAnalyzer",
]
