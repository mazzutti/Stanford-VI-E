"""Facies correlation analysis subpackage.

This subpackage implements quantitative analysis of seismic-facies correlation
through gradient correlation, boundary amplitude extraction, interface reflection
aggregation, and facies discrimination.

Main Components
---------------
- FaciesCorrelationAnalyzer: Main orchestrator for the analysis pipeline
- AnalysisPipeline: Multi-stage workflow orchestration
- Domain enum: DEPTH or TIME analysis domain selection
- FaciesCorrelationConfig: Configuration class for analysis parameters

Quick Start
-----------
>>> from src.analysis.facies import FaciesCorrelationAnalyzer, Domain
>>> analyzer = FaciesCorrelationAnalyzer()
>>> fig = analyzer.run(cache_dir=".cache", domain=Domain.DEPTH)
"""

from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer
from src.analysis.types import Domain
from src.analysis.models import FaciesCorrelationConfig

__all__ = [
    "FaciesCorrelationAnalyzer",
    "Domain",
    "FaciesCorrelationConfig",
]
