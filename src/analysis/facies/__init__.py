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

from src.analysis.domain.enum import Domain
from src.analysis.facies.analyzer import FaciesCorrelationAnalyzer
from src.analysis.facies.config import FaciesAnalysisConfig
from src.analysis.facies.processor_setup import register_facies_processors
from src.analysis.models import FaciesCorrelationConfig

# NOTE: Do not register processors at import time. Calling
# `register_facies_processors()` here introduced import-time side-effects
# that create import cycles (package-level registration-on-import).
#
# Callers (application bootstrap or CLIs) should invoke
# `src.analysis.facies.processor_setup.register_facies_processors()`
# explicitly during startup if they need the processors registered.

__all__ = [
    "FaciesCorrelationAnalyzer",
    "FaciesAnalysisConfig",
    "Domain",
    "FaciesCorrelationConfig",
    "register_facies_processors",
]
