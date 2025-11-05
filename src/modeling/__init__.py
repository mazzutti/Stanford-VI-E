"""AVO modeling and synthesis module.

Provides fully object-oriented interfaces for AVO seismogram synthesis,
caching, and pipeline orchestration.

Main classes:
    - AVOSynthesizer: AVO synthesis with angle-dependent stacks
    - AngleModel: Angle-dependent weights and noise characteristics
    - CacheManager: Caching infrastructure for modeling results
    - ModelingPipeline: Complete workflow orchestrator
"""

from src.modeling.modeling import (
    AVOSynthesizer,
    AngleModel,
    CacheHasher,
)
from src.modeling.model_cache import CacheManager
from src.modeling.api import ModelingPipeline

__all__ = [
    # Core classes
    "AVOSynthesizer",
    "AngleModel",
    "CacheHasher",
    "CacheManager",
    "ModelingPipeline",
]
