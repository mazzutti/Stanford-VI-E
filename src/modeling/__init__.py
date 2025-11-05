"""AVO modeling and synthesis module.

Provides fully object-oriented interfaces for AVO seismogram synthesis,
caching, and pipeline orchestration.

Main classes:
    - AVOSynthesizer: Angle-dependent AVO synthesis with convolution
    - AngleModel: Quality weights and noise characteristics for angles
    - SynthesisConfig: Configuration for synthesis parameters
    - CacheManager: Caching infrastructure for modeling results
"""

from src.modeling.modeling import (
    AVOSynthesizer,
    AngleModel,
    SynthesisConfig,
)
from src.modeling.model_cache import CacheManager
from src.modeling.api import run_full_modeling

__all__ = [
    # Core classes
    "AVOSynthesizer",
    "AngleModel",
    "SynthesisConfig",
    "CacheManager",
    # Pipeline API
    "run_full_modeling",
]
