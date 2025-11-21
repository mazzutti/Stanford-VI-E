"""AVO modeling and synthesis module.

Provides fully object-oriented interfaces for AVO seismogram synthesis,
caching, and pipeline orchestration.

Core classes:
    - AVOSynthesizer: Angle-dependent AVO synthesis with convolution
    - AngleModel: Quality weights and noise characteristics for angles
    - SynthesisConfig: Configuration for synthesis parameters
    - CacheManager: Caching infrastructure for modeling results

Processors:
    - ReflectivityComputer: Zoeppritz-based reflectivity computation
    - WaveletConvolver: Efficient 3D wavelet convolution

Pipeline:
    - ModelingDefaults: Centralized defaults for all parameters
    - ModelingConfig: Configuration for a modeling run
    - ModelingPipeline: High-level orchestration with simplified API
    - ResamplingService: Depth-to-time resampling service

Convenience:
    - run_full_modeling: Simple function for quick pipeline execution
"""

from src.modeling.api import run_full_modeling
from src.modeling.config import ModelingConfig, ModelingDefaults
from src.modeling.model_cache import CacheManager
from src.modeling.modeling import AngleModel, AVOSynthesizer, SynthesisConfig
from src.modeling.pipeline import ModelingPipeline
from src.modeling.processors import ReflectivityComputer, WaveletConvolver
from src.modeling.resampler import ResamplingService

__all__ = [
    # Core classes
    "AVOSynthesizer",
    "AngleModel",
    "SynthesisConfig",
    "CacheManager",
    # Processors
    "ReflectivityComputer",
    "WaveletConvolver",
    # Configuration
    "ModelingDefaults",
    "ModelingConfig",
    # Pipeline
    "ModelingPipeline",
    "ResamplingService",
    # Convenience API
    "run_full_modeling",
]
