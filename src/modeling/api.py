"""Programmatic API for the modeling pipeline.

High-level convenience functions for AVO modeling.
Uses ModelingPipeline for orchestration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any
import logging

from src.modeling.pipeline import ModelingPipeline
from src.modeling.config import ModelingConfig, ModelingDefaults

__all__ = ["run_full_modeling"]

logger = logging.getLogger(__name__)


def run_full_modeling(
    cache_dir: str = ".cache",
    add_avo_noise: bool = False,
    *,
    skip_cleanup: bool = False,
    verbose: bool = False,
) -> dict[
    str, bool | list[NDArray[np.floating[Any]]] | None | NDArray[np.floating[Any]]
]:
    """Run the full modeling pipeline from depth to time domain.

    Orchestrates: data loading, depth-to-time resampling, and AVO synthesis.

    This is a convenience wrapper around ModelingPipeline with sensible defaults.

    Args:
        cache_dir: Cache directory for synthetics
        add_avo_noise: Add realistic angle-dependent noise

    Returns:
        Dictionary with modeling results:
        - 'avo_cached': bool - whether result came from cache
        - 'angle_stacks': list[np.ndarray] - per-angle seismic stacks
        - 'full_stack': NDArray[np.floating[Any]] - combined seismic stack

    Example:
        >>> result = run_full_modeling(cache_dir=".cache", add_avo_noise=True)
        >>> full_stack = result['full_stack']
    """
    # Create config with standard settings
    defaults = ModelingDefaults(cache_dir=cache_dir)
    config = ModelingConfig(
        defaults=defaults,
        add_noise=add_avo_noise,
        use_quality_weighting=True,
        snr_db=20.0,
        cache_enabled=True,
    )

    # Run pipeline
    pipeline = ModelingPipeline(config=config)
    return pipeline.run()
