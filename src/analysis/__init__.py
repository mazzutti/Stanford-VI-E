"""Analysis pipelines package (formerly `src.regenerate`).

This package contains multi-step pipelines for seismic and rock-physics
analysis. Invoke with e.g. `python -m src.analysis.seismograms`.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "common_imports",
    "common",
    "header",
    "rock_physics",
    "seismograms",
]
