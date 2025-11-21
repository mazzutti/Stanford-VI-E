"""Lightweight package initializer for the analysis package.

This module purposefully avoids importing heavy submodules at import-time
to prevent large circular import graphs during test collection and runtime.
Import specific submodules (e.g. ``src.analysis.processors`` or
``src.analysis.models``) directly where needed.
"""

from __future__ import annotations

import logging

from .common import AnalysisCommon
from .exceptions import AnalysisException, ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisCommon",
    "AnalysisException",
    "ValidationError",
]
