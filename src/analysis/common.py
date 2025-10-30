"""Shared helpers for analysis scripts (moved from regenerate_common).

This centralizes environment/venv helpers used by the analysis pipelines and
exposes a small set of commonly-used stdlib names for convenience.
"""

from pathlib import Path
import os

# Optional typing removed to avoid unused-import linter warnings

# Also expose common stdlib names that were previously available via
# `src.analysis.common_imports` so analysis modules can import them from here.
import sys
import time
import shutil

from src.processing.process import process_manager
import logging

logger = logging.getLogger(__name__)


# Keep helper factory but defer constructing the mapping until first use
def get_analysis_helpers():
    # Only export helpers that are safe for in-process execution. Subprocess
    # orchestration helpers have been removed in favor of direct in-process
    # programmatic APIs (see src.modeling.api).
    return {
        "clear_cache": process_manager.clear_cache,
        "open_file": process_manager.open_file,
        "summarize_cache_files": process_manager.summarize_cache_files,
    }


from src.utils.facades import LazyObjectProxy


# Lazy proxy for the ANALYSIS_HELPERS mapping to avoid import-time work.
# We return the concrete dict so callers that use dict-like accessors continue
# to work unchanged. The proxy will instantiate the dict on first use.
ANALYSIS_HELPERS = LazyObjectProxy(lambda: get_analysis_helpers())


def clear_cache(*args, **kwargs):
    return ANALYSIS_HELPERS["clear_cache"](*args, **kwargs)


def open_file(*args, **kwargs):
    return ANALYSIS_HELPERS["open_file"](*args, **kwargs)


def summarize_cache_files(*args, **kwargs):
    return ANALYSIS_HELPERS["summarize_cache_files"](*args, **kwargs)


__all__ = [
    "Path",
    "os",
    "sys",
    "time",
    "shutil",
    "clear_cache",
    "open_file",
    "summarize_cache_files",
]


# Object-oriented facade for analysis helpers
class AnalysisCommon:
    def __init__(self):
        self.helpers = get_analysis_helpers()

    def clear_cache(self, *args, **kwargs):
        return self.helpers["clear_cache"](*args, **kwargs)

    def open_file(self, *args, **kwargs):
        return self.helpers["open_file"](*args, **kwargs)

    def summarize_cache_files(self, *args, **kwargs):
        return self.helpers["summarize_cache_files"](*args, **kwargs)


# Module-level singleton for gradual migration (lazy proxy)


from src.utils.facades import LazyObjectProxy


# Module-level lazy proxy for AnalysisCommon
analysis_common = LazyObjectProxy(lambda: AnalysisCommon())


__all__.extend(["AnalysisCommon", "analysis_common"])


def get_analysis_common(instance: AnalysisCommon | None = None) -> "AnalysisCommon":
    """Return the provided AnalysisCommon instance or the module-level lazy singleton.

    This helper follows the repository pattern for get_* helpers and allows
    tests to pass their own instance.
    """
    return instance if instance is not None else analysis_common


__all__.append("get_analysis_common")
