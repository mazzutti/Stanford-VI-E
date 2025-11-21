"""Compatibility shim: re-export decorators from `src.utils.decorators`.

This module previously implemented these decorators directly. The
implementations were moved to `src.utils.decorators` so lightweight
modules (like resampling) can import the decorators without depending on
the larger `analysis` package. Keep this shim so existing imports remain
valid.
"""

from __future__ import annotations

from src.utils.decorators import (
    log_execution,
    memoize,
    retry,
    time_operation,
    validate_input,
)

__all__ = [
    "log_execution",
    "time_operation",
    "validate_input",
    "memoize",
    "retry",
]
