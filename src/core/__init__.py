"""`src.core` package root.

This module intentionally exposes a tiny, low-dependency surface for
lightweight types (e.g. `Buildable`) while providing lazy, backward
compatible access to historically exported symbols. Accessing heavy
implementations (base classes, factories, validators) will import the
corresponding submodule lazily on attribute access.

This approach preserves the old public API for external callers and
tests while avoiding import-time side-effects that lead to cycles.
"""

from __future__ import annotations

import importlib
from typing import Any

from .types import Buildable, ValidatorResult

__all__ = ["Buildable", "ValidatorResult"]


# Order of submodules to attempt when resolving an attribute lazily.
_SUBMODULES = [
    "configuration",
    "factory",
    "analyzers",
    "processors",
    "validation",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised by tests
    """Lazily import and return attributes from core submodules.

    Tries each submodule in `_SUBMODULES` and returns the first attribute
    that matches `name`. Raises AttributeError if not found.
    """
    for sub in _SUBMODULES:
        try:
            module = importlib.import_module(f".{sub}", package=__package__)
        except (ImportError, ModuleNotFoundError):
            # Ignore missing optional submodules; other import-time errors
            # will propagate (since we only catch the import-related errors
            # here), which helps surface genuine bugs in submodules.
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    names = list(globals().keys()) + list(__all__)
    return sorted(set(names))
