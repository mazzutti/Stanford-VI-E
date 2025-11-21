"""Minimal type stubs for `matplotlib.cm` used by the project.

Provides a lightweight ``ScalarMappable`` and ``get_cmap`` stub so static
type-checkers (mypy, Pyright) can reason about code that calls
``cm.ScalarMappable(...)`` and ``cm.get_cmap(...)``.
"""

from __future__ import annotations

from typing import Any

from .colors import Colormap, Normalize

class ScalarMappable:
    """Simplified stub for ``matplotlib.cm.ScalarMappable``.

    Only the small surface used by the project is provided: initializer
    accepting ``cmap`` and ``norm``, and ``set_array``. Utility methods
    are included with permissive Any-typed signatures to help type
    checkers without constraining runtime usage.
    """

    cmap: Any
    norm: Normalize | None

    def __init__(self, cmap: Any = None, norm: Normalize | None = None) -> None: ...
    def set_array(self, A: Any) -> None: ...
    def to_rgba(self, X: Any, alpha: Any = None, bytes: bool = False) -> Any: ...

def get_cmap(name: str | None = ..., lut: int | None = ...) -> Colormap: ...

__all__ = ["get_cmap"]
