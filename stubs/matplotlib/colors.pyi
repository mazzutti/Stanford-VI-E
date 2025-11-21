"""Type stubs for parts of `matplotlib.colors` used by this project.

Only a minimal `Normalize` class is provided to satisfy static
type-checkers (mypy, Pyright) when `from matplotlib.colors import
Normalize` is used.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

class Normalize:
    """Simplified type stub for ``matplotlib.colors.Normalize``.

    This stub includes the initializer signature and the commonly-used
    methods/attributes accessed by user code in this repository.
    """

    vmin: float | None
    vmax: float | None
    clip: bool

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        clip: bool = False,
    ) -> None: ...
    def __call__(self, value: Any) -> np.ndarray: ...
    def autoscale(self, A: Iterable[float]) -> None: ...
    def autoscale_None(self, A: Iterable[float] | None) -> None: ...
    def inverse(self, value: Any) -> np.ndarray: ...

from collections.abc import Sequence
from typing import Any, Protocol

class Colormap(Protocol):
    """Minimal Colormap protocol for typing in this repo."""

    name: str

    def __call__(self, x: Any) -> Any: ...
    def __getitem__(self, index: int) -> Any: ...

class ListedColormap:
    name: str
    colors: Sequence[Any]

    def __init__(
        self, colors: Sequence[Any], name: str | None = None, N: int | None = None
    ) -> None: ...
    def __call__(self, x: Any) -> Any: ...
    def __getitem__(self, index: int) -> Any: ...

class BoundaryNorm:
    def __init__(self, boundaries: Any, ncolors: int, clip: bool = False) -> None: ...
    def __call__(self, value: Any) -> Any: ...

def to_hex(c: Any) -> str: ...

__all__ = ["Colormap", "ListedColormap", "BoundaryNorm", "to_hex"]
