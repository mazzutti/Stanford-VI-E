from typing import Any
from matplotlib.colors import Colormap

# Minimal, permissive stubs for commonly used pyplot functions referenced
# in the codebase. Keep most return types as `Any` to avoid cross-module
# assignment issues; `get_cmap` returns the shared `Colormap` protocol
# defined in `matplotlib.colors` so call-sites like `cmap(...)` are accepted.

Figure = Any
Axes = Any
AxesImage = Any
Colorbar = Any

rcParams: dict[str, Any]

def get_cmap(name: str | None = ..., lut: int | None = ...) -> Colormap: ...
def colorbar(mappable: Any, *args: Any, **kwargs: Any) -> Any: ...
def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *args: Any,
    **kwargs: Any,
) -> tuple[Figure, Any]: ...
def imshow(*args: Any, **kwargs: Any) -> Any: ...
def figure(*args: Any, **kwargs: Any) -> Any: ...
def savefig(fname: Any, *args: Any, **kwargs: Any) -> None: ...
def close(fig: Any = ...) -> None: ...
def show(*args: Any, **kwargs: Any) -> None: ...
def tight_layout(*args: Any, **kwargs: Any) -> None: ...
def suptitle(*args: Any, **kwargs: Any) -> Any: ...

# Lightweight module-like object used by some code patterns `import matplotlib.pyplot as plt`
class _Plt:
    figure: Any
    get_cmap: Any
    tight_layout: Any
    close: Any

plt: _Plt

__all__ = [
    "get_cmap",
    "figure",
    "subplots",
    "colorbar",
    "imshow",
    "savefig",
    "close",
    "show",
]
