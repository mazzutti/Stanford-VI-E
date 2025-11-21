__all__: list[str]
__all__ = ["pyplot"]
"""Package shim: re-export submodule stubs for matplotlib.

Pyright will resolve submodule stubs placed under `stubs/matplotlib/`.
This file re-exports the primary names used by the codebase.
"""

from .axes import Axes
from .colorbar import Colorbar
from .figure import Figure
from .image import AxesImage
from .pyplot import plt

def use(backend: str) -> None: ...

__all__ = ["Figure", "Axes", "AxesImage", "Colorbar", "plt", "use"]
