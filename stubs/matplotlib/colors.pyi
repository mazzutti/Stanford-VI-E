from typing import Protocol, Any, Sequence, Optional

class Colormap(Protocol):
    """Minimal Colormap protocol for typing in this repo."""

    name: str

    def __call__(self, x: Any) -> Any: ...
    def __getitem__(self, index: int) -> Any: ...

class ListedColormap:
    name: str
    colors: Sequence[Any]

    def __init__(
        self, colors: Sequence[Any], name: Optional[str] = None, N: Optional[int] = None
    ) -> None: ...
    def __call__(self, x: Any) -> Any: ...
    def __getitem__(self, index: int) -> Any: ...

class BoundaryNorm:
    def __init__(self, boundaries: Any, ncolors: int, clip: bool = False) -> None: ...
    def __call__(self, value: Any) -> Any: ...

def to_hex(c: Any) -> str: ...

__all__ = ["Colormap", "ListedColormap", "BoundaryNorm", "to_hex"]
