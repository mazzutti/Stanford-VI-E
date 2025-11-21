"""Type stubs for `mpl_toolkits.mplot3d.art3d`.

Minimal shapes used by the project: `Poly3DCollection`.
"""

from typing import Any

class Poly3DCollection:
    def __init__(
        self,
        verts: Any,  # accept numpy arrays or nested sequences (e.g. verts[faces])
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def set_facecolor(self, color: Any) -> None: ...
    def set_edgecolor(self, color: Any) -> None: ...

__all__ = ["Poly3DCollection"]
