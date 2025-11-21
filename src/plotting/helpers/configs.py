"""Small configuration dataclasses for Plotly trace and color helpers.

These dataclasses reduce long parameter lists when constructing traces
and colorscale/bounds, improving readability and reducing pylint
complaints about too-many-arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ColorsConfig:
    """Configuration describing colorscale selection.

    Attributes
    - colorscale: matplotlib colormap name or explicit Plotly colorscale
    - is_categorical: whether to use categorical palette and fixed bounds
    """

    colorscale: str | list[list[float | str]] = "RdBu"
    is_categorical: bool = False

@dataclass(frozen=True)
class TraceConfig:
    """Configuration for trace construction.

    This groups commonly passed parameters into one object to reduce the
    number of positional arguments in helper functions.
    """

    k_scale: float = 1.0
    colorscale_to_use: Any | None = None
    cmin: float | None = None
    cmax: float | None = None
    show_colorbar: bool = False
    k_unit: str | None = None
    colorbar_len: float = 0.75
