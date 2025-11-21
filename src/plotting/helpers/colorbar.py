"""Helpers to construct Plotly colorbar dicts for surface traces.

Centralizing colorbar construction reduces duplicated dict literals and
keeps `traces.py` focused on mesh / trace geometry.
"""

from __future__ import annotations

from typing import Any

def make_plotly_colorbar(
    k_unit: str | None,
    show_colorbar: bool,
    colorbar_len: float,
    *,
    for_inline: bool = False,
) -> dict[str, Any] | None:
    """Return a Plotly `colorbar` dict or None.

    If `for_inline` is True, produce a colorbar layout oriented for the
    inline trace (positioned to the right). Otherwise return a standard
    colorbar dict used for depth traces.
    """
    if not show_colorbar:
        return None

    title = f"Value ({k_unit})" if k_unit else "Value"
    if for_inline:
        return {"title": title, "x": 1.02, "len": colorbar_len}

    return {"title": title, "thickness": 20, "len": colorbar_len}
