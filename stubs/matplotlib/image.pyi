"""Compatibility shim: expose `AxesImage` under `matplotlib.image`.

This module aliases `AxesImage` to the canonical type defined in
`matplotlib.axes` so type checkers see a single consistent type.
"""

from matplotlib.axes import AxesImage

__all__ = ["AxesImage"]
