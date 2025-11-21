"""Debug helpers for interactive development and visualization.

This package contains utilities intended only for debugging and exploratory
workflows (not for production code). Use the functions directly from
`src.debug.plot3d` when needed.
"""

# This package intentionally contains small, duplicated shims and re-exports
# used for interactive debugging and examples. Silence duplicate-code noise
# coming from these developer-focused utilities.
# pylint: disable=duplicate-code
# The disable above is deliberate: these utilities are developer-facing
# convenience shims and intentionally duplicate small helper code used
# in interactive sessions and examples.

from .plot3d import plot_volume

__all__ = ["plot_volume"]
