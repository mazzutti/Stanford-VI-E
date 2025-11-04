"""Plotting package for all plotting functionality.

This package provides canonical locations for plotting utilities and scripts.
Import from `src.plotting.<module>`.

It also initializes matplotlib/NumPy once for importers by calling
`src.plotting.helpers.plot.init_plotting` so `src.plotting.plt` and
`src.plotting.np` are available to consumers.
"""

from importlib import import_module

from src.plotting.helpers.plot import init_plotting

# Initialize pyplot and numpy once for importers (Agg backend by default)
plt, np = init_plotting(backend="Agg")

# Re-export main plotting modules for convenience when importing the package
__all__ = [
    "plt",
    "np",
    "plot_2d_slices",
    "plot_3d_interactive",
    "plot_3d_slices",
    "plot_facies_overlay",
    "plot_multiangle_ei",
    "plot_rock_physics_attributes",
]


def _load(name: str):
    return _impl_load(name)


def _impl_load(name: str):
    """Canonical dynamic import used by the plotting package.

    Centralizing this dynamic import makes it easier to mock during tests
    and provides a single place to control import-time behavior for
    optional plotting backends.
    """
    return import_module(f"src.plotting.{name}")


for _m in list(__all__):
    # skip symbols provided directly by this package
    if _m in ("plt", "np"):
        globals()[_m] = globals().get(_m)
        continue
    try:
        globals()[_m] = _load(_m)
    except Exception:
        # lazy import - some plotting backends or optional deps may not be
        # available at import time; expose None so callers can handle it.
        globals()[_m] = None
