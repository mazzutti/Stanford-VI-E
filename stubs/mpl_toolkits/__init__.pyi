"""Package marker for `mpl_toolkits` stubs used by mypy.

This file ensures `mpl_toolkits` is recognized as a package root
so submodules such as `mpl_toolkits.mplot3d` are not also treated
as top-level modules (e.g. `mplot3d`).
"""

__all__: list[str] = []
