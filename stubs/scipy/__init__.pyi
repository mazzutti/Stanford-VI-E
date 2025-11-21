"""Minimal stub package marker for scipy.

This file ensures mypy recognizes `scipy` as a package and maps
submodule stubs (e.g. `scipy.stats`) correctly instead of exposing them
as top-level modules like `stats`.
"""

__all__: list[str]
