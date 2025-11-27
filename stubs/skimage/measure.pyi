"""Minimal stubs for `skimage.measure` used in this repository.

This provides typing for the marching cubes functions used by
`src.debug.plot3d` so the type checker recognizes their signatures.
"""

from typing import Any
from numpy.typing import NDArray

def marching_cubes(
    volume: NDArray[Any],
    level: float | None = ...,
    *,
    spacing: tuple[float, float, float] = ...,
    gradient_direction: str = ...,
    step_size: int = ...,
    allow_degenerate: bool = ...,
    method: str = ...,
    mask: NDArray[Any] | None = ...,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]: ...
def marching_cubes_lewiner(
    volume: NDArray[Any],
    level: float | None = ...,
    *,
    spacing: tuple[float, float, float] = ...,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]: ...

__all__ = ["marching_cubes", "marching_cubes_lewiner"]
