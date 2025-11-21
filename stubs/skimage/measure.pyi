"""Minimal stubs for `skimage.measure` used in this repository.

This provides typing for the marching cubes functions used by
`src.debug.plot3d` so the type checker recognizes their signatures.
"""

import numpy as np

def marching_cubes(
    volume: np.ndarray,
    level: float | None = ...,
    *,
    spacing: tuple[float, float, float] = ...,
    gradient_direction: str = ...,
    step_size: int = ...,
    allow_degenerate: bool = ...,
    method: str = ...,
    mask: np.ndarray | None = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def marching_cubes_lewiner(
    volume: np.ndarray,
    level: float | None = ...,
    *,
    spacing: tuple[float, float, float] = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

__all__ = ["marching_cubes", "marching_cubes_lewiner"]
