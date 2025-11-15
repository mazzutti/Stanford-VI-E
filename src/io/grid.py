"""Small utilities for grid metadata.

This module provides a typed dataclass `GridSpec` that encapsulates the
3D grid shape and spacing (dz, dt). It's useful to avoid passing loose
tuples around and gives a single place for grid-related helpers.
"""

from dataclasses import dataclass
import logging

# Type alias for clarity
GridShape = tuple[int, int, int]


@dataclass
class GridSpec:
    """Encapsulate a 3D grid shape and spacing.

    Attributes:
        shape: (nx, ny, nz) integer tuple
        dz: vertical/sample spacing for depth domain
        dt: time sampling (seconds) for time domain
    """

    shape: tuple[int, int, int]
    dz: float = 1.0
    dt: float = 0.001

    @classmethod
    def from_dimensions(
        cls, nx: int, ny: int, nz: int, dz: float = 1.0, dt: float = 0.001
    ) -> "GridSpec":
        """Convenience constructor from separate dimensions."""
        return cls((nx, ny, nz), dz=dz, dt=dt)

    @property
    def nx(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nz(self) -> int:
        return self.shape[2]

    def voxel_count(self) -> int:
        nx, ny, nz = self.shape
        return int(nx * ny * nz)

    def validate(self) -> None:
        nx, ny, nz = self.shape
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("Grid shape dimensions must be positive integers")

    def as_tuple(self) -> tuple[tuple[int, int, int], float, float]:
        return self.shape, self.dz, self.dt


__all__ = ["GridSpec"]

# Module logger for consistent logging across the package
logger = logging.getLogger(__name__)
