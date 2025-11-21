"""Small utilities for grid metadata.

This module provides a typed dataclass `GridSpec` that encapsulates the
3D grid shape and spacing (dz, dt). It's useful to avoid passing loose
tuples around and gives a single place for grid-related helpers.
"""

import logging
from dataclasses import dataclass

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
        """Convenience constructor from separate dimensions.

        Keeps the classic signature used across the codebase for callers.
        """
        # Accept a somewhat long argument list for convenience; keep
        # compatibility with existing callers. Disable the too-many-args
        # warning locally.

        return cls((nx, ny, nz), dz=dz, dt=dt)

    @property
    def nx(self) -> int:
        """Number of samples in the X dimension."""
        return self.shape[0]

    @property
    def ny(self) -> int:
        """Number of samples in the Y dimension."""
        return self.shape[1]

    @property
    def nz(self) -> int:
        """Number of samples in the Z/depth dimension."""
        return self.shape[2]

    def voxel_count(self) -> int:
        """Return the total number of voxels (nx * ny * nz)."""
        nx, ny, nz = self.shape
        return int(nx * ny * nz)

    def validate(self) -> None:
        """Validate that the grid shape dimensions are positive integers."""
        nx, ny, nz = self.shape
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("Grid shape dimensions must be positive integers")

    def as_tuple(self) -> tuple[tuple[int, int, int], float, float]:
        """Return `(shape, dz, dt)` tuple for convenience.

        Useful for compatibility with older helpers that expect a tuple.
        """
        return self.shape, self.dz, self.dt


__all__ = ["GridSpec"]

# Module logger for consistent logging across the package
logger = logging.getLogger(__name__)
