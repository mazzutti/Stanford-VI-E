"""Cache-related data models.

This module contains models for cached data and display cube results.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import ModelUtilities

__all__ = [
    "CacheLoadResult",
    "DisplayCubesResult",
]


@dataclass
class CacheLoadResult:
    """Represents cached data load results with validation."""

    avo: NDArray[np.float64]
    filename: str

    def __post_init__(self) -> None:
        """Validate data integrity."""
        if self.avo.size == 0:
            raise ValueError("AVO array cannot be empty")
        if not self.filename:
            raise ValueError("Filename cannot be empty")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the AVO array."""
        return self.avo.shape

    @property
    def size(self) -> int:
        """Return the total number of elements in AVO array."""
        return self.avo.size

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Return the data type of the AVO array."""
        return self.avo.dtype

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "filename": self.filename,
            "shape": self.shape,
            "size": self.size,
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheLoadResult:
        """Create result from dictionary representation.

        Note:
            This class intentionally does not support from_dict() because
            reconstructing NDArray objects from dictionaries requires external
            serialization (e.g., numpy or h5py). Use the constructor directly
            with the loaded avo array: `CacheLoadResult(avo=loaded_array, ...)`

        Raises:
            NotImplementedError: Always raises; use constructor instead.
        """
        raise NotImplementedError(
            "CacheLoadResult.from_dict() cannot reconstruct NDArray objects from dict format. "
            "Use the constructor directly: CacheLoadResult(avo=np.load(...), filename=...)"
        )

    def __str__(self) -> str:
        """Return string representation."""
        return f"CacheLoadResult(filename={self.filename}, shape={self.shape})"


@dataclass
class DisplayCubesResult:
    """Represents display cube data with shape validation."""

    avo_display: NDArray[np.float64]
    facies_display: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate cube dimensions match."""
        if self.avo_display.shape != self.facies_display.shape:
            raise ValueError(
                f"AVO and facies arrays must have the same shape. "
                f"Got {self.avo_display.shape} and {self.facies_display.shape}"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the 3D shape of the display cubes."""
        return self.avo_display.shape

    @property
    def volume(self) -> int:
        """Return the total number of voxels."""
        return self.avo_display.size

    @cached_property
    def avo_min(self) -> float:
        """Return cached minimum AVO value."""
        return float(np.min(self.avo_display))

    @cached_property
    def avo_max(self) -> float:
        """Return cached maximum AVO value."""
        return float(np.max(self.avo_display))

    @cached_property
    def avo_mean(self) -> float:
        """Return cached mean AVO value."""
        return float(np.mean(self.avo_display))

    @cached_property
    def avo_std(self) -> float:
        """Return cached standard deviation of AVO values."""
        return float(np.std(self.avo_display))

    @cached_property
    def avo_stats(self) -> dict[str, float]:
        """Return cached statistical summary of AVO data using helper function."""
        return ModelUtilities.compute_array_stats(self.avo_display)

    @property
    def facies_types(self) -> NDArray[np.int64]:
        """Return unique facies indices present in the cube."""
        return np.unique(self.facies_display)

    @cached_property
    def facies_count(self) -> int:
        """Return the number of unique facies types."""
        return len(self.facies_types)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "shape": self.shape,
            "volume": self.volume,
            "facies_count": self.facies_count,
            "facies_types": self.facies_types.tolist(),
            "avo_stats": self.avo_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisplayCubesResult:
        """Create result from dictionary representation.

        Note:
            This class intentionally does not support from_dict() because
            reconstructing NDArray objects from dictionaries requires external
            serialization (e.g., numpy or h5py). Use the constructor directly
            with the loaded arrays: `DisplayCubesResult(avo_display=..., facies_display=...)`

        Raises:
            NotImplementedError: Always raises; use constructor instead.
        """
        raise NotImplementedError(
            "DisplayCubesResult.from_dict() cannot reconstruct NDArray objects from dict format. "
            "Use the constructor directly: "
            "DisplayCubesResult(avo_display=np.load(...), facies_display=np.load(...))"
        )

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"DisplayCubesResult(shape={self.shape}, "
            f"facies_count={self.facies_count})"
        )
