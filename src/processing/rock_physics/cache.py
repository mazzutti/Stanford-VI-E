"""Cache for rock physics model derived attributes."""

from typing import Optional
import numpy as np

from src.io.disk_cache import DiskCache

__all__ = ["ModelCache"]


class ModelCache:
    """Manages caches for derived attributes of a rock physics model.

    Keeps cache logic separate from the data model, allowing for cleaner
    separation of concerns and easier testing.
    """

    def __init__(self, disk_cache: Optional[DiskCache] = None):
        """Initialize cache manager.

        Args:
            disk_cache: Optional shared disk cache for expensive results
        """
        self.disk_cache = disk_cache
        self._derived_cache: Optional[np.ndarray] = None
        self._refl_cache: Optional[np.ndarray] = None

    def invalidate(self) -> None:
        """Invalidate all internal caches."""
        self._derived_cache = None
        self._refl_cache = None

    def get_derived(self) -> Optional[np.ndarray]:
        """Get cached derived attributes."""
        return self._derived_cache

    def set_derived(self, data: np.ndarray) -> None:
        """Cache derived attributes."""
        self._derived_cache = data

    def get_reflectivity(self) -> Optional[np.ndarray]:
        """Get cached reflectivity."""
        return self._refl_cache

    def set_reflectivity(self, data: np.ndarray) -> None:
        """Cache reflectivity."""
        self._refl_cache = data
