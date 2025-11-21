"""Cache for rock physics model derived attributes."""

from typing import Any

from numpy.typing import NDArray

from src.io.disk_cache import DiskCache

__all__ = ["ModelCache"]


class ModelCache:
    """Manages caches for derived attributes of a rock physics model.

    Keeps cache logic separate from the data model, allowing for cleaner
    separation of concerns and easier testing.
    """

    def __init__(self, disk_cache: DiskCache | None = None):
        """Initialize cache manager.

        Args:
            disk_cache: Optional shared disk cache for expensive results
        """
        self.disk_cache = disk_cache
        self._derived_cache: NDArray[Any] | None = None
        self._refl_cache: NDArray[Any] | None = None

    def invalidate(self) -> None:
        """Invalidate all internal caches."""
        self._derived_cache = None
        self._refl_cache = None

    def get_derived(self) -> NDArray[Any] | None:
        """Get cached derived attributes."""
        return self._derived_cache

    def set_derived(self, data: NDArray[Any]) -> None:
        """Cache derived attributes."""
        self._derived_cache = data

    def get_reflectivity(self) -> NDArray[Any] | None:
        """Get cached reflectivity."""
        return self._refl_cache

    def set_reflectivity(self, data: NDArray[Any]) -> None:
        """Cache reflectivity."""
        self._refl_cache = data
