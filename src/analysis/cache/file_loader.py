"""Simplified file loading strategy for cache operations.

This module provides a cleaner, more maintainable approach to loading
cache files by extracting file loading logic into dedicated classes.
"""

from pathlib import Path
from typing import Any, cast
from collections.abc import Callable
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ["FileLoader", "NPZExtractor"]


class NPZExtractor:
    """Extract arrays from NPZ archives with a simple, clear interface."""

    DEFAULT_KEY = "full_stack"

    @classmethod
    def extract(
        cls, archive: NpzFile, key: str | None = None
    ) -> NDArray[Any] | None:
        """Extract array from NPZ archive.

        Args:
            archive: Loaded NPZ archive
            key: Specific key to extract (defaults to 'full_stack')

        Returns:
            Extracted array or None if extraction fails
        """
        try:
            # Try specified key or default
            target_key = key or cls.DEFAULT_KEY
            if target_key in archive:
                return np.asarray(archive[target_key])

            # Fallback to first available array
            files = getattr(archive, "files", [])
            if files:
                return np.asarray(archive[files[0]])

        except Exception as e:
            logger.error(f"Failed to extract from NPZ: {e}")

        return None


class FileLoader:
    """Simplified file loading with clean separation of concerns."""

    def __init__(
        self,
        np_load: Callable[..., Any] = np.load,
        extractor: NPZExtractor | None = None,
    ) -> None:
        """Initialize file loader.

        Args:
            np_load: NumPy load function (for testing)
            extractor: NPZ extraction strategy
        """
        self._np_load = np_load
        self._extractor = extractor or NPZExtractor()

    def load(
        self,
        path: Path,
        mmap_mode: str | None = None,
        convert_to_float64: bool = True,
    ) -> NDArray[Any] | None:
        """Load file with simplified logic.

        Args:
            path: File path to load
            mmap_mode: Memory mapping mode ('r', 'r+', etc)
            convert_to_float64: Convert arrays to float64 (not for memmap)

        Returns:
            Loaded array or None on error
        """
        try:
            # Load file
            loaded = self._load_file(path, mmap_mode)

            # Handle memory-mapped arrays (return as-is)
            if isinstance(loaded, np.memmap):
                return loaded

            # Handle NPZ archives
            if isinstance(loaded, NpzFile):
                return self._handle_npz(loaded)

            # Convert regular arrays to float64 if requested
            if convert_to_float64:
                return np.asarray(loaded, dtype=np.float64)

            return np.asarray(loaded)

        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def _load_file(
        self, path: Path, mmap_mode: str | None
    ) -> NDArray[Any] | NpzFile:
        """Load file using NumPy."""
        kwargs: dict[str, Any] = {"allow_pickle": False}
        if mmap_mode:
            kwargs["mmap_mode"] = mmap_mode
        # np.load may return different types; cast to the declared union for callers
        return cast(NDArray[Any] | NpzFile, self._np_load(str(path), **kwargs))

    def _handle_npz(self, loaded: NpzFile) -> NDArray[Any] | None:
        """Extract array from NPZ archive."""
        with loaded as archive:
            return self._extractor.extract(archive)
