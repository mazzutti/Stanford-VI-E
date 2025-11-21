"""Cache archive and data extraction strategies.

This module provides extraction logic for converting various cache file
formats (NPZ, NPY) into usable NumPy arrays. The extraction logic is
separated from the cache loader to improve maintainability and testability.

Core Components:
    - ArrayExtractor: Base extractor for all formats
    - NpzExtractor: Specialized NPZ archive extraction
    - NpyExtractor: NPY file handling
    - ExtractorFactory: Creates appropriate extractors

Design Notes:
    - Extractors follow the Strategy pattern
    - Each format has dedicated extraction logic
    - Extraction is separate from loading (single responsibility)
    - Designed for easy testing and extension

Example Usage:
    >>> from src.analysis.cache.extractors import ExtractorFactory
    >>> extractor = ExtractorFactory.for_path("data.npz")
    >>> data = extractor.extract_data(archive_or_array)
"""

import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "ArrayExtractor",
    "NpzExtractor",
    "NpyExtractor",
    "ExtractorFactory",
]

# Constants
_FULL_STACK_KEY = "full_stack"

class ArrayExtractor(ABC):
    """Abstract base for array extraction from various file formats.

    Defines the interface that all extractors must implement, ensuring
    consistent handling of different cache file formats.
    """

    @abstractmethod
    def extract(self, source: object) -> NDArray[np.float64] | None:
        """Extract array data from source object.

        Parameters
        ----------
        source : object
            The source to extract from (depends on extractor type).

        Returns
        -------
        NDArray[np.float64] | None
            Extracted array as float64, or None if extraction failed.

        """

class NpzExtractor(ArrayExtractor):
    """Extracts arrays from NPZ (compressed) archive files.

    NPZ files can contain multiple arrays. This extractor uses a
    priority-based selection strategy:
    1. Look for 'full_stack' key (project convention)
    2. Return first available array
    3. Return None on failure

    Attributes
    ----------
    full_stack_key : str
        The preferred key to search for in the archive.
    """

    def __init__(self, full_stack_key: str = _FULL_STACK_KEY):
        """Initialize NPZ extractor.

        Parameters
        ----------
        full_stack_key : str, default="full_stack"
            The key name to look for first in the archive.
        """
        self.full_stack_key = full_stack_key

    def extract(self, source: object) -> NDArray[np.float64] | None:
        """Extract array from NPZ archive using priority strategy.

        Parameters
        ----------
        source : object
            Open NPZ archive object.

        Returns
        -------
        NDArray[np.float64] | None
            Extracted array as float64, or None on error.

        Raises
        ------
        CacheExtractionError
            If extraction fails and raise_on_error is enabled.

        Examples
        --------
        >>> import numpy as np
        >>> extractor = NpzExtractor()
        >>> # With open NPZ file:
        >>> # archive = np.load("data.npz")
        >>> # data = extractor.extract(archive)

        """
        try:
            # Cast to NpzFile for type checking
            archive = cast(NpzFile, source)

            # Priority 1: Look for full_stack key
            if self.full_stack_key in archive:
                result = np.asarray(archive[self.full_stack_key])
                logger.debug("Extracted array using key '%s'", self.full_stack_key)
                return result

            # Priority 2: Use first available array
            files = getattr(archive, "files", [])
            if files:
                first_key = files[0]
                result = np.asarray(archive[first_key])
                logger.debug("Extracted array using first key '%s'", first_key)
                return result

            # No arrays found
            logger.warning("NPZ archive appears to be empty")
            return None

        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            MemoryError,
        ) as exc:
            # NPZ extraction can fail for corrupt archives, unexpected contents,
            # or type/shape problems. Catch the expected error classes, log the
            # original exception for diagnostics, and return None so callers can
            # handle the missing data gracefully.
            logger.exception(
                "Failed to extract from NPZ archive: %s: %s", type(exc).__name__, exc
            )
            return None

class NpyExtractor(ArrayExtractor):
    """Extracts arrays from NPY (uncompressed) single-array files.

    NPY files contain a single array. This extractor simply ensures
    the data is converted to float64 format.
    """

    def extract(self, source: object) -> NDArray[np.float64] | None:
        """Extract array from NPY file.

        Parameters
        ----------
        source : object
            The loaded NPY array.

        Returns
        -------
        NDArray[np.float64] | None
            Array converted to float64, or None on error.

        Examples
        --------
        >>> import numpy as np
        >>> extractor = NpyExtractor()
        >>> data = np.random.randn(10, 10)
        >>> result = extractor.extract(data)

        """
        try:
            result = np.asarray(source).astype(np.float64)
            logger.debug(
                "Extracted NPY array, shape=%s, dtype=%s", result.shape, result.dtype
            )
            return result
        except (TypeError, ValueError, MemoryError, OSError) as exc:
            # NPY conversion may raise type/shape related errors or memory
            # allocation failures. Catch expected exceptions, log them, and
            # return None so callers can fall back or report missing data.
            logger.exception(
                "Failed to extract from NPY array: %s: %s", type(exc).__name__, exc
            )
            return None

class ExtractorFactory:
    """Factory for creating appropriate extractors based on file type.

    This factory follows the Strategy pattern to select the correct
    extraction logic based on file extension or explicit type.

    Examples
    --------
    By file path:
    >>> factory = ExtractorFactory()
    >>> extractor = factory.for_path("data.npz")  # Returns NpzExtractor
    >>> extractor = factory.for_path("data.npy")  # Returns NpyExtractor

    By explicit type:
    >>> extractor = ExtractorFactory.npz()
    >>> extractor = ExtractorFactory.npy()
    """

    _npz_extension = ".npz"
    _npy_extension = ".npy"

    @classmethod
    def for_path(cls, path: str | PathLike[str]) -> ArrayExtractor:
        """Get appropriate extractor for file path.

        Parameters
        ----------
        path : str | PathLike[str]
            File path to determine extractor from.

        Returns
        -------
        ArrayExtractor
            NpzExtractor for .npz files, NpyExtractor for .npy files.

        Raises
        ------
        ValueError
            If file extension is not recognized.

        Examples
        --------
        >>> extractor = ExtractorFactory.for_path("data.npz")
        >>> extractor = ExtractorFactory.for_path("/path/to/data.npy")

        """
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix == cls._npz_extension:
            logger.debug("Selected NpzExtractor for %s", p.name)
            return cls.npz()
        if suffix == cls._npy_extension:
            logger.debug("Selected NpyExtractor for %s", p.name)
            return cls.npy()
        raise ValueError(
            f"Unsupported file extension '{suffix}' for {p.name}. "
            f"Expected {cls._npz_extension} or {cls._npy_extension}"
        )

    @staticmethod
    def npz() -> NpzExtractor:
        """Create NPZ archive extractor.

        Returns
        -------
        NpzExtractor
            Configured NPZ extractor instance.
        """
        return NpzExtractor()

    @staticmethod
    def npy() -> NpyExtractor:
        """Create NPY file extractor.

        Returns
        -------
        NpyExtractor
            Configured NPY extractor instance.
        """
        return NpyExtractor()
