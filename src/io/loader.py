"""Main data loader for Stanford VI-E dataset.

This module provides the primary API for loading Stanford VI-E dataset files.

Design:
- DatasetManager: Orchestrates loading all dataset files
- GslibLoader: Convenience singleton for loading individual GSLIB files
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.io.file_locator import FileLocator
from src.io.grid import GridSpec
from src.io.gslib_reader import GSLibConfig, GSLibReader

__all__ = ["DatasetManager", "GslibLoader"]

logger = logging.getLogger(__name__)


@dataclass
class DatasetManager:
    """Manager for loading and accessing Stanford VI-E dataset properties.

    Handles locating and reading GSLIB .dat files and organizing them
    into a structured dataset with grid metadata.

    Attributes
    ----------
    data_path : str
        Root path to the dataset directory.
    file_map : dict[str, str]
        Mapping of property keys to their folder names.
    grid_spec : GridSpec
        Grid specification for the dataset.
    vp : NDArray[np.float64] | None
        P-wave velocity data.
    vs : NDArray[np.float64] | None
        S-wave velocity data.
    rho : NDArray[np.float64] | None
        Density data.
    facies : NDArray[np.float64] | None
        Facies data.
    full_stack : NDArray[np.float64] | None
        Full stack seismic data.

    """

    data_path: str
    file_map: dict[str, str]
    grid_spec: GridSpec

    vp: NDArray[np.float64] | None = None
    vs: NDArray[np.float64] | None = None
    rho: NDArray[np.float64] | None = None
    facies: NDArray[np.float64] | None = None
    full_stack: NDArray[np.float64] | None = None

    _other: dict[str, NDArray[np.float64]] = field(
        default_factory=lambda: cast(dict[str, NDArray[np.float64]], {}), repr=False
    )
    _logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__), init=False, repr=False
    )
    _reader: GSLibReader = field(default_factory=GSLibReader, init=False, repr=False)
    _locator: FileLocator = field(default_factory=FileLocator, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate DatasetManager configuration after initialization.

        Raises
        ------
        ValueError
            If data_path or file_map is empty.
        """
        if not self.data_path or not self.data_path.strip():
            raise ValueError("data_path cannot be empty")
        if not self.file_map:
            raise ValueError("file_map cannot be empty")

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Get the grid shape from grid_spec.

        Returns
        -------
        tuple[int, ...]
            The shape of the grid.
        """
        return tuple(self.grid_spec.shape)

    @property
    def grid_size(self) -> int:
        """Get the total number of grid elements.

        Returns
        -------
        int
            Total number of elements in the grid.
        """
        return int(np.prod(self.grid_shape))

    def __repr__(self) -> str:
        """Return string representation for debugging.

        Returns
        -------
        str
            Detailed representation of the DatasetManager state.
        """
        loaded_props = [
            k
            for k in GSLibConfig.KNOWN_PROPERTIES
            if getattr(self, k, None) is not None
        ]
        other_props = list(self._other.keys())
        return (
            f"DatasetManager(data_path={self.data_path!r}, "
            f"grid_shape={self.grid_shape}, "
            f"loaded_properties={loaded_props}, "
            f"custom_properties={other_props})"
        )

    def __enter__(self) -> DatasetManager:
        """Context manager entry.

        Returns
        -------
        DatasetManager
            Returns self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type if an exception occurred.
        exc_val : BaseException | None
            Exception value if an exception occurred.
        __exc_tb : TracebackType | None
            Exception traceback if an exception occurred.

        """
        # No cleanup needed currently - provides hook for future resource management

    def _assign_property(self, key: str, arr: NDArray[np.float64]) -> None:
        """Assign a loaded array to the appropriate property or _other mapping.

        For known properties, assigns to the corresponding attribute.
        Unknown properties go into _other.

        Parameters
        ----------
        key : str
            The property key.
        arr : NDArray[np.float64]
            The data array to assign.

        Raises
        ------
        ValueError
            If attempting to set a known property that doesn't exist as an attribute.
        """
        if key in GSLibConfig.KNOWN_PROPERTIES:
            if not hasattr(self, key):
                raise ValueError(
                    f"Property '{key}' in KNOWN_PROPERTIES but not found as attribute"
                )
            setattr(self, key, arr)
        else:
            self._other[key] = arr

    def get_property(self, key: str) -> NDArray[np.float64] | None:
        """Get a loaded property by key.

        Provides uniform access to both known properties and custom properties.

        Parameters
        ----------
        key : str
            The property key (e.g., "vp", "vs", or custom key).

        Returns
        -------
        NDArray[np.float64] | None
            The loaded array, or None if not loaded or not found.

        """
        if key in GSLibConfig.KNOWN_PROPERTIES:
            return getattr(self, key, None)
        return self._other.get(key, None)

    def has_property(self, key: str) -> bool:
        """Check if a property is loaded.

        Parameters
        ----------
        key : str
            The property key to check.

        Returns
        -------
        bool
            True if the property is loaded (not None).
        """
        prop = self.get_property(key)
        return prop is not None

    def load(self) -> None:
        """Locate and read .dat files, assigning arrays to attributes.

        For each known property the corresponding dat file is read and the
        resulting array is assigned to the canonical attribute (e.g., `self.vp`).
        Unknown keys are stored in the `_other` mapping.

        Raises
        ------
        FileNotFoundError
            If required data folders or files are not found.
        ValueError
            If loaded arrays have incorrect dimensions or size.
        """
        expected_shape = self.grid_shape
        loaded_count = 0

        for key, folder_name in self.file_map.items():
            dir_path = Path(self.data_path) / folder_name
            full_path = self._locator.find(key, folder_name, dir_path)

            try:
                self._logger.info("Loading %s from %s...", key, full_path)
                arr = self._reader.read(full_path, expected_shape)
                self._assign_property(key, arr)
                loaded_count += 1
            except (OSError, ValueError) as e:
                self._logger.error("Failed to load %s from %s: %s", key, full_path, e)
                raise

        self._logger.info(
            "Successfully loaded %s properties. Grid shape: %s",
            loaded_count,
            expected_shape,
        )

    def align_cache_array(
        self,
        arr: NDArray[np.float64] | None,
        *,
        try_reshape: bool = True,
    ) -> NDArray[np.float64] | None:
        """Validate and align a cache array to this DatasetManager's grid.

        Ensures the provided array matches the manager's grid_spec.shape.
        If the incoming array has the same number of elements but a different
        shape, and try_reshape is True, the function attempts to reshape.

        Reshaping tries Fortran-order first (to match GSLIB ordering),
        then C-order as a fallback.

        Parameters
        ----------
        arr : NDArray[np.float64] | None
            The array to align, or None.
        try_reshape : bool, optional
            Whether to attempt reshaping if dimensions match but shape differs.

        Returns
        -------
        NDArray[np.float64] | None
            The aligned array as an ndarray of dtype float64, or None if
            alignment failed or input was None.
        """
        if arr is None:
            return None

        data = np.asarray(arr)
        expected = self.grid_shape

        # Exact shape match
        if data.shape == expected:
            return data.astype(np.float64)

        # If same number of elements, try reshaping
        if try_reshape and data.size == self.grid_size:
            # Try Fortran order first (matches GSLIB usage)
            try:
                reshaped = data.reshape(expected, order="F")
                return reshaped.astype(np.float64)
            except (ValueError, RuntimeError) as e:
                self._logger.debug(
                    "Fortran-order reshape failed: %s. Trying C-order reshape...",
                    e,
                )

            # Fallback to C-order reshape
            try:
                reshaped = data.reshape(expected, order="C")
                return reshaped.astype(np.float64)
            except (ValueError, RuntimeError) as e:
                self._logger.debug("C-order reshape also failed: %s", e)

        # Could not align
        self._logger.debug(
            "Cache array shape %s cannot be aligned to grid shape %s",
            data.shape,
            expected,
        )
        return None

    @classmethod
    def from_stanfordsix(
        cls, data_path: str, file_map: dict[str, str], grid_spec: GridSpec
    ) -> DatasetManager:
        """Create a DatasetManager for the Stanford-VI-E layout and load data.

        This is a factory method that creates a DatasetManager instance,
        loads all data files, and returns the populated manager.

        Parameters
        ----------
        data_path : str
            Root path to the dataset directory.
        file_map : dict[str, str]
            Mapping of property keys to their folder names.
        grid_spec : GridSpec
            Grid specification for the dataset.

        Returns
        -------
        DatasetManager
            A fully initialized and loaded DatasetManager instance.

        Raises
        ------
        FileNotFoundError
            If required data folders or files are not found.
        ValueError
            If loaded arrays have incorrect dimensions or size.
        """
        dm = cls(data_path=data_path, file_map=file_map, grid_spec=grid_spec)
        dm.load()
        return dm


class GslibLoader:
    """Singleton factory for loading GSLIB files.

    This class provides a convenient interface to read individual GSLIB files
    and manages a singleton instance for module-level access.

    Attributes
    ----------
    _instance : GslibLoader | None
        Singleton instance.
    _reader : GSLibReader | None
        Shared reader instance.

    """

    _instance: GslibLoader | None = None
    _reader: GSLibReader | None = None

    def __new__(cls) -> GslibLoader:
        """Ensure singleton pattern for the default instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._reader = GSLibReader()
        return cls._instance

    def read(self, filepath: str | Path, grid_spec: GridSpec) -> NDArray[np.float64]:
        """Read a GSLIB file and return a 3D NumPy array.

        Parameters
        ----------
        filepath : str | Path
            Path to the GSLIB .dat file.
        grid_spec : GridSpec
            Grid specification for reshaping the data.

        Returns
        -------
        NDArray[np.float64]
            The loaded data reshaped to the grid specification.

        """
        assert self._reader is not None, "GSLibReader not initialized"
        return self._reader.read(filepath, grid_spec.shape)

    @classmethod
    def get_instance(cls) -> GslibLoader:
        """Get the singleton instance of GslibLoader.

        Returns
        -------
        GslibLoader
            The singleton instance.
        """
        return cls()
