"""Data loading helpers.

This module contains utilities to load GSLIB files and the Stanford VI-E
dataset used by the project.
"""

from pathlib import Path
import numpy as np
import logging

from dataclasses import dataclass, field
from typing import Dict, Optional, Union, FrozenSet, Tuple as TupleType
from src.io.grid import GridSpec
from numpy.typing import NDArray
from types import TracebackType


class GSLibConfig:
    """Configuration for GSLIB file reading.

    This class encapsulates all GSLIB-related constants and patterns
    in a single, organized location following OOP principles.
    """

    HEADER_LINES: int = 3
    """Number of header lines to skip when reading GSLIB files."""

    KNOWN_PROPERTIES: FrozenSet[str] = frozenset(
        {"vp", "vs", "rho", "facies", "full_stack"}
    )
    """Set of known property keys that map to class attributes."""

    VELOCITY_PATTERNS: Dict[str, str] = {
        "p-wave": "Pvelocity.dat",
        "s-wave": "Svelocity.dat",
    }
    """Mapping of velocity folder name patterns to special filename conventions."""


class GSLibReader:
    """Reader for GSLIB format files.

    Encapsulates all logic for reading, parsing, and validating GSLIB .dat files.
    """

    def __init__(self) -> None:
        """Initialize the GSLibReader with logging."""
        self._logger = logging.getLogger(__name__)

    def read(
        self, filepath: Union[str, Path], shape: TupleType[int, ...]
    ) -> NDArray[np.float64]:
        """Read a GSLIB `.dat` file and return a 3D NumPy array.

        The GSLIB files used here include a short header (GSLibConfig.HEADER_LINES lines)
        followed by a single column of numeric values in Fortran ordering. We skip
        the header lines and reshape with order="F".

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the GSLIB .dat file.
        shape : Tuple[int, ...]
            Target shape for the reshaped array.

        Returns
        -------
        NDArray[np.float64]
            The data reshaped to the specified shape.

        Raises
        ------
        OSError
            If the file cannot be read.
        ValueError
            If the data cannot be reshaped to the target shape or has wrong size.
        """
        filepath = Path(filepath)

        # Use numpy's efficient loadtxt with skiprows
        try:
            data_column = np.loadtxt(
                filepath, skiprows=GSLibConfig.HEADER_LINES, dtype=np.float64
            )
        except (ValueError, OSError) as e:
            raise OSError(f"Failed to read GSLIB file {filepath}: {e}") from e

        expected_size = int(np.prod(shape))
        if data_column.size != expected_size:
            raise ValueError(
                f"Array size mismatch in {filepath}: got {data_column.size} elements, "
                f"expected {expected_size} (shape {shape})"
            )

        reshaped = data_column.reshape(shape, order="F")

        # Debug logging with safe min/max computation
        try:
            min_val = float(np.min(reshaped))
            max_val = float(np.max(reshaped))
            self._logger.debug(
                f"Loaded {filepath}: shape={reshaped.shape}, dtype={reshaped.dtype}, "
                f"min={min_val:.4f}, max={max_val:.4f}"
            )
        except (TypeError, ValueError):
            # Fallback if min/max computation fails (NumPy compatibility edge case)
            self._logger.debug(
                f"Loaded {filepath}: shape={reshaped.shape}, dtype={reshaped.dtype}"
            )

        return reshaped


class FileLocator:
    """Locates data files using various search strategies.

    Encapsulates the logic for finding GSLIB .dat files in a directory
    using candidate filenames and pattern matching.
    """

    def __init__(self) -> None:
        """Initialize the FileLocator with logging."""
        self._logger = logging.getLogger(__name__)

    def _normalize_filename(self, filename: str) -> str:
        """Normalize a filename by removing spaces, dashes, and converting to lowercase.

        Parameters
        ----------
        filename : str
            The filename to normalize.

        Returns
        -------
        str
            The normalized filename.
        """
        return filename.lower().replace("_", "").replace("-", "").replace(" ", "")

    def _generate_candidate_filenames(self, folder_name: str) -> list[str]:
        """Generate candidate filenames for a given folder name.

        Produces various naming conventions including underscored,
        space-replaced, and special cases for wave velocities.

        Parameters
        ----------
        folder_name : str
            The folder name to generate candidates from.

        Returns
        -------
        list[str]
            Candidate filenames in priority order.
        """
        candidates = [
            f"{folder_name}.dat",
            f"{folder_name.replace(' ', '_')}.dat",
        ]
        # Add special case filenames for wave velocity patterns
        folder_lower = folder_name.lower()
        for pattern_key, special_filename in GSLibConfig.VELOCITY_PATTERNS.items():
            if folder_lower.startswith(pattern_key):
                candidates.insert(0, special_filename)
                break
        candidates.append("".join(folder_name.split()) + ".dat")
        return candidates

    def _log_file_fallback(self, candidates: list[str], full_path: str) -> None:
        """Log a warning when a fallback file is used instead of candidates.

        Parameters
        ----------
        candidates : list[str]
            Expected candidate filenames.
        full_path : str
            The fallback file path being used.
        """
        self._logger.warning(
            f"Expected one of {candidates} not found. Using data file: {full_path}"
        )

    def _search_files_by_pattern(
        self, dat_files: list[str], key: str, folder_name: str, dir_path: Path
    ) -> Optional[str]:
        """Search for a data file using multiple pattern matching strategies.

        Tries to match by key name first, then by normalized folder name.

        Parameters
        ----------
        dat_files : list[str]
            List of available .dat file names.
        key : str
            The property key to match (e.g., "vp", "vs").
        folder_name : str
            The folder name to use for normalization matching.
        dir_path : Path
            Path to the directory containing files.

        Returns
        -------
        Optional[str]
            Full path to matched file, or None if no match found.
        """
        # Search by key name match
        for f in dat_files:
            if key.lower() in f.lower():
                return str(dir_path / f)

        # Search by normalized folder name match
        folder_compact = self._normalize_filename(folder_name)
        for f in dat_files:
            clean_name = self._normalize_filename(f)
            if folder_compact in clean_name:
                return str(dir_path / f)

        return None

    def find(self, key: str, folder_name: str, dir_path: Path) -> str:
        """Find the data file for a given key and folder.

        Uses the following search strategy in order of priority:
        1. Try exact candidate filenames (folder_name.dat, folder_name_underscore.dat, etc.)
        2. Search by property key name match (e.g., "vp" in filename)
        3. Search by normalized folder name match
        4. Use first available .dat file as fallback

        Parameters
        ----------
        key : str
            The property key (e.g., "vp", "vs").
        folder_name : str
            The folder name for this property.
        dir_path : Path
            Path to the directory containing the data files.

        Returns
        -------
        str
            Full path to the found data file.

        Raises
        ------
        FileNotFoundError
            If the folder doesn't exist or no matching .dat files are found.
        """
        # Generate candidate filenames
        candidates = self._generate_candidate_filenames(folder_name)

        # Try candidates first
        for fn in candidates:
            candidate_path = dir_path / fn
            if candidate_path.exists():
                return str(candidate_path)

        # If not found, search in directory
        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"Data folder not found: {dir_path}. "
                "Please ensure you have downloaded the Stanford VI-E data."
            )

        dat_files = [f.name for f in dir_path.glob("*.dat")]
        if not dat_files:
            raise FileNotFoundError(
                f"No .dat files found in expected folder: {dir_path}. "
                "Please ensure you have downloaded the Stanford VI-E data."
            )

        # Try pattern-based search strategies
        matched_path = self._search_files_by_pattern(
            dat_files, key, folder_name, dir_path
        )
        if matched_path:
            self._log_file_fallback(candidates, matched_path)
            return matched_path

        # Fallback to first file found
        full_path = str(dir_path / dat_files[0])
        self._log_file_fallback(candidates, full_path)
        self._logger.warning(
            f"For key '{key}': searched for candidates {candidates}, found files {dat_files}, "
            f"using {dat_files[0]}"
        )
        return full_path


@dataclass
class DatasetManager:
    """Manager for loading and accessing Stanford VI-E dataset properties.

    Attributes
    ----------
    data_path : str
        Root path to the dataset directory.
    file_map : Dict[str, str]
        Mapping of property keys to their folder names.
    grid_spec : GridSpec
        Grid specification for the dataset.
    vp : Optional[NDArray[np.float64]]
        P-wave velocity data.
    vs : Optional[NDArray[np.float64]]
        S-wave velocity data.
    rho : Optional[NDArray[np.float64]]
        Density data.
    facies : Optional[NDArray[np.float64]]
        Facies data.
    full_stack : Optional[NDArray[np.float64]]
        Full stack seismic data.
    """

    data_path: str
    file_map: Dict[str, str]
    grid_spec: GridSpec

    vp: Optional[NDArray[np.float64]] = None
    vs: Optional[NDArray[np.float64]] = None
    rho: Optional[NDArray[np.float64]] = None
    facies: Optional[NDArray[np.float64]] = None
    full_stack: Optional[NDArray[np.float64]] = None

    _other: Dict[str, NDArray[np.float64]] = field(default_factory=dict, repr=False)
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
            If data_path is empty or file_map is empty.
        """
        if not self.data_path or not self.data_path.strip():
            raise ValueError("data_path cannot be empty")
        if not self.file_map:
            raise ValueError("file_map cannot be empty")

    @property
    def grid_shape(self) -> TupleType[int, ...]:
        """Get the grid shape from grid_spec.

        Returns
        -------
        Tuple[int, ...]
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

    def __enter__(self) -> "DatasetManager":
        """Context manager entry.

        Returns
        -------
        DatasetManager
            Returns self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit. Cleans up resources if needed.

        Parameters
        ----------
        exc_type : Optional[type[BaseException]]
            Exception type if an exception occurred.
        exc_val : Optional[BaseException]
            Exception value if an exception occurred.
        exc_tb : Optional[TracebackType]
            Exception traceback if an exception occurred.
        """
        # Currently, no cleanup is needed as arrays are stored in memory.
        # This provides a hook for future resource management.
        pass

    def _assign_property(self, key: str, arr: NDArray[np.float64]) -> None:
        """Assign a loaded array to the appropriate property or _other mapping.

        For known properties (vp, vs, rho, facies, full_stack), assigns to
        the corresponding attribute. Unknown properties go into _other.

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

    def get_property(self, key: str) -> Optional[NDArray[np.float64]]:
        """Get a loaded property by key.

        Provides uniform access to both known properties and custom properties
        stored in _other.

        Parameters
        ----------
        key : str
            The property key (e.g., "vp", "vs", or custom key).

        Returns
        -------
        Optional[NDArray[np.float64]]
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
            True if the property is loaded (not None), False otherwise.
        """
        prop = self.get_property(key)
        return prop is not None

    def load(self) -> None:
        """Locate and read .dat files, assigning arrays to attributes.

        For each known property the corresponding dat file is read and the
        resulting array is assigned to the canonical attribute (for example
        `self.vp` or `self.facies`). Unknown keys are stored in the
        `_other` mapping to preserve access to non-standard properties.

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
                self._logger.info(f"Loading {key} from {full_path}...")
                arr = self._reader.read(full_path, expected_shape)
                self._assign_property(key, arr)
                loaded_count += 1
            except (OSError, ValueError) as e:
                self._logger.error(f"Failed to load {key} from {full_path}: {e}")
                raise

        self._logger.info(
            f"Successfully loaded {loaded_count} properties. Grid shape: {expected_shape}"
        )

    def align_cache_array(
        self,
        arr: Optional[NDArray[np.float64]],
        *,
        try_reshape: bool = True,
    ) -> Optional[NDArray[np.float64]]:
        """Validate and align a cache array to this DatasetManager's grid.

        The function ensures the provided array matches the manager's
        ``grid_spec.shape``. If the incoming array has the same number of
        elements but a different shape, and ``try_reshape`` is True, the
        function will attempt to reshape the array to the expected shape.

        Reshaping tries Fortran-order first (to match GSLIB "F" ordering),
        then C-order as a fallback. If alignment is not possible the
        function returns ``None``.

        Parameters
        ----------
        arr : Optional[NDArray[np.float64]]
            The array to align, or None.
        try_reshape : bool, optional
            Whether to attempt reshaping if dimensions match but shape differs.
            Default is True.

        Returns
        -------
        Optional[NDArray[np.float64]]
            The aligned array as an ndarray of dtype float64, or ``None`` if
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
            # Try Fortran order first (matches GSLIB usage in this project)
            try:
                reshaped = data.reshape(expected, order="F")
                return reshaped.astype(np.float64)
            except (ValueError, RuntimeError) as e:
                self._logger.debug(
                    f"Fortran-order reshape failed: {e}. Trying C-order reshape..."
                )

            # Fallback to C-order reshape
            try:
                reshaped = data.reshape(expected, order="C")
                return reshaped.astype(np.float64)
            except (ValueError, RuntimeError) as e:
                self._logger.debug(f"C-order reshape also failed: {e}")

        # Could not align
        self._logger.debug(
            f"Cache array shape {data.shape} cannot be aligned to grid shape {expected}"
        )
        return None

    @classmethod
    def from_stanfordsix(
        cls, data_path: str, file_map: Dict[str, str], grid_spec: GridSpec
    ) -> "DatasetManager":
        """Create a DatasetManager for the Stanford-VI-E layout and load data.

        This is a factory method that creates a DatasetManager instance,
        loads all data files, and returns the populated manager. It's a
        convenience method that combines instantiation with data loading.

        Parameters
        ----------
        data_path : str
            Root path to the dataset directory.
        file_map : Dict[str, str]
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


# Thin facade to read individual GSLIB files using the existing GSLibReader
class GslibLoader:
    """Singleton factory for loading GSLIB files.

    This class provides a convenient interface to read individual GSLIB files
    and manages a singleton instance for module-level access.
    """

    _instance: Optional["GslibLoader"] = None
    _reader: Optional[GSLibReader] = None

    def __new__(cls) -> "GslibLoader":
        """Ensure singleton pattern for the default instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._reader = GSLibReader()
        return cls._instance

    def read(
        self, filepath: Union[str, Path], grid_spec: GridSpec
    ) -> NDArray[np.float64]:
        """Read a GSLIB file and return a 3D NumPy array.

        Parameters
        ----------
        filepath : Union[str, Path]
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
    def get_instance(cls) -> "GslibLoader":
        """Get the singleton instance of GslibLoader.

        Returns
        -------
        GslibLoader
            The singleton instance.
        """
        return cls()


__all__ = ["DatasetManager", "GslibLoader", "GSLibConfig", "GSLibReader", "FileLocator"]
