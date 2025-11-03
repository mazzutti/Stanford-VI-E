"""Data loading helpers.

This module contains utilities to load GSLIB files and the Stanford VI-E
dataset used by the project.
"""

from pathlib import Path
import numpy as np
import logging

from dataclasses import dataclass, field
from typing import (
    Dict,
    Optional,
    Mapping,
    Union,
    Iterator,
    MutableMapping,
    Any,
    List,
    Tuple,
)
from collections.abc import (
    MutableMapping as _MutableMapping,
    KeysView,
    ItemsView,
    ValuesView,
)
from src.io.grid import GridSpec
from numpy.typing import NDArray
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)


@dataclass
class DatasetManager:
    data_path: str
    file_map: Dict[str, str]
    grid_spec: GridSpec

    vp: Optional[NDArray[np.float64]] = None
    vs: Optional[NDArray[np.float64]] = None
    rho: Optional[NDArray[np.float64]] = None
    facies: Optional[NDArray[np.float64]] = None
    full_stack: Optional[NDArray[np.float64]] = None

    _other: Dict[str, NDArray[np.float64]] = field(default_factory=dict, repr=False)

    def _read_gslib(self, filepath: Union[str, Path]) -> NDArray[np.float64]:
        """Read a GSLIB `.dat` file and return a 3D NumPy array.

        The GSLIB files used here include a short header (3 lines) followed by
        a single column of numeric values in Fortran ordering. We mirror the
        previous behaviour: skip three header lines and reshape with order="F".
        """
        with Path(filepath).open("r") as f:
            # Skip header lines
            f.readline()
            f.readline()
            f.readline()
            values = [float(line.strip()) for line in f if line.strip()]

        data_column = np.array(values)
        shape = self.grid_spec.shape
        return data_column.reshape(shape, order="F")

    def load(self) -> None:
        """Locate and read .dat files, assigning arrays to attributes.

        For each known property the corresponding dat file is read and the
        resulting array is assigned to the canonical attribute (for example
        `self.vp` or `self.facies`). Unknown keys are stored in the
        `_other` mapping to preserve access to non-standard properties.
        """
        for key, folder_name in self.file_map.items():
            dir_path = Path(self.data_path) / folder_name
            candidates = [f"{folder_name}.dat", f"{folder_name.replace(' ', '_')}.dat"]
            if folder_name.lower().startswith("p-wave"):
                candidates.insert(0, "Pvelocity.dat")
            if folder_name.lower().startswith("s-wave"):
                candidates.insert(0, "Svelocity.dat")

            candidates.append("".join(folder_name.split()) + ".dat")

            full_path: Optional[str] = None
            for fn in candidates:
                candidate_path = dir_path / fn
                if candidate_path.exists():
                    full_path = str(candidate_path)
                    break

            if full_path is None:
                if not dir_path.is_dir():
                    raise FileNotFoundError(
                        (
                            f"Data folder not found: {dir_path}. "
                            "Please ensure you have downloaded the Stanford VI-E data."
                        )
                    )

                dat_files = [f.name for f in dir_path.glob("*.dat")]
                if not dat_files:
                    raise FileNotFoundError(
                        (
                            f"No .dat files found in expected folder: {dir_path}. "
                            "Please ensure you have downloaded the Stanford VI-E data."
                        )
                    )

                candidate = None
                for f in dat_files:
                    if key.lower() in f.lower():
                        candidate = f
                        break

                if candidate is None:
                    folder_compact = "".join(folder_name.lower().split())
                    for f in dat_files:
                        clean_name = (
                            f.lower().replace("_", "").replace("-", "").replace(" ", "")
                        )
                        if folder_compact in clean_name:
                            candidate = f
                            break

                if candidate is None:
                    candidate = dat_files[0]

                full_path = str(dir_path / candidate)
                logger.warning(
                    "Warning: expected one of %s not found. Using data file: %s",
                    candidates,
                    full_path,
                )

            logger.info("Loading %s from %s...", key, full_path)
            # use the instance method to read the gslib-formatted file
            arr = self._read_gslib(full_path)

            # Assign to the explicit attribute when it's a known key; store
            # unknown keys in the `_other` mapping so they remain accessible
            # to callers that look up non-canonical keys.
            if key == "vp":
                self.vp = arr
            elif key == "vs":
                self.vs = arr
            elif key == "rho":
                self.rho = arr
            elif key == "facies":
                self.facies = arr
            elif key == "full_stack":
                self.full_stack = arr
            else:
                self._other[key] = arr

        logger.info(
            "All data loaded successfully. Grid shape: %s", self.grid_spec.shape
        )

    def align_cache_array(
        self,
        arr: "NDArray[np.float64]",
        *,
        try_reshape: bool = True,
    ) -> "NDArray[np.float64] | None":
        """Validate and align a cache array to this DatasetManager's grid.

        The function ensures the provided array matches the manager's
        ``grid_spec.shape``. If the incoming array has the same number of
        elements but a different shape, and ``try_reshape`` is True, the
        function will attempt to reshape the array to the expected shape.

        Reshaping tries Fortran-order first (to match GSLIB "F" ordering),
        then C-order as a fallback. If alignment is not possible the
        function returns ``None``.

        Returns
        -------
        ndarray or None
            The aligned array as an ndarray of dtype float64, or ``None`` if
            alignment failed.
        """
        if arr is None:
            return None

        data = np.asarray(arr)
        expected = tuple(self.grid_spec.shape)

        # Exact shape match
        if data.shape == expected:
            return data.astype(np.float64)

        # If same number of elements, try reshaping
        if try_reshape and data.size == int(self.grid_spec.voxel_count()):
            # Try Fortran order first (matches GSLIB usage in this project)
            try:
                reshaped = data.reshape(expected, order="F")
                return reshaped.astype(np.float64)
            except Exception:
                pass

            # Fallback to C-order reshape
            try:
                reshaped = data.reshape(expected, order="C")
                return reshaped.astype(np.float64)
            except Exception:
                pass

        # Could not align
        logger.debug(
            "Cache array shape %s cannot be aligned to grid shape %s",
            data.shape,
            expected,
        )
        return None

    @classmethod
    def from_stanfordsix(
        cls, data_path: str, file_map: Dict[str, str], grid_spec: GridSpec
    ) -> "DatasetManager":
        """Create a DatasetManager for the Stanford-VI-E layout.

        This requires a GridSpec instance.
        """
        dm = cls(data_path=data_path, file_map=file_map, grid_spec=grid_spec)
        dm.load()
        return dm


# Module API: prefer the OO facades and proxies. Use `gslib_loader.read(...)`
# or obtain an instance via `get_gslib_loader()`.


__all__ = ["DatasetManager"]


# Thin facade to read individual GSLIB files using the existing DatasetManager
class GslibLoader:
    def read(
        self, filepath: Union[str, Path], grid_spec: GridSpec
    ) -> NDArray[np.float64]:
        dm = DatasetManager(data_path=".", file_map={}, grid_spec=grid_spec)
        return dm._read_gslib(filepath)


# Module-level lazy proxy for convenience (standardized)
gslib_loader = LazyObjectProxy(lambda: GslibLoader())

__all__.extend(["GslibLoader", "gslib_loader", "get_gslib_loader"])


def get_gslib_loader(config: Dict[str, Any] | None = None) -> object:
    """Return the module-level `gslib_loader` proxy when `config` is None,
    otherwise return a new `GslibLoader` instance. This centralizes access
    patterns and matches other `get_*` helpers added during the refactor.

    The function returns the module-level proxy when ``config`` is None,
    otherwise a fresh ``GslibLoader`` instance is returned.
    """

    if config is None:
        return gslib_loader
    return GslibLoader()


def _impl_get_gslib_loader(config: Dict[str, Any] | None = None) -> object:
    if config is None:
        return gslib_loader
    return GslibLoader()
