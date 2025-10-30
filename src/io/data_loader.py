"""Data loading helpers moved into `src.io`.

This module contains functions to load GSLIB files and the Stanford VI-E
dataset used by the project.
"""

from pathlib import Path
import numpy as np
import logging

from dataclasses import dataclass, field
from typing import Dict, Optional, Mapping, Union
from src.io.grid import GridSpec

logger = logging.getLogger(__name__)


@dataclass
class DatasetManager:
    """Container for loading and accessing the Stanford-VI-E dataset.

    Attributes:
        data_path: root path where property folders live
        file_map: mapping property_key -> folder name
        grid_spec: GridSpec describing the expected shape and spacing for
            each property cube
        data: mapping property_key -> 3D numpy array (populated by load())
    """

    data_path: str
    # file_map maps property keys (e.g. 'porosity') to folder names
    file_map: Mapping[str, str]
    grid_spec: GridSpec
    data: Dict[str, np.ndarray] = field(default_factory=dict)

    def _read_gslib(self, filepath: Union[str, Path]) -> np.ndarray:
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
        """Populate self.data by locating and reading .dat files.

        This mirrors the original procedural implementation but scopes the
        loaded cubes under the `DatasetManager` instance.
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
            self.data[key] = self._read_gslib(full_path)

        logger.info(
            "All data loaded successfully. Grid shape: %s", self.grid_spec.shape
        )

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


# Backwards-compatible top-level helper for reading a single gslib file.
def load_gslib_file(filepath: Union[str, Path], grid_spec: GridSpec) -> np.ndarray:
    """Compatibility wrapper that reads a GSLIB file using a transient
    DatasetManager instance. Kept for API compatibility with existing callers.
    """
    # Delegate to the GslibLoader facade for a consistent OO API.
    return gslib_loader.read(filepath, grid_spec)


__all__ = ["load_gslib_file", "DatasetManager"]


# Thin facade to read individual GSLIB files using the existing DatasetManager
class GslibLoader:
    def read(self, filepath: Union[str, Path], grid_spec: GridSpec) -> np.ndarray:
        dm = DatasetManager(data_path=".", file_map={}, grid_spec=grid_spec)
        return dm._read_gslib(filepath)


from src.utils.facades import LazyObjectProxy


# Module-level lazy proxy for convenience (standardized)
gslib_loader = LazyObjectProxy(lambda: GslibLoader())

__all__.extend(["GslibLoader", "gslib_loader"])


def get_gslib_loader(config: dict | None = None):
    """Return the module-level `gslib_loader` proxy when `config` is None,
    otherwise return a new `GslibLoader` instance. This centralizes access
    patterns and matches other `get_*` helpers added during the refactor.
    """
    if config is None:
        return gslib_loader
    return GslibLoader()


__all__.append("get_gslib_loader")
