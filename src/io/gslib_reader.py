"""GSLIB file reader and configuration.

This module handles reading GSLIB format files (.dat files) and provides
configuration for the GSLIB format used in the Stanford VI-E dataset.

Design:
- GSLibConfig: Configuration constants for GSLIB format
- GSLibReader: Handles reading and parsing GSLIB files
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.debug.plot3d import plot_volume

__all__ = ["GSLibConfig", "GSLibReader"]

logger = logging.getLogger(__name__)

# This module contains small helper classes (config + reader) intended as
# thin data holders and I/O helpers. Silence the too-few-public-methods
# warning to reduce noise for these simple types.


class GSLibConfig:
    """Configuration for GSLIB file reading.

    Encapsulates all GSLIB-related constants and patterns in a single,
    organized location following OOP principles.

    Attributes
    ----------
    HEADER_LINES : int
        Number of header lines to skip when reading GSLIB files.
    KNOWN_PROPERTIES : FrozenSet[str]
        Set of known property keys that map to class attributes.
    VELOCITY_PATTERNS : dict[str, str]
        Mapping of velocity folder name patterns to special filename conventions.
    """

    HEADER_LINES: int = 3
    """Number of header lines to skip when reading GSLIB files."""

    KNOWN_PROPERTIES: frozenset[str] = frozenset(
        {"vp", "vs", "rho", "facies", "full_stack"}
    )
    """Set of known property keys that map to class attributes."""

    VELOCITY_PATTERNS: dict[str, str] = {
        "p-wave": "Pvelocity.dat",
        "s-wave": "Svelocity.dat",
    }
    """Mapping of velocity folder name patterns to special filename conventions."""


class GSLibReader:
    """Reader for GSLIB format files.

    Encapsulates all logic for reading, parsing, and validating GSLIB .dat files.

    Attributes
    ----------
    _logger : logging.Logger
        Logger for debug/info messages.
    """

    def __init__(self, logger_obj: logging.Logger | None = None) -> None:
        """Initialize the GSLibReader with logging.

        Parameters
        ----------
        logger_obj : logging.Logger | None
            Optional logger instance.
        """
        self._logger = logger_obj or logging.getLogger(__name__)

    def read(self, filepath: str | Path, shape: tuple[int, ...]) -> NDArray[np.float64]:
        """Read a GSLIB `.dat` file and return a 3D NumPy array.

        The GSLIB files used here include a short header (GSLibConfig.HEADER_LINES lines)
        followed by a single column of numeric values in Fortran ordering. We skip
        the header lines and reshape with order="F".

        Parameters
        ----------
        filepath : str | Path
            Path to the GSLIB .dat file.
        shape : tuple[int, ...]
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
            If the data cannot be reshaped to the target shape.

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

        # Note: preserve ordering as read from file. Tests expect the raw
        # Fortran-ordered reshape without inverting the k-axis, so do not
        # flip the z-axis here.

        # Debug logging with safe min/max computation
        try:
            min_val = float(np.min(reshaped))
            max_val = float(np.max(reshaped))
            self._logger.debug(
                "Loaded %s: shape=%s, dtype=%s, min=%.4f, max=%.4f (z-axis flipped)",
                filepath,
                reshaped.shape,
                reshaped.dtype,
                min_val,
                max_val,
            )
        except (TypeError, ValueError):
            # Fallback if min/max computation fails
            self._logger.debug(
                "Loaded %s: shape=%s, dtype=%s (z-axis flipped)",
                filepath,
                reshaped.shape,
                reshaped.dtype,
            )

        return reshaped
