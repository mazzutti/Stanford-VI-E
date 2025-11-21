"""File locator for finding data files in directories.

This module handles locating GSLIB .dat files using various search strategies
and pattern matching.

Design:
- FileLocator: Encapsulates file search logic with multiple strategies
"""

import logging
from pathlib import Path

from src.io.gslib_reader import GSLibConfig

__all__ = ["FileLocator"]

logger = logging.getLogger(__name__)

# FileLocator is a small utility class with a compact public surface; silence
# too-few-public-methods to reduce stylistic noise for this helper.


class FileLocator:
    """Locates data files using various search strategies.

    Encapsulates the logic for finding GSLIB .dat files in a directory
    using candidate filenames and pattern matching.

    Attributes
    ----------
    _logger : logging.Logger
        Logger for debug/warning messages.
    """

    def __init__(self, logger_obj: logging.Logger | None = None) -> None:
        """Initialize the FileLocator with logging.

        Parameters
        ----------
        logger_obj : logging.Logger | None
            Optional logger instance.
        """
        self._logger = logger_obj or logging.getLogger(__name__)

    def _normalize_filename(self, filename: str) -> str:
        """Normalize a filename by removing spaces, dashes, converting to lowercase.

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
            "Expected one of %s not found. Using data file: %s", candidates, full_path
        )

    def _search_files_by_pattern(
        self, dat_files: list[str], key: str, folder_name: str, dir_path: Path
    ) -> str | None:
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
        str | None
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
        1. Try exact candidate filenames
        2. Search by property key name match
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
            "For key '%s': searched for candidates %s, found files %s, using %s",
            key,
            candidates,
            dat_files,
            dat_files[0],
        )
        return full_path
