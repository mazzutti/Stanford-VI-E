"""Simplified cache file selection strategies.

Provides clean, maintainable approaches to finding cache files.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["CacheFileSelector"]

# Small selector helper with a compact public surface; silence the
# too-few-public-methods warning for this simple utility class.


class CacheFileSelector:
    """Select cache files with clear, simple logic."""

    FILE_PREFIX = "seismic_"
    EXTENSIONS = [".npz", ".npy"]

    def __init__(self, prefer_npz: bool = True) -> None:
        """Initialize selector.

        Args:
            prefer_npz: Prefer compressed NPZ over NPY format
        """
        self.prefer_npz = prefer_npz

    def select(
        self,
        cache_dir: Path,
        domain: str,
        allow_npy: bool = True,
        find_latest: bool = False,
    ) -> Path | None:
        """Select cache file for domain.

        Args:
            cache_dir: Directory containing cache files
            domain: Domain identifier (e.g., 'acoustic')
            allow_npy: Allow .npy files as fallback
            find_latest: If no exact match, find latest matching file

        Returns:
            Path to cache file or None
        """
        if not domain:
            raise ValueError("domain cannot be empty")

        if not cache_dir.exists():
            logger.warning("Cache directory does not exist: %s", cache_dir)
            return None

        # Try exact matches first
        exact_match = self._find_exact_match(cache_dir, domain, allow_npy)
        if exact_match:
            return exact_match

        # Try pattern matching if requested
        if find_latest:
            return self._find_latest_match(cache_dir, domain, allow_npy)

        return None

    def _find_exact_match(
        self, cache_dir: Path, domain: str, allow_npy: bool
    ) -> Path | None:
        """Find exact filename match."""
        # Try NPZ first (preferred format)
        npz_path = cache_dir / f"{self.FILE_PREFIX}{domain}.npz"
        if npz_path.exists():
            logger.debug("Found exact NPZ match: %s", npz_path)
            return npz_path

        # Try NPY if allowed
        if allow_npy:
            npy_path = cache_dir / f"{self.FILE_PREFIX}{domain}.npy"
            if npy_path.exists():
                logger.debug("Found exact NPY match: %s", npy_path)
                return npy_path

        return None

    def _find_latest_match(
        self, cache_dir: Path, domain: str, allow_npy: bool
    ) -> Path | None:
        """Find latest file matching pattern."""
        matches = self._find_all_matches(cache_dir, domain, allow_npy)

        if not matches:
            logger.debug("No pattern matches found for domain: %s", domain)
            return None

        # Return most recently modified file
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        logger.debug("Found latest match: %s", latest)
        return latest

    def _find_all_matches(
        self, cache_dir: Path, domain: str, allow_npy: bool
    ) -> list[Path]:
        """Find all files matching domain pattern."""
        patterns = [f"{self.FILE_PREFIX}*{domain}*.npz"]
        if allow_npy:
            patterns.append(f"{self.FILE_PREFIX}*{domain}*.npy")

        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(cache_dir.glob(pattern))

        # Filter to only existing files
        return [p for p in matches if p.exists()]
