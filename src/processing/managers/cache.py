"""Cache management utilities."""

import logging
import os
from pathlib import Path

from src.processing.managers.resource_manager import ResourceManager

__all__ = ["CacheManager", "CacheClearStrategy", "CacheSummarizeStrategy"]

# Small strategy/helper classes in this module intentionally expose a
# compact public surface. Silence too-few-public-methods to reduce noise
# for these simple DSL-like helper types.

class CacheClearStrategy:
    """Strategy for clearing cache files."""

    def clear(self, resource_dir: Path, patterns: list[str] | None = None) -> int:
        """Clear cache files matching patterns or using size-based pruning."""
        if not resource_dir.exists():
            return 0
        return (
            self._clear_by_pattern(resource_dir, patterns)
            if patterns
            else self._clear_by_size(resource_dir)
        )

    def _clear_by_pattern(self, resource_dir: Path, patterns: list[str]) -> int:
        """Remove files matching glob patterns."""
        removed = 0
        for pattern in patterns:
            removed += self._remove_matching_files(resource_dir, pattern)
        return removed

    def _remove_matching_files(self, resource_dir: Path, pattern: str) -> int:
        """Remove all files matching pattern."""
        removed = 0
        try:
            for file_path in resource_dir.glob(pattern):
                if self._safe_remove(file_path):
                    removed += 1
        except OSError:
            pass
        return removed

    @staticmethod
    def _safe_remove(file_path: Path) -> bool:
        """Safely remove a file, returning True if successful."""
        try:
            os.remove(file_path)
            return True
        except OSError:
            return False

    def _clear_by_size(self, resource_dir: Path) -> int:
        """Clear cache using size-based pruning strategy."""
        try:
            from src.io.pruning import (
                Pruner,
                PruneStrategy,
            )

            strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
            result = Pruner(strategy).prune(resource_dir)
            return result.count
        except (ImportError, RuntimeError, OSError):
            return 0

class CacheSummarizeStrategy:
    """Strategy for summarizing cache files."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize summarize strategy with an optional logger."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def summarize(self, resource_dir: Path, keys: list[str] | None = None) -> None:
        """Print a summary of cache files in the directory."""
        if not resource_dir.exists():
            self.logger.info("Cache directory not found: %s", resource_dir)
            return

        self.logger.info("Cache summary (%s):", resource_dir)

        # List all .npz files in the cache directory
        npz_files = sorted(
            resource_dir.glob("*.npz"), key=lambda x: x.stat().st_mtime, reverse=True
        )

        if not npz_files:
            self.logger.info("  (empty cache)")
            return

        # Group by key prefix
        groups: dict[str, list[Path]] = {}
        for f in npz_files:
            key = f.name.split("_")[0]
            if key not in groups:
                groups[key] = []
            groups[key].append(f)

        # Show summary for requested keys
        if keys is None:
            keys = ["avo", "rock_physics"]

        for k in keys:
            candidates = groups.get(k, [])
            if candidates:
                latest = candidates[0]
                size_mb = latest.stat().st_size / (1024**2)
                self.logger.info("  %s: %s (%.1f MB)", k, latest.name, size_mb)
            else:
                self.logger.info("  %s: <none>", k)

class CacheManager(ResourceManager[Path]):
    """Manages cache directory operations: clearing and summarizing cache files."""

    def __init__(
        self, cache_dir: Path | None = None, logger: logging.Logger | None = None
    ) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Cache directory (defaults to ".cache")
            logger: Optional logger instance
        """
        if cache_dir is None:
            cache_dir = Path(".cache")

        super().__init__(
            resource_dir=cache_dir,
            clear_strategy=CacheClearStrategy(),
            summarize_strategy=CacheSummarizeStrategy(logger),
            logger=logger,
        )
