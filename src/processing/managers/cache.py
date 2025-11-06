"""Cache management utilities."""


from pathlib import Path
from typing import List, Optional


from src.processing.managers.base import BaseManager


__all__ = ["CacheManager"]


class CacheManager(BaseManager):
    """Manages cache directory operations: clearing and summarizing cache files."""

    def clear(
        self,
        patterns: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        """Clear cache files matching patterns or using size-based pruning.

        Args:
            patterns: Optional list of glob patterns to match
            cache_dir: Cache directory path (defaults to ".cache")
            prefix: Prefix for log messages

        Returns:
            Number of removed files
        """
        import os

        target_dir = str(cache_dir) if cache_dir is not None else ".cache"
        p = Path(target_dir)
        if not p.exists():
            return 0

        removed = 0
        if patterns:
            for pat in patterns:
                try:
                    for fn in p.glob(pat):
                        try:
                            os.remove(fn)
                            removed += 1
                        except Exception as e:
                            self._log_warning("%sError removing %s: %s", prefix, fn, e)
                except Exception:
                    self._log_warning("%sPattern %s evaluation failed", prefix, pat)
        else:
            # Use modern pruning API for size-based cleanup
            from src.io.pruning import Pruner, PruneStrategy

            try:
                strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
                pruner = Pruner(strategy)
                result = pruner.prune(p)
                removed = result.count
            except Exception as e:
                self._log_warning("%sCache pruning failed: %s", prefix, e)

        self._log_info("%sRemoved %d files from %s", prefix, removed, str(p))
        return removed

    def summarize(
        self,
        cache_dir: str = ".cache",
        keys: Optional[List[str]] = None,
        prefix: str = "",
    ) -> None:
        """Print a summary of cache files in the directory.

        Args:
            cache_dir: Cache directory path
            keys: Optional list of key prefixes to filter by
            prefix: Prefix for log messages
        """
        p = Path(cache_dir)
        if not p.exists():
            self._log_info("%sCache directory not found: %s", prefix, cache_dir)
            return

        self._log_info("%sCache summary (%s):", prefix, cache_dir)

        # List all .npz files in the cache directory
        npz_files = sorted(
            p.glob("*.npz"), key=lambda x: x.stat().st_mtime, reverse=True
        )

        if not npz_files:
            self._log_info("%s  (empty cache)", prefix)
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
                self._log_info(
                    "%s  %s: %s (%.1f MB)",
                    prefix,
                    k,
                    latest.name,
                    size_mb,
                )
            else:
                self._log_info("%s  %s: <none>", prefix, k)
