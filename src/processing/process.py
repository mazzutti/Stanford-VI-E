"""Process and regeneration helpers.

Helpers for running external commands, managing caches, and running
regeneration pipelines. Local imports are used to avoid circular
dependencies.
"""

from pathlib import Path
import logging
from typing import List, Optional
from src.utils.facades import LazyObjectProxy

__all__ = []

# Module-level logger to avoid repeated dynamic imports
logger = logging.getLogger(__name__)


class ProcessManager:
    """Object-oriented facade for process-related utilities.

    Provides an API surface scoped to an instance for easier testing and
    injection.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def clear_cache(
        self,
        patterns: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int:
        return _impl_clear_cache(patterns=patterns, cache_dir=cache_dir, prefix=prefix)

    def open_file(
        self, filepath: str, description: Optional[str] = None, prefix: str = ""
    ) -> bool:
        return _impl_open_file(
            filepath=filepath, description=description, prefix=prefix
        )

    def summarize_cache_files(
        self,
        cache_dir: str = ".cache",
        keys: Optional[List[str]] = None,
        prefix: str = "",
    ):
        return _impl_summarize_cache_files(
            cache_dir=cache_dir, keys=keys, prefix=prefix
        )


# Module-level default instance for convenience (lazy)
# Module-level lazy proxy using shared LazyObjectProxy
process_manager = LazyObjectProxy(lambda: ProcessManager())


__all__.extend(["ProcessManager", "process_manager"])


def get_process_manager(manager: ProcessManager | None = None) -> "ProcessManager":
    """Return the provided ProcessManager or the module-level lazy singleton.

    This helper follows the repository convention of providing get_* accessors
    for module-level lazy singletons to simplify dependency injection in
    tests and client code.
    """
    return manager if manager is not None else process_manager


__all__.append("get_process_manager")


def clear_cache(
    patterns: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    prefix: str = "",
) -> int:
    # If caller didn't pass explicit patterns, prefer the centralized cleanup
    # heuristics implemented in `src.io.cleanup.cleanup_old_cache`. This keeps
    # the logic for what counts as "old cache" in one place. If patterns are
    # provided, keep the existing behavior (pattern-based deletions).
    # Delegate to canonical implementation kept under _impl_clear_cache.
    return _impl_clear_cache(patterns=patterns, cache_dir=cache_dir, prefix=prefix)


def _impl_clear_cache(
    patterns: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    prefix: str = "",
) -> int:
    """Canonical implementation for clearing cache files.

    - If `patterns` is provided, perform simple glob-based removals under
      `cache_dir` (or the default cache directory).
    - If no patterns are provided, delegate to the shared `CacheManager`
      cleanup/main entrypoint which implements the repository-wide cleanup
      heuristics.
    Returns the number of removed files (int).
    """
    try:
        from src.io.cache import cache_for_dir
    except Exception:
        logger.warning("%sCache utilities are unavailable", prefix)
        return 0

    # Normalize cache_dir to a string that cache_for_dir understands
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    cm = cache_for_dir(cache_dir_str)

    # If explicit patterns are supplied, perform a simple pattern-based cleanup
    if patterns:
        removed = 0
        import os
        from pathlib import Path

        target_dir = cache_dir_str or getattr(cm, "cache_dir", ".cache")
        p = Path(target_dir)
        if not p.exists():
            return 0

        for pat in patterns:
            try:
                for fn in p.glob(pat):
                    try:
                        os.remove(fn)
                        removed += 1
                    except Exception as e:
                        logger.warning("%sError removing %s: %s", prefix, fn, e)
            except Exception:
                logger.warning("%sPattern %s evaluation failed", prefix, pat)

        logger.info("%sRemoved %d files from %s", prefix, removed, str(p))
        return removed

    # Otherwise delegate to CacheManager.main which returns (removed_count, size_mb)
    try:
        removed, _size_mb = cm.main(dry_run=False, verbose=False)
        return int(removed)
    except Exception as e:
        logger.warning("%sCache cleanup failed: %s", prefix, e)
        return 0


def open_file(
    filepath: str, description: Optional[str] = None, prefix: str = ""
) -> bool:
    return _impl_open_file(filepath=filepath, description=description, prefix=prefix)


def _impl_open_file(
    filepath: str, description: Optional[str] = None, prefix: str = ""
) -> bool:
    """Open `filepath` in a platform-friendly way.

    Prefer a pure-Python approach (`webbrowser.open`) and fall back to
    platform shell openers (`open`, `xdg-open`) if necessary.
    Returns True if an attempt to open the file was made, False otherwise.
    """
    from pathlib import Path

    p = Path(filepath)
    if not p.exists():
        logger.error("%sMissing file: %s", prefix, filepath)
        return False

    # Try webbrowser which is cross-platform for file:// URLs
    try:
        import webbrowser

        webbrowser.open(f"file://{p.resolve()}")
        return True
    except Exception:
        pass

    # Fallback to platform-specific opener
    try:
        import shutil
        import subprocess

        if shutil.which("open"):
            subprocess.run(["open", str(p)], check=False)
            return True
        if shutil.which("xdg-open"):
            subprocess.run(["xdg-open", str(p)], check=False)
            return True
    except Exception:
        pass

    logger.warning("%sCould not open file: %s", prefix, filepath)
    return False


def summarize_cache_files(
    cache_dir: str = ".cache", keys: Optional[List[str]] = None, prefix: str = ""
):
    return _impl_summarize_cache_files(cache_dir=cache_dir, keys=keys, prefix=prefix)


def _impl_summarize_cache_files(
    cache_dir: str = ".cache", keys: Optional[List[str]] = None, prefix: str = ""
):
    try:
        pass
    except Exception:
        logger.warning("%sCache utilities not available", prefix)
        return

    from src.io.cache import cache_for_dir

    groups = cache_for_dir(cache_dir).select_latest_cache_entries()
    if keys is None:
        keys = ["avo_depth", "rock_physics_attributes"]
    logger.info("%sCache summary (%s):", prefix, cache_dir)
    for k in keys:
        candidates = groups.get(k + "_", groups.get(k, []))
        if candidates:
            entry = candidates[-1]
            try:
                size_mb = entry.size_bytes / (1024**2)
                cfg = entry.config or {}
                cfg_summary = f" cfg_keys={list(cfg.keys())}" if cfg else ""
                logger.info(
                    "%s  %s: %s (%.1f MB)%s",
                    prefix,
                    k,
                    entry.path.name,
                    size_mb,
                    cfg_summary,
                )
            except Exception:
                logger.info("%s  %s: %s", prefix, k, entry.path.name)
        else:
            logger.info("%s  %s: <none>", prefix, k)
