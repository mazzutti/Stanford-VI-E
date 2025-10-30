"""Process and regenerate helpers (migrated from src.utils.process)

This module contains helpers for running external commands, managing caches,
and running regeneration pipelines. It intentionally keeps local imports to
avoid circular dependencies.
"""

from pathlib import Path
import subprocess
import logging
from typing import List, Optional

__all__ = []

# Module-level logger to avoid repeated dynamic imports
logger = logging.getLogger(__name__)


class ProcessManager:
    """Object-oriented facade for process-related utilities.

    Provides the same API surface as the module-level helpers but scoped
    to an instance for easier testing and injection.
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
from src.utils.facades import LazyObjectProxy


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
    cache_dir = cache_dir or Path(".cache")
    if patterns is None:
        try:

            from src.io.cache import cache_for_dir

            removed_count, _ = cache_for_dir(str(cache_dir)).cleanup_old_cache(
                dry_run=False
            )
            return removed_count
        except Exception:
            # Fallback to naive behavior if the centralized cleanup isn't
            # available for some reason (e.g., during early bootstrap).
            patterns = ["*"]

    if not cache_dir.exists():
        cache_dir.mkdir()

    removed_count = 0
    for pattern in patterns:
        for filepath in cache_dir.glob(pattern):
            try:
                size = filepath.stat().st_size / (1024 * 1024)
                logger.info("%s  Removing: %s (%.1f MB)", prefix, filepath.name, size)
                filepath.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning("%s  Failed to remove %s: %s", prefix, filepath, e)

    # use module logger
    if removed_count > 0:
        logger.info("%sRemoved %d files from %s", prefix, removed_count, cache_dir)
    else:
        logger.info("%sNo matching cache files to remove in %s", prefix, cache_dir)
    return removed_count


def _impl_clear_cache(
    patterns: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    prefix: str = "",
) -> int:
    return clear_cache(patterns=patterns, cache_dir=cache_dir, prefix=prefix)


def open_file(
    filepath: str, description: Optional[str] = None, prefix: str = ""
) -> bool:
    path = Path(filepath)
    # use module logger
    if not path.exists():
        logger.warning("%sNot found: %s", prefix, filepath)
        return False
    if description:
        logger.info("%sOpening: %s", prefix, description)
    try:
        subprocess.run(["open", str(path)], check=True)
        return True
    except Exception as e:
        logger.error("%sFailed to open %s: %s", prefix, filepath, e)

        return False


def _impl_open_file(
    filepath: str, description: Optional[str] = None, prefix: str = ""
) -> bool:
    return open_file(filepath=filepath, description=description, prefix=prefix)


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
        keys = [
            "avo_depth",
            "ai_depth",
            "ei_depth",
            "ei_time",
            "rock_physics_attributes",
        ]
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
