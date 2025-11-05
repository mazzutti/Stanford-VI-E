"""Process and regeneration helpers.

Helpers for running external commands, managing caches, and running
regeneration pipelines. Local imports are used to avoid circular
dependencies.
"""

from pathlib import Path
import logging
from typing import List, Optional
from src.processing._singleton import SingletonFactory

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


# Module-level factory for the process manager
_process_manager_factory: SingletonFactory[ProcessManager] = SingletonFactory(
    lambda: ProcessManager()
)


def get_process_manager(manager: ProcessManager | None = None) -> ProcessManager:
    """Return the provided ProcessManager or the module-level lazy singleton.

    This helper follows the repository convention of providing get_* accessors
    for module-level lazy singletons to simplify dependency injection in
    tests and client code.
    """
    return _process_manager_factory.get(manager)


__all__.extend(["ProcessManager", "get_process_manager"])


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
    Returns the number of removed files (int).
    """
    import os
    from pathlib import Path

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
                        logger.warning("%sError removing %s: %s", prefix, fn, e)
            except Exception:
                logger.warning("%sPattern %s evaluation failed", prefix, pat)
    else:
        # Use modern pruning API
        from src.io.pruning import Pruner, PruneStrategy

        try:
            strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
            pruner = Pruner(strategy)
            result = pruner.prune(p)
            removed = result.count_removed
        except Exception as e:
            logger.warning("%sCache pruning failed: %s", prefix, e)

    logger.info("%sRemoved %d files from %s", prefix, removed, str(p))
    return removed


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
    from pathlib import Path

    p = Path(cache_dir)
    if not p.exists():
        logger.info("%sCache directory not found: %s", prefix, cache_dir)
        return

    logger.info("%sCache summary (%s):", prefix, cache_dir)

    # List all .npz files in the cache directory
    npz_files = sorted(p.glob("*.npz"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not npz_files:
        logger.info("%s  (empty cache)", prefix)
        return

    # Group by key prefix
    groups = {}
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
            logger.info(
                "%s  %s: %s (%.1f MB)",
                prefix,
                k,
                latest.name,
                size_mb,
            )
        else:
            logger.info("%s  %s: <none>", prefix, k)
