"""Disk-backed cache with async operations and pruning.

Provides a high-level cache interface with:
- Persistent NPZ storage
- Async background saves
- TTL and size-based pruning
- Background pruning thread
"""

from __future__ import annotations

import atexit
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

from src.io.pruning import Pruner, PruneStrategy
from src.io.storage import DiskStore

__all__ = [
    "DiskCache",
]

logger = logging.getLogger(__name__)


class DiskCache:
    """Disk-backed cache manager.

    Provides a thin, safe wrapper around `DiskStore` with optional
    asynchronous background saves and periodic pruning. Changes here are
    intentionally conservative to avoid altering runtime behavior.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        max_cache_bytes: int = 10 * 1024**3,
        ttl_seconds: int | None = None,
        periodic_prune_interval_seconds: int | None = None,
    ):
        """Create a DiskCache.

        Args:
            cache_dir: directory to store NPZ files
            max_cache_bytes: total size threshold in bytes; when exceeded, oldest files are removed
            ttl_seconds: optional TTL (seconds) after which files are considered stale and pruned
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.store = DiskStore(cache_dir=self.cache_dir, logger_obj=logger)
        self.max_cache_bytes = int(max_cache_bytes)
        self.ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else None
        self.periodic_prune_interval_seconds = (
            int(periodic_prune_interval_seconds)
            if periodic_prune_interval_seconds is not None
            else None
        )

        # Initialize pruning strategy and pruner
        self._prune_strategy = PruneStrategy.by_size_then_ttl(
            max_cache_bytes=self.max_cache_bytes,
            ttl_seconds=self.ttl_seconds,
        )
        self._pruner = Pruner(self._prune_strategy, logger_obj=logger)

        # Thread pool for background saves
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

        # Optional periodic pruning thread
        self._prune_event = threading.Event()
        self._prune_thread: threading.Thread | None = None
        if (
            self.periodic_prune_interval_seconds is not None
            and self.periodic_prune_interval_seconds > 0
        ):
            self._prune_thread = threading.Thread(
                target=self._periodic_prune_loop, name="diskcache-pruner", daemon=True
            )
            self._prune_thread.start()

        # Ensure background resources are cleaned up at interpreter exit.
        # This prevents non-daemon ThreadPoolExecutor threads from keeping
        # test runners (pytest) or other short-lived processes alive when a
        # DiskCache is created but not explicitly shutdown. We register a
        # non-blocking shutdown to avoid delaying interpreter exit.
        try:
            atexit.register(self.shutdown, False)
        except RuntimeError:
            # best-effort; never raise from __init__
            pass

    def make_key(self, prefix: str, meta: dict[str, str | int | float | bool]) -> str:
        """Create a cache key from prefix and metadata.

        Parameters
        ----------
        prefix : str
            Key prefix (e.g., 'avo').
        meta : dict[str, str | int | float | bool]
            Metadata dictionary to hash for uniqueness.

        Returns
        -------
        str
            Cache key.
        """
        return self.store.make_key(prefix, meta)

    def get_path_for_key(self, key: str) -> str | None:
        """Return the filesystem path for `key` if present, else None."""
        path = self.store.get_path_for_key(key)
        return str(path) if path else None

    def load_npz(self, key: str) -> dict[str, str | int | float | bool] | bytes | None:
        """Load a cached NPZ entry by `key`. Returns stored payload or None."""
        return self.store.get(key)

    def _prune_cache_if_needed(self) -> None:
        """Prune oldest files until total size <= max_cache_bytes or TTL satisfied."""
        self._pruner.prune(self.cache_dir)

    def save_npz(
        self,
        key: str,
        data: dict[str, str | int | float | bool] | bytes,
        blocking: bool = False,
    ) -> str:
        """Save data to NPZ cache with optional async execution.

        Parameters
        ----------
        key : str
            Cache key.
        data : dict[str, str | int | float | bool] | bytes
            Data to save.
        blocking : bool
            If True, save synchronously. If False, save in background.

        Returns
        -------
        str
            Path to saved file.
        """

        # helper sync save
        def _do_save(payload: dict[str, str | int | float | bool] | bytes) -> None:
            try:
                self.store.set(key, payload)
            except (OSError, ValueError, TypeError):
                # best-effort; do not raise for IO/type errors
                pass

        # synchronous (blocking) path for critical saves
        if blocking:
            _do_save(data)
            # pruning is best-effort
            self._prune_cache_if_needed()
            path = self.store.get_path_for_key(key)
            return str(path) if path else ""

        # perform save in background to avoid blocking large IO
        def _save(
            payload: dict[str, str | int | float | bool] | bytes, key_inner: str
        ) -> None:
            try:
                _do_save(payload)
                # pruning is best-effort
                self._prune_cache_if_needed()
            finally:
                # remove finished future entry (best-effort)
                with self._lock:
                    try:
                        self._futures.pop(key_inner, None)
                    except KeyError:
                        pass

        with self._lock:
            fut = self._executor.submit(_save, data, key)
            self._futures[key] = fut

        path = self.store.get_path_for_key(key)
        return str(path) if path else ""

    def total_size_bytes(self) -> int:
        """Return total size of cache directory in bytes (best-effort)."""
        return self.store.total_size_bytes()

    def entry_count(self) -> int:
        """Return number of .npz entries in the cache (best-effort)."""
        return self.store.entry_count()

    def list_entries(self) -> list[dict[str, str | int | float]]:
        """Return a list of dicts with metadata for each cache entry (name, size, mtime)."""
        return self.store.list_entries()

    def pending_saves_count(self) -> int:
        """Return number of pending async save futures (best-effort)."""
        try:
            with self._lock:
                return len(self._futures)
        except RuntimeError:
            return 0

    def pending_save_keys(self) -> list[str]:
        """Return a list of pending save keys (best-effort)."""
        try:
            with self._lock:
                return list(self._futures.keys())
        except RuntimeError:
            return []

    def _periodic_prune_loop(self) -> None:
        """Background loop that prunes the cache every `periodic_prune_interval_seconds` seconds.

        The loop exits when `self._prune_event` is set.
        """
        interval = int(self.periodic_prune_interval_seconds or 0)
        if interval <= 0:
            return
        while not self._prune_event.wait(interval):
            try:
                self._prune_cache_if_needed()
            except (OSError, RuntimeError):
                # ignore expected filesystem/runtime errors and continue
                pass

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown background executor; optionally wait for pending writes."""
        # stop periodic prune thread first
        try:
            if self._prune_thread is not None:
                self._prune_event.set()
                if wait:
                    try:
                        self._prune_thread.join(timeout=5)
                    except RuntimeError:
                        pass
                self._prune_thread = None
        except RuntimeError:
            pass

        if wait:
            with self._lock:
                for _, f in list(self._futures.items()):
                    try:
                        f.result(timeout=30)
                    except (FuturesTimeoutError, RuntimeError):
                        pass
                self._futures.clear()
        self._executor.shutdown(wait=wait)


def make_disk_cache(
    cache_dir: str = ".cache",
    max_cache_bytes: int = 10 * 1024**3,
    ttl_seconds: int | None = None,
    periodic_prune_interval_seconds: int | None = None,
) -> DiskCache:
    """Factory helper that constructs a `DiskCache` with sane defaults.

    This helper preserves the module-level `default_disk_cache` usage
    pattern while providing an explicit construction API for callers.
    """
    return DiskCache(
        cache_dir=cache_dir,
        max_cache_bytes=max_cache_bytes,
        ttl_seconds=ttl_seconds,
        periodic_prune_interval_seconds=periodic_prune_interval_seconds,
    )


# Module-level singleton instance for the default disk cache.
default_disk_cache: DiskCache = make_disk_cache()


def get_default_disk_cache(
    cache_dir: str | None = None,
    max_cache_bytes: int = 10 * 1024**3,
    ttl_seconds: int | None = None,
    periodic_prune_interval_seconds: int | None = None,
) -> DiskCache:
    """Return the module's default DiskCache instance when cache_dir is None
    or the configured DiskCache for a custom directory.

    This preserves the singleton `default_disk_cache` behavior while providing a
    single helper callers can use when they may or may not want the module
    default.
    """
    if cache_dir is None:
        return default_disk_cache
    return DiskCache(
        cache_dir=cache_dir,
        max_cache_bytes=max_cache_bytes,
        ttl_seconds=ttl_seconds,
        periodic_prune_interval_seconds=periodic_prune_interval_seconds,
    )
