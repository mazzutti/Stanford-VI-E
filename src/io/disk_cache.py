"""Disk-backed cache helper for numpy-backed artifacts.

Implements a simple compressed-NPZ disk cache keyed by content hash (SHA1).
Provides get/save helpers and optional TTL/pruning behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import logging
from src.utils.facades import LazyObjectProxy

__all__ = ["DiskCache", "_hash_for_obj"]

logger = logging.getLogger(__name__)


def _hash_for_obj(obj: Any) -> str:
    """Create a SHA1 hex digest for JSON-serializable objects or raw bytes."""
    if isinstance(obj, (bytes, bytearray)):
        data = bytes(obj)
    else:
        try:
            data = json.dumps(obj, sort_keys=True, default=str).encode("utf8")
        except Exception:
            data = str(obj).encode("utf8")
    return hashlib.sha1(data).hexdigest()


class DiskCache:
    def __init__(
        self,
        cache_dir: str = ".cache",
        max_cache_bytes: int = 10 * 1024**3,
        ttl_seconds: Optional[int] = None,
        periodic_prune_interval_seconds: Optional[int] = None,
    ):
        """Create a DiskCache.

        Args:
            cache_dir: directory to store NPZ files
            max_cache_bytes: total size threshold in bytes; when exceeded, oldest files are removed
            ttl_seconds: optional TTL (seconds) after which files are considered stale and pruned
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from src.io.cache import cache_for_dir

        self.cm = cache_for_dir(str(self.cache_dir))
        self.max_cache_bytes = int(max_cache_bytes)
        self.ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else None
        self.periodic_prune_interval_seconds = (
            int(periodic_prune_interval_seconds)
            if periodic_prune_interval_seconds is not None
            else None
        )

        # Thread pool for background saves
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

        # Optional periodic pruning thread
        self._prune_event = threading.Event()
        self._prune_thread: Optional[threading.Thread] = None
        if (
            self.periodic_prune_interval_seconds is not None
            and self.periodic_prune_interval_seconds > 0
        ):
            self._prune_thread = threading.Thread(
                target=self._periodic_prune_loop, name="diskcache-pruner", daemon=True
            )
            self._prune_thread.start()

    def make_key(self, prefix: str, meta: Dict[str, Any]) -> str:
        h = _hash_for_obj(meta)
        return f"{prefix}_{h}"

    def get_path_for_key(self, key: str) -> Optional[str]:
        # look for files under cache_dir starting with key
        now = time.time()
        for p in sorted(self.cache_dir.iterdir()):
            if p.name.startswith(key):
                if self.ttl_seconds is not None:
                    # check TTL
                    try:
                        if now - p.stat().st_mtime > self.ttl_seconds:
                            # stale
                            continue
                    except Exception:
                        pass
                return str(p)
        return None

    def load_npz(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.get_path_for_key(key)
        if not path:
            return None
        try:
            with np.load(path, allow_pickle=True) as npz:
                return dict(npz)
        except Exception:
            return None

    def _prune_cache_if_needed(self) -> None:
        """Prune oldest files until total size <= max_cache_bytes or TTL satisfied."""
        try:
            files = list(self.cache_dir.glob("*.npz"))
            if not files:
                return
            total = sum(f.stat().st_size for f in files)
            # remove TTL-expired files first
            now = time.time()
            if self.ttl_seconds is not None:
                for f in files:
                    try:
                        if now - f.stat().st_mtime > self.ttl_seconds:
                            f.unlink()
                    except Exception:
                        pass
                files = list(self.cache_dir.glob("*.npz"))
                total = sum(f.stat().st_size for f in files)

            if total <= self.max_cache_bytes:
                return

            # sort by mtime (oldest first) and remove until under limit
            files_sorted = sorted(files, key=lambda p: p.stat().st_mtime)
            for f in files_sorted:
                try:
                    size = f.stat().st_size
                    f.unlink()
                    total -= size
                    if total <= self.max_cache_bytes:
                        break
                except Exception:
                    pass
        except Exception:
            # best-effort only
            pass

    def save_npz(self, key: str, data: Dict[str, Any], blocking: bool = False) -> str:
        # filename includes key and short hash for uniqueness and readability
        short = key.split("_")[-1][:20]
        fn = self.cache_dir / f"{key}_{short}.npz"

        # helper sync save
        def _do_save(path, payload):
            try:
                self.cm.save_npz(str(path), payload)
            except Exception:
                # best-effort; do not raise
                pass

        # synchronous (blocking) path for critical saves
        if blocking:
            _do_save(fn, data)
            # pruning is best-effort
            self._prune_cache_if_needed()
            return str(fn)

        # perform save in background to avoid blocking large IO
        def _save(path, payload, key_inner):
            try:
                _do_save(path, payload)
                # pruning is best-effort
                self._prune_cache_if_needed()
            finally:
                # remove finished future entry (best-effort)
                with self._lock:
                    try:
                        self._futures.pop(key_inner, None)
                    except Exception:
                        pass

        with self._lock:
            fut = self._executor.submit(_save, fn, data, key)
            self._futures[key] = fut

        return str(fn)

    def total_size_bytes(self) -> int:
        """Return total size of cache directory in bytes (best-effort)."""
        try:
            files = list(self.cache_dir.glob("*.npz"))
            return sum(f.stat().st_size for f in files)
        except Exception:
            return 0

    def entry_count(self) -> int:
        """Return number of .npz entries in the cache (best-effort)."""
        try:
            return len(list(self.cache_dir.glob("*.npz")))
        except Exception:
            return 0

    def list_entries(self):
        """Return a list of dicts with metadata for each cache entry (name, size, mtime)."""
        out = []
        try:
            for f in sorted(self.cache_dir.glob("*.npz")):
                try:
                    st = f.stat()
                    out.append(
                        {"name": f.name, "size": st.st_size, "mtime": st.st_mtime}
                    )
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def pending_saves_count(self) -> int:
        """Return number of pending async save futures (best-effort)."""
        try:
            with self._lock:
                return len(self._futures)
        except Exception:
            return 0

    def pending_save_keys(self):
        """Return a list of pending save keys (best-effort)."""
        try:
            with self._lock:
                return list(self._futures.keys())
        except Exception:
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
            except Exception:
                # ignore errors and continue
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
                    except Exception:
                        pass
                self._prune_thread = None
        except Exception:
            pass

        if wait:
            with self._lock:
                for k, f in list(self._futures.items()):
                    try:
                        f.result(timeout=30)
                    except Exception:
                        pass
                self._futures.clear()
        self._executor.shutdown(wait=wait)


def make_disk_cache(
    cache_dir: str = ".cache",
    max_cache_bytes: int = 10 * 1024**3,
    ttl_seconds: Optional[int] = None,
    periodic_prune_interval_seconds: Optional[int] = None,
) -> DiskCache:
    return DiskCache(
        cache_dir=cache_dir,
        max_cache_bytes=max_cache_bytes,
        ttl_seconds=ttl_seconds,
        periodic_prune_interval_seconds=periodic_prune_interval_seconds,
    )


# Module-level lazy proxy instance to preserve the symbol `default_disk_cache`.
default_disk_cache = LazyObjectProxy(lambda: make_disk_cache())


__all__.extend(["make_disk_cache", "default_disk_cache"])


def get_default_disk_cache(
    cache_dir: str | None = None,
    max_cache_bytes: int = 10 * 1024**3,
    ttl_seconds: Optional[int] = None,
    periodic_prune_interval_seconds: Optional[int] = None,
):
    """Return the module's default DiskCache instance when cache_dir is None
    or the configured DiskCache for a custom directory.

    This preserves the lazy `default_disk_cache` behavior while providing a
    single helper callers can use when they may or may not want the module
    default.
    """
    return _impl_get_default_disk_cache(
        cache_dir=cache_dir,
        max_cache_bytes=max_cache_bytes,
        ttl_seconds=ttl_seconds,
        periodic_prune_interval_seconds=periodic_prune_interval_seconds,
    )


def _impl_get_default_disk_cache(
    cache_dir: str | None = None,
    max_cache_bytes: int = 10 * 1024**3,
    ttl_seconds: Optional[int] = None,
    periodic_prune_interval_seconds: Optional[int] = None,
) -> DiskCache:
    if cache_dir is None:
        return default_disk_cache
    return DiskCache(
        cache_dir=cache_dir,
        max_cache_bytes=max_cache_bytes,
        ttl_seconds=ttl_seconds,
        periodic_prune_interval_seconds=periodic_prune_interval_seconds,
    )


__all__.append("get_default_disk_cache")
