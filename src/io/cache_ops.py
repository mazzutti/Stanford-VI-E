"""Background operations service for async caching.

This module extracts threading and background operation concerns from DiskCache,
providing a reusable service for async saves and periodic pruning.

This design follows the Separation of Concerns principle, making it easy to
test, mock, or disable background operations independently.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Callable, Dict, Optional

__all__ = ["BackgroundOpsService"]

logger = logging.getLogger(__name__)


class BackgroundOpsService:
    """Service for managing background async operations.

    Handles:
    - Async execution of long-running tasks via ThreadPoolExecutor
    - Periodic background pruning via background thread
    - Graceful shutdown with cleanup

    This service is optional - caches can be used without it for
    synchronous-only operation.

    Attributes
    ----------
    max_workers : int
        Maximum number of concurrent background tasks.
    enable_periodic_ops : bool
        Whether to enable periodic background operations.
    periodic_interval_seconds : int
        Interval between periodic operations.
    """

    def __init__(
        self,
        max_workers: int = 2,
        enable_periodic_ops: bool = False,
        periodic_interval_seconds: int = 300,
        logger_obj: Optional[logging.Logger] = None,
    ):
        """Initialize background operations service.

        Parameters
        ----------
        max_workers : int
            Maximum concurrent worker threads.
        enable_periodic_ops : bool
            Enable periodic background operations.
        periodic_interval_seconds : int
            Interval between periodic operations (seconds).
        logger_obj : Optional[logging.Logger]
            Logger instance.
        """
        self.max_workers = max_workers
        self.enable_periodic_ops = enable_periodic_ops
        self.periodic_interval_seconds = periodic_interval_seconds
        self.logger = logger_obj or logger

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

        self._periodic_event = threading.Event()
        self._periodic_thread: Optional[threading.Thread] = None
        self._periodic_fn: Optional[Callable[[], None]] = None

        if self.enable_periodic_ops:
            self._start_periodic_thread()

    def _start_periodic_thread(self) -> None:
        """Start the periodic operations thread."""
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop,
            name="bg-ops-periodic",
            daemon=True,
        )
        self._periodic_thread.start()

    def _periodic_loop(self) -> None:
        """Loop that calls periodic function at specified intervals."""
        interval = self.periodic_interval_seconds
        if interval <= 0 or not self._periodic_fn:
            return

        while not self._periodic_event.wait(interval):
            try:
                if self._periodic_fn:
                    self._periodic_fn()
            except Exception as e:
                self.logger.debug(f"Error in periodic operation: {e}")

    def submit_async(
        self, fn: Callable[..., Any], *args, key: Optional[str] = None, **kwargs
    ) -> Future:
        """Submit a task for async execution.

        Parameters
        ----------
        fn : Callable
            Function to execute.
        *args
            Positional arguments to pass to fn.
        key : Optional[str]
            Optional key to track this future (useful for preventing duplicates).
        **kwargs
            Keyword arguments to pass to fn.

        Returns
        -------
        Future
            Future object for tracking completion.
        """
        with self._lock:
            # Cancel previous future for this key if it exists
            if key and key in self._futures:
                try:
                    self._futures[key].cancel()
                except Exception:
                    pass

            def _wrapper():
                try:
                    return fn(*args, **kwargs)
                finally:
                    # Clean up future reference
                    with self._lock:
                        self._futures.pop(key, None) if key else None

            future = self._executor.submit(_wrapper)
            if key:
                self._futures[key] = future
            return future

    def set_periodic_fn(self, fn: Callable[[], None]) -> None:
        """Set the function to be called periodically.

        Parameters
        ----------
        fn : Callable
            Function to call (must be callable with no arguments).
        """
        self._periodic_fn = fn
        if not self._periodic_thread and self.enable_periodic_ops:
            self._start_periodic_thread()

    def pending_count(self) -> int:
        """Get count of pending async operations."""
        try:
            with self._lock:
                return len(self._futures)
        except Exception:
            return 0

    def pending_keys(self) -> list[str]:
        """Get list of pending operation keys."""
        try:
            with self._lock:
                return list(self._futures.keys())
        except Exception:
            return []

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending operations to complete.

        Parameters
        ----------
        timeout : Optional[float]
            Maximum time to wait in seconds. If None, wait indefinitely.

        Returns
        -------
        bool
            True if all operations completed, False if timeout.
        """
        start_time = time.time()
        with self._lock:
            futures = list(self._futures.values())

        for future in futures:
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    return False
            else:
                remaining = None

            try:
                future.result(timeout=remaining)
            except Exception as e:
                self.logger.debug(f"Error waiting for future: {e}")
                return False

        return True

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown background operations service.

        Parameters
        ----------
        wait : bool
            If True, wait for pending operations to complete.
        """
        # Stop periodic operations
        try:
            if self._periodic_thread:
                self._periodic_event.set()
                if wait:
                    self._periodic_thread.join(timeout=5)
                self._periodic_thread = None
        except Exception as e:
            self.logger.debug(f"Error shutting down periodic thread: {e}")

        # Wait for pending futures
        if wait:
            self.wait_all(timeout=30)

        # Shutdown executor
        self._executor.shutdown(wait=wait)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown on exit."""
        self.shutdown(wait=True)
        return False
