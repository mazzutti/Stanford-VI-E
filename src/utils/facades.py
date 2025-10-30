"""Small utilities for creating lazy facade proxies.

This module provides a minimal LazyObjectProxy that can be reused across the
codebase to reduce boilerplate when exposing thin class facades with
thread-safe lazy instantiation.
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar, Generic

T = TypeVar("T")


class LazyObjectProxy(Generic[T]):
    """Thread-safe lazy proxy for creating a single instance of T.

    Usage:
        proxy = LazyObjectProxy(lambda: MyFacade())
        proxy.do_thing()  # creates the MyFacade instance on first access
    """

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._instance: T | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> T:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str):
        inst = self._ensure()
        return getattr(inst, name)

    def __setattr__(self, name: str, value):
        # Internal attributes (starting with an underscore) belong to the
        # proxy instance itself. Forward other attribute sets to the
        # underlying object, creating it if necessary.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        inst = self._ensure()
        setattr(inst, name, value)

    def __repr__(self) -> str:
        if self._instance is None:
            return f"<{self.__class__.__name__} (uninitialized)>"
        return repr(self._instance)
