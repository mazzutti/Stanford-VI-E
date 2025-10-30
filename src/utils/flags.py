"""Small feature-flag helper used for staged rollouts.

toggle from CI or developer machines. Remove this module after the
This module provides a minimal, environment-driven feature-flag helper.

Usage:
    from src.utils.flags import use_flag

    if use_flag("MODEL_USE_FACADE", default=False):
        # use new facade path
        ...

Flags are intentionally simple (env-var driven) so they are easy to
toggle from CI or developer machines.
"""

from __future__ import annotations

import os
from typing import Any
from src.utils.facades import LazyObjectProxy


__all__ = ["use_flag", "get_flag_value"]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("1", "true", "yes", "on")


def use_flag(name: str, default: bool = False) -> bool:
    """Return whether the named feature flag is enabled.

    The flag is read from the environment variable `name`. If the
    environment variable is not set, `default` is returned.
    """
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return _coerce_bool(v)


def get_flag_value(name: str, default: str | None = None) -> str | None:
    """Return the raw flag value from the environment or the default."""
    return os.environ.get(name, default)


class Flags:
    def use_flag(self, name: str, default: bool = False) -> bool:
        return use_flag(name, default=default)

    def get_flag_value(self, name: str, default: str | None = None) -> str | None:
        return get_flag_value(name, default=default)


# Module-level lazy flags facade
flags = LazyObjectProxy(lambda: Flags())


def get_flags(instance: Flags | None = None) -> Flags:
    return instance if instance is not None else flags


try:
    __all__
except NameError:
    __all__ = ["use_flag", "get_flag_value"]

__all__.extend(["flags", "get_flags"])
