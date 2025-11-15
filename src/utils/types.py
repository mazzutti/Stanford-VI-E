"""Project-wide shared typing aliases.

Keep small, stable type aliases used across the codebase here so they can be
re-used without duplication. This module contains compact aliases for the
`ProcessManager` helper signatures used by `src/analysis/common.py` and
other modules that delegate to the process manager.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Callable
from pathlib import Path

from src.utils.constants import CACHE_DIR_DEFAULT

# Aliases for the ProcessManager helper signatures
ClearCacheType = Callable[[list[str] | None, Path | None, str], int]
OpenFileType = Callable[[str, str | None, str], bool]
SummarizeType = Callable[[str, list[str] | None, str], None]


@runtime_checkable
class ProcessManagerProtocol(Protocol):
    """Protocol describing the minimal ProcessManager interface used by
    analysis helpers.

    This allows duck-typing in tests and production while keeping typing
    strict and centralized.
    """

    def clear_cache(
        self,
        patterns: list[str] | None = None,
        cache_dir: Path | None = None,
        prefix: str = "",
    ) -> int: ...

    def open_file(
        self, filepath: str, description: str | None = None, prefix: str = ""
    ) -> bool: ...

    def summarize_cache_files(
        self,
        cache_dir: str = CACHE_DIR_DEFAULT,
        keys: list[str] | None = None,
        prefix: str = "",
    ) -> None: ...


__all__ = ["ClearCacheType", "OpenFileType", "SummarizeType", "ProcessManagerProtocol"]
