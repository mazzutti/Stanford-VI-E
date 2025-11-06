"""Project-wide shared typing aliases.

Keep small, stable type aliases used across the codebase here so they can be
re-used without duplication. This module contains compact aliases for the
`ProcessManager` helper signatures used by `src/analysis/common.py` and
other modules that delegate to the process manager.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol, runtime_checkable
from pathlib import Path

from src.utils.constants import CACHE_DIR_DEFAULT

# Aliases for the ProcessManager helper signatures
ClearCacheType = Callable[[Optional[List[str]], Optional[Path], str], int]
OpenFileType = Callable[[str, Optional[str], str], bool]
SummarizeType = Callable[[str, Optional[List[str]], str], None]


@runtime_checkable
class ProcessManagerProtocol(Protocol):
    """Protocol describing the minimal ProcessManager interface used by
    analysis helpers.

    This allows duck-typing in tests and production while keeping typing
    strict and centralized.
    """

    def clear_cache(
        self,
        patterns: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> int: ...

    def open_file(
        self, filepath: str, description: Optional[str] = None, prefix: str = ""
    ) -> bool: ...

    def summarize_cache_files(
        self,
        cache_dir: str = CACHE_DIR_DEFAULT,
        keys: Optional[List[str]] = None,
        prefix: str = "",
    ) -> None: ...


__all__ = ["ClearCacheType", "OpenFileType", "SummarizeType", "ProcessManagerProtocol"]
