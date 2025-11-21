"""Small shared helpers for CLI tools to avoid duplication.

This module contains tiny, safe helpers used by `tools.py` and
`tools_modeling.py` to keep both files small and reduce duplicate-code
findings from pylint.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent(path: Path) -> None:
    """Ensure the parent directory of `path` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, /, **arrays: Any) -> Path:
    """Save arrays to a compressed npz file, creating parent dirs.

    Returns the path saved.
    """
    ensure_parent(path)
    np.savez_compressed(path, **arrays)
    return path


def save_npz_with_timestamp(
    cache_dir: str | Path, prefix: str, /, **arrays: Any
) -> Path:
    """Save arrays into `cache_dir` with a timestamped filename.

    Returns the created file path.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    cache_file = cache_path / f"{prefix}_{ts}.npz"
    np.savez_compressed(cache_file, **arrays)
    return cache_file


def choose_html_path(
    plot_out: str | None, out: str | None, default_dir: Path, default_name: str
) -> Path:
    """Decide an output HTML `Path` given optional `plot_out` or `out`.

    - If `plot_out` is provided, treat it as the exact path.
    - Else if `out` is provided, use same filename but with `.html` suffix.
    - Else use `default_dir / default_name`.

    This function does not raise; it returns a Path and ensures the parent
    directory exists.
    """
    if plot_out:
        p = Path(plot_out)
    elif out:
        p = Path(out).with_suffix(".html")
    else:
        p = default_dir / default_name

    p.parent.mkdir(parents=True, exist_ok=True)
    return p
