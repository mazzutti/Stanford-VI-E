"""Cache entry data model and utilities.

This module provides the CacheEntry dataclass which represents a single
cache file and its metadata.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Represents a single cache file and basic metadata.

    Attributes:
        key: logical key prefix (e.g., 'avo')
        path: absolute Path to the cache file
        mtime: modification time (seconds since epoch)
        size_bytes: file size in bytes
        config_hash: optional config hash extracted from filename
        config: optional configuration data from NPZ file
        valid: whether file could be successfully read
    """

    key: str
    path: Path
    mtime: float
    size_bytes: int
    config_hash: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    valid: Optional[bool] = None

    @classmethod
    def _extract_config_hash(cls, filename: str) -> Optional[str]:
        """Extract config/hash from filename (20+ hex chars)."""
        m = re.search(r"([0-9a-f]{20,})", filename)
        return m.group(1) if m else None

    @classmethod
    def _load_npz_config(cls, path: Path) -> tuple[Optional[Dict[str, Any]], bool]:
        """Load configuration from NPZ file. Returns (config, valid)."""
        try:
            import numpy as _np

            with _np.load(path, allow_pickle=True) as npz:
                if "config" not in npz:
                    return None, True
                cfg = npz["config"]
                try:
                    return dict(cfg), True
                except Exception:
                    try:
                        return cfg.item(), True
                    except Exception:
                        return None, True
        except Exception:
            return None, False

    @classmethod
    def from_path(cls, p: Union[str, os.PathLike]) -> "CacheEntry":
        """Create CacheEntry with full inspection of file contents."""
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        stat = p.stat()
        config_hash = cls._extract_config_hash(p.name)
        config, valid = cls._load_npz_config(p)
        key = p.name.split("_")[0] if "_" in p.name else p.name

        return cls(
            key=key,
            path=p,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            config_hash=config_hash,
            config=config,
            valid=valid,
        )

    @classmethod
    def from_path_shallow(cls, p: Union[str, os.PathLike]) -> "CacheEntry":
        """Create CacheEntry without reading file contents (fast)."""
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        stat = p.stat()
        config_hash = cls._extract_config_hash(p.name)
        key = p.name.split("_")[0] if "_" in p.name else p.name

        return cls(
            key=key,
            path=p,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            config_hash=config_hash,
        )

    def __repr__(self) -> str:
        return (
            f"CacheEntry(key={self.key!r}, path={str(self.path)!r}, "
            f"mtime={self.mtime:.0f}, size_bytes={self.size_bytes}, valid={self.valid})"
        )

    def to_dict(self) -> Dict[str, Union[str, int, float, None]]:
        """Convert entry to dictionary for serialization."""
        return {
            "key": self.key,
            "path": str(self.path),
            "mtime": self.mtime,
            "size_bytes": self.size_bytes,
            "config_hash": self.config_hash,
            "config": self.config,
            "valid": self.valid,
        }


__all__ = ["CacheEntry"]
