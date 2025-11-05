"""Configuration dataclasses for cache and I/O operations.

This module centralizes all configuration used by cache and data loading
utilities, reducing parameter passing and improving maintainability.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging


@dataclass
class CachePolicy:
    """Unified configuration for cache storage and pruning.

    Combines CacheConfig and PruneConfig into a single, cohesive configuration
    object that eliminates duplication and reduces parameter passing.

    Attributes
    ----------
    cache_dir : Path
        Directory to store cache files.
    max_cache_bytes : int
        Maximum total cache size in bytes before pruning.
    ttl_seconds : Optional[int]
        Time-to-live for cache entries in seconds. None means no TTL.
    glob_pattern : str
        Glob pattern for finding cache files (default: "*.npz").
    enable_background_pruning : bool
        Enable periodic background pruning.
    prune_interval_seconds : int
        Interval between background pruning operations (seconds).
    """

    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    max_cache_bytes: int = 10 * 1024**3  # 10 GB default
    ttl_seconds: Optional[int] = None
    glob_pattern: str = "*.npz"
    enable_background_pruning: bool = False
    prune_interval_seconds: int = 300  # 5 minutes default

    @classmethod
    def default(cls) -> "CachePolicy":
        """Create cache policy with default settings."""
        return cls()

    @classmethod
    def memory_only(cls) -> "CachePolicy":
        """Create policy for memory-only cache (no disk)."""
        return cls(cache_dir=Path(), max_cache_bytes=0)

    @classmethod
    def with_ttl(
        cls, ttl_seconds: int, cache_dir: Optional[Path] = None
    ) -> "CachePolicy":
        """Create policy with TTL expiration."""
        return cls(
            cache_dir=cache_dir or Path(".cache"),
            ttl_seconds=ttl_seconds,
        )

    @classmethod
    def by_size_only(cls, max_cache_bytes: int) -> "CachePolicy":
        """Create policy for size-only pruning."""
        return cls(max_cache_bytes=max_cache_bytes, ttl_seconds=None)

    @classmethod
    def by_ttl_only(cls, ttl_seconds: int) -> "CachePolicy":
        """Create policy for TTL-only pruning."""
        return cls(max_cache_bytes=int(1e18), ttl_seconds=ttl_seconds)

    @classmethod
    def by_size_then_ttl(
        cls,
        max_cache_bytes: int,
        ttl_seconds: Optional[int] = None,
    ) -> "CachePolicy":
        """Create policy for combined size and TTL pruning."""
        return cls(
            max_cache_bytes=max_cache_bytes,
            ttl_seconds=ttl_seconds,
        )

    @classmethod
    def with_background_pruning(
        cls,
        max_cache_bytes: int,
        ttl_seconds: Optional[int] = None,
        interval_seconds: int = 300,
    ) -> "CachePolicy":
        """Create policy with background pruning enabled."""
        return cls(
            max_cache_bytes=max_cache_bytes,
            ttl_seconds=ttl_seconds,
            enable_background_pruning=True,
            prune_interval_seconds=interval_seconds,
        )

    def __post_init__(self) -> None:
        """Ensure cache_dir is a Path object."""
        if not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class DataLoaderConfig:
    """Configuration for data loading operations.

    Attributes
    ----------
    data_path : str
        Root path to dataset files.
    logger : logging.Logger
        Logger for debug messages.
    """

    data_path: str
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self) -> None:
        """Validate data_path."""
        if not self.data_path or not self.data_path.strip():
            raise ValueError("data_path cannot be empty")


__all__ = [
    "CachePolicy",
    "DataLoaderConfig",
]
