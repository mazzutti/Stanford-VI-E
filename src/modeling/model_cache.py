"""Caching for AVO modeling results.

CacheManager handles persistence of expensive computations like AVO synthetics,
providing a simple key-value interface backed by NPZ files.
"""

from __future__ import annotations

from pathlib import Path
import os
import hashlib
import numpy as np
from typing import Callable
import logging

from src.io.cache import cache_for_dir
from src.modeling.modeling import SynthesisConfig, _unwrap_quantity
from src.utils.quantity import Quantity

# Configuration constants
CACHE_HASH_LENGTH: int = 20
"""Length of cache key hash in characters"""

__all__ = ["CacheManager"]

logger = logging.getLogger(__name__)


class _CacheHasher:
    """Internal hash computation for cache keys."""

    @staticmethod
    def hash_properties(
        arrays: list[np.ndarray],
        extras: list[str | float | int | bool | np.ndarray] | None = None,
    ) -> str:
        """Hash reflectivity and parameters for cache keys.

        Args:
            arrays: List of numpy arrays (e.g., [vp, vs, rho])
            extras: Additional parameters to include (angles, wavelet, etc.)

        Returns:
            20-character hex hash
        """
        h = hashlib.sha256()

        for a in arrays:
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())

        if extras:
            for e in extras:
                if isinstance(e, (list, tuple)):
                    h.update(str(list(e)).encode())
                elif isinstance(e, np.ndarray):
                    h.update(e.tobytes())
                else:
                    h.update(str(e).encode())

        return h.hexdigest()[:CACHE_HASH_LENGTH]


class CacheManager:
    """Manages caching of AVO and other modeling results.

    Provides high-level cache operations: saving/loading synthetics,
    handling cache keys via hashing, and respecting environment flags.
    """

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self._hasher = _CacheHasher()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def save_avo_synthetics(
        self,
        filename: str,
        full_stack: np.ndarray,
        angle_stacks: list[np.ndarray] | None = None,
    ) -> None:
        """Save AVO synthetics to cache.

        Args:
            filename: Output filename (relative to cache_dir)
            full_stack: Combined seismic stack
            angle_stacks: Optional list of angle-dependent stacks
        """
        save_dict: dict[str, np.ndarray] = {"full_stack": full_stack}

        if angle_stacks:
            for i, angle_stack in enumerate(angle_stacks):
                save_dict[f"angle_{i}"] = angle_stack

        filepath = Path(self.cache_dir) / filename
        cache_for_dir(self.cache_dir).save_npz(filepath, save_dict)
        logger.info("Saved AVO synthetics to cache: %s", filename)

    def load_avo_synthetics(
        self, filename: str
    ) -> tuple[list[np.ndarray] | None, np.ndarray]:
        """Load AVO synthetics from cache.

        Args:
            filename: Input filename (relative to cache_dir)

        Returns:
            (angle_stacks, full_stack): Angle-dependent stacks (if present) and combined stack
        """
        filepath = Path(self.cache_dir) / filename
        data = np.load(filepath)
        logger.info("Loaded AVO synthetics from cache: %s", filename)

        full_stack = data["full_stack"]
        angle_stacks = None

        if "angle_0" in data:
            angle_stacks = [data[f"angle_{i}"] for i in range(len(data) - 1)]

        return angle_stacks, full_stack

    def compute_cache_key(
        self,
        vp: np.ndarray,
        vs: np.ndarray,
        rho: np.ndarray,
        angles: list[float],
        wavelet: np.ndarray,
        use_quality_weighting: bool = False,
        add_noise: bool = False,
        snr_db: float = 20,
        noise_seed: int | None = None,
    ) -> str:
        """Compute cache key from parameters.

        Args:
            vp, vs, rho: Rock properties
            angles: Incidence angles
            wavelet: Source wavelet
            use_quality_weighting: Whether quality weighting is applied
            add_noise: Whether noise is added
            snr_db: Signal-to-noise ratio
            noise_seed: Random seed for noise

        Returns:
            Cache key string
        """
        extra_params: list[str | float | int | bool | np.ndarray] = [
            str(angles),
            wavelet,
            use_quality_weighting,
            add_noise,
            snr_db,
            noise_seed if noise_seed is not None else 0,
        ]
        return self._hasher.hash_properties([vp, vs, rho], extras=extra_params)

    def get_avo_synthetics(
        self,
        props_time: dict[str, np.ndarray | Quantity],
        angles: list[float],
        wavelet: np.ndarray,
        create_fn: Callable[
            [dict[str, np.ndarray], list[float], np.ndarray, SynthesisConfig | None],
            tuple[list[np.ndarray], np.ndarray],
        ],
        config: SynthesisConfig | None = None,
    ) -> tuple[list[np.ndarray] | None, np.ndarray]:
        """Get AVO synthetics, computing if not cached.

        Args:
            props_time: Rock properties dict with 'vp', 'vs', 'rho'
            angles: Incidence angles
            wavelet: Source wavelet
            create_fn: Callable to create synthetics if not cached
            config: SynthesisConfig with weighting, noise, and SNR settings

        Returns:
            (angle_stacks, full_stack)
        """
        config = config or SynthesisConfig()

        # Unwrap properties
        vp = _unwrap_quantity(props_time["vp"])
        vs = _unwrap_quantity(props_time["vs"])
        rho = _unwrap_quantity(props_time["rho"])

        # Compute cache key and check for force recompute
        key = self.compute_cache_key(
            vp,
            vs,
            rho,
            angles,
            wavelet,
            use_quality_weighting=config.use_quality_weighting,
            add_noise=config.add_noise,
            snr_db=config.snr_db,
            noise_seed=config.noise_seed,
        )
        force = os.environ.get("FORCE_RECOMPUTE", "0") == "1"
        filename = f"avo_time_{key}.npz"
        filepath = Path(self.cache_dir) / filename

        # Return cached if available
        if (not force) and filepath.exists():
            logger.info("Cache hit for AVO synthetics: %s", filename)
            return self.load_avo_synthetics(filename)

        logger.info("Cache miss for AVO synthetics; computing...")

        # Unwrap Quantity objects before passing to create_fn
        props_unwrapped: dict[str, np.ndarray] = {
            k: _unwrap_quantity(v) for k, v in props_time.items()
        }

        # Create and cache
        angle_stacks, full_stack = create_fn(
            props_unwrapped,
            angles,
            wavelet,
            config,
        )

        self.save_avo_synthetics(filename, full_stack, angle_stacks)
        return angle_stacks, full_stack
