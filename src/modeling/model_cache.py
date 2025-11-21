"""Caching for AVO modeling results.

CacheManager handles persistence of expensive computations like AVO synthetics,
providing a simple key-value interface backed by NPZ files.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.modeling.modeling import SynthesisConfig, unwrap_quantity
from src.utils.quantity import Quantity

# Configuration constants
CACHE_HASH_LENGTH: int = 20
"""Length of cache key hash in characters"""

__all__ = ["CacheManager"]

logger = logging.getLogger(__name__)


class _CacheHasher:
    """Internal hash computation for cache keys.

    This internal helper is small by design; the pylint warning about
    "too-few-public-methods" is intentional and safe to ignore here.
    """

    @staticmethod
    def hash_properties(
        arrays: list[NDArray[np.floating[Any]]],
        extras: (
            list[str | float | int | bool | NDArray[np.floating[Any]]] | None
        ) = None,
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
        full_stack: NDArray[np.floating[Any]],
        angle_stacks: list[NDArray[np.floating[Any]]] | None = None,
    ) -> None:
        """Save AVO synthetics to cache.

        Args:
            filename: Output filename (relative to cache_dir)
            full_stack: Combined seismic stack
            angle_stacks: Optional list of angle-dependent stacks
        """
        save_dict: dict[str, NDArray[np.floating[Any]]] = {"full_stack": full_stack}

        if angle_stacks:
            for i, angle_stack in enumerate(angle_stacks):
                save_dict[f"angle_{i}"] = angle_stack

        filepath = Path(self.cache_dir) / filename
        # Cast dict to satisfy type checkers that may have strict numpy stubs
        np.savez_compressed(file=str(filepath), **cast(dict[str, Any], save_dict))
        logger.info("Saved AVO synthetics to cache: %s", filename)

    def load_avo_synthetics(
        self, filename: str
    ) -> tuple[list[NDArray[np.floating[Any]]] | None, NDArray[np.floating[Any]]]:
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
        vp: NDArray[np.floating[Any]],
        vs: NDArray[np.floating[Any]],
        rho: NDArray[np.floating[Any]],
        angles: list[float],
        wavelet: NDArray[np.floating[Any]],
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
        # High-arity signature is intentional: cache keys depend on many params.

        extra_params: list[str | float | int | bool | NDArray[np.floating[Any]]] = [
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
        props_time: dict[str, NDArray[np.floating[Any]] | Quantity],
        angles: list[float],
        wavelet: NDArray[np.floating[Any]],
        create_fn: Callable[
            [
                dict[str, NDArray[np.floating[Any]]],
                list[float],
                NDArray[np.floating[Any]],
                SynthesisConfig | None,
            ],
            tuple[list[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        ],
        config: SynthesisConfig | None = None,
    ) -> tuple[list[NDArray[np.floating[Any]]] | None, NDArray[np.floating[Any]]]:
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
        # This facade coordinates multiple responsibilities; suppress
        # high-arity/local-variable warnings as they are noise for this
        # high-level orchestration function.

        config = config or SynthesisConfig()

        # Unwrap properties
        vp = unwrap_quantity(props_time["vp"])
        vs = unwrap_quantity(props_time["vs"])
        rho = unwrap_quantity(props_time["rho"])

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
        props_unwrapped: dict[str, NDArray[np.floating[Any]]] = {
            k: unwrap_quantity(v) for k, v in props_time.items()
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
