```python
"""Facies extraction helpers for the `gen` package.

This module mirrors `seismogram_extractor` but works with facies cubes.
It provides the same API surface: class-based extractor, cache-aware
factory methods and a small default cache provider that can load or
generate facies depth caches.

Conventions are the same as the seismogram extractor: input arrays are
3D with shape (ni, nj, nk) and "top" refers to the first indices along
the last axis.
"""
from __future__ import annotations

from typing import Optional, Protocol, Any
import numpy as np


class FaciesTopLayersExtractor:
    """Extract top N layers from a 3D facies cube.

    API mirrors SeismogramTopLayersExtractor so callers can reuse logic.
    """

    def __init__(self, cube: np.ndarray):
        if not isinstance(cube, np.ndarray):
            raise TypeError("`cube` must be a NumPy ndarray")
        if cube.ndim != 3:
            raise ValueError("`cube` must be a 3D array with shape (ni, nj, nk)")
        self.cube = cube

    def extract_top_layers(self, n_layers: int = 2) -> np.ndarray:
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        nk = self.cube.shape[2]
        if n_layers > nk:
            raise ValueError("n_layers cannot be larger than number of samples (nk={})".format(nk))
        return self.cube[:, :, :n_layers].copy()

    def extract_top_layer(self, index: int = 0) -> np.ndarray:
        nk = self.cube.shape[2]
        if index < 0 or index >= nk:
            raise IndexError("layer index out of range")
        return self.cube[:, :, index].copy()

    def top_layers_mean(self, n_layers: int = 2) -> np.ndarray:
        layers = self.extract_top_layers(n_layers)
        return float(np.mean(layers, axis=2)).astype(np.float64) if layers.size == 0 else np.mean(layers, axis=2)

    def extract_top_two_geological_layers(self, thicknesses: tuple[int, int] = (80, 40)) -> np.ndarray:
        top_m, second_m = thicknesses
        total = int(top_m) + int(second_m)
        nk = self.cube.shape[2]
        if total < 1:
            raise ValueError("total thickness must be >= 1")
        if total > nk:
            raise ValueError(f"requested total layers {total} exceeds available samples nk={nk}")
        return self.cube[:, :, :total].copy()

    @classmethod
    def from_npz(cls, path: str, key: Optional[str] = None) -> "FaciesTopLayersExtractor":
        data = np.load(path, allow_pickle=True)
        if key is not None and key in data:
            arr = data[key]
        else:
            arr = None
            for v in data.values():
                if isinstance(v, np.ndarray):
                    arr = v
                    break
            if arr is None:
                raise ValueError(f"no ndarray found in npz: {path}")
        return cls(arr)

    @classmethod
    def from_cache_or_generate(
        cls,
        cache_provider: Optional["CacheProvider"] = None,
        cache_dir: str = ".cache",
        prefer_latest: bool = True,
        generate_if_missing: bool = True,
        force_generate: bool = False,
    ) -> "FaciesTopLayersExtractor":
        if cache_provider is None:
            cache_provider = DefaultCacheProvider(cache_dir=cache_dir)

        if prefer_latest:
            arr = cache_provider.load_latest_depth()
            if arr is not None:
                return cls(np.asarray(arr))

        if generate_if_missing:
            arr = cache_provider.generate_depth(force=force_generate)
            return cls(np.asarray(arr))

        raise FileNotFoundError("No facies cache found and generation not allowed")


class CacheProvider(Protocol):
    def load_latest_depth(self) -> Optional[np.ndarray]:
        ...

    def generate_depth(self, force: bool = False) -> np.ndarray:
        ...


class DefaultCacheProvider:
    """Default provider that looks for facies caches and can call a pipeline."""

    def __init__(self, cache_dir: str = ".cache", cache_manager: Any | None = None, pipeline: Any | None = None):
        self.cache_dir = cache_dir
        self._cache_manager = cache_manager
        self._pipeline = pipeline

    def _find_latest(self) -> Optional[str]:
        from pathlib import Path

        d = Path(self.cache_dir)
        if not d.exists():
            return None
        files = list(d.glob("facies_depth_*.npz")) + list(d.glob("facies_*.npz")) + list(d.glob("facies_top_layers_*.npz"))
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0]) if files else None

    def load_latest_depth(self) -> Optional[np.ndarray]:
        from pathlib import Path
        import numpy as _np

        path = self._find_latest()
        if path is None:
            return None
        p = Path(path)
        name = p.name
        try:
            data = _np.load(str(p), allow_pickle=True)
            # try to load known keys
            if "facies" in data:
                return data["facies"]
            if "top" in data:
                return data["top"]
            # fallback: first ndarray
            for v in data.values():
                if isinstance(v, _np.ndarray):
                    return v
        except Exception:
            pass

        # fallback: try cache manager if available
        try:
            from src.modeling.model_cache import CacheManager

            cm = self._cache_manager or CacheManager(cache_dir=str(p.parent))
            # try generic load (may not be available for facies)
            _, arr = cm.load_avo_synthetics(name)
            return arr
        except Exception:
            return None

    def generate_depth(self, force: bool = False) -> np.ndarray:
        # If not forced and cache exists, return it
        if not force:
            path = self._find_latest()
            if path:
                return self.load_latest_depth()

        # Attempt to call a pipeline if present
        try:
            from src.modeling.pipeline import ModelingPipeline

            mp = self._pipeline or ModelingPipeline()
            result = mp.run()
            # try to find facies in result
            if isinstance(result, dict):
                for k in ("facies", "facies_depth", "full_stack_depth"):
                    if k in result and result[k] is not None:
                        return result[k]
        except Exception:
            pass

        # fallback to cached file if pipeline didn't produce one
        path = self._find_latest()
        if path:
            return self.load_latest_depth()

        raise RuntimeError("Failed to generate or locate facies depth cache")


__all__ = ["FaciesTopLayersExtractor", "CacheProvider", "DefaultCacheProvider"]

```
