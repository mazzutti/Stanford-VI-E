"""Seismogram extraction helpers for the `gen` package.

Provides a small utility class to extract the top N layers (time/depth
samples) from a 3D seismogram cube.

Conventions
- Input cube shape: (ni, nj, nk) where nk is the time/depth axis.
- "Top" layers are the first indices along the last axis (index 0, 1, ...).

This file is intentionally small and dependency-free except for NumPy.
"""

from __future__ import annotations

from typing import Optional, Protocol, Any, cast
import numpy as np
from numpy.typing import NDArray
from src.utils.quantity import Quantity


class SeismogramTopLayersExtractor:
    """Extract top N layers from a 3D seismogram cube.

    Example:
        extractor = SeismogramTopLayersExtractor(cube)
        top2 = extractor.extract_top_layers(2)  # shape (ni, nj, 2)
        top0 = extractor.extract_top_layer(0)   # shape (ni, nj)
    """

    def __init__(self, cube: NDArray[Any]):
        if cube.ndim != 3:
            raise ValueError("`cube` must be a 3D array with shape (ni, nj, nk)")
        self.cube = cube

    def extract_top_layers(self, n_layers: int = 2) -> NDArray[Any]:
        """Return the top `n_layers` along the last axis.

        Returns an array with shape (ni, nj, n_layers).
        """
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        nk = self.cube.shape[2]
        if n_layers > nk:
            raise ValueError(
                "n_layers cannot be larger than number of samples (nk={})".format(nk)
            )
        return self.cube[:, :, :n_layers].copy()

    def extract_top_layer(self, index: int = 0) -> NDArray[Any]:
        """Return a single top layer (2D array) at the requested index.

        `index` is 0-based (0 is the very top sample).
        """
        nk = self.cube.shape[2]
        if index < 0 or index >= nk:
            raise IndexError("layer index out of range")
        return self.cube[:, :, index].copy()

    def top_layers_mean(self, n_layers: int = 2) -> NDArray[Any]:
        """Return the mean across the top `n_layers` for each (i, j).

        Result shape: (ni, nj)
        """
        layers = self.extract_top_layers(n_layers)
        if layers.size == 0:
            ni, nj = self.cube.shape[0], self.cube.shape[1]
            return np.zeros((ni, nj), dtype=np.float64)
        return cast(NDArray[Any], np.mean(layers, axis=2))

    def extract_top_two_geological_layers(
        self, thicknesses: tuple[int, int] = (80, 40)
    ) -> NDArray[Any]:
        """Extract the full thickness of the top two geological layers.

        The Stanford VI model uses layers with thicknesses 80 m (top) and 40 m
        (second). Because the vertical cell size is 1 m, these correspond to
        80 and 40 samples respectively. This convenience method returns the
        top (80+40) samples along the depth axis.

        Args:
            thicknesses: tuple with (top_layer_m, second_layer_m). Defaults to (80, 40).

        Returns:
            ndarray with shape (ni, nj, top_layer_samples + second_layer_samples)
        """
        top_m, second_m = thicknesses
        total = int(top_m) + int(second_m)
        nk = self.cube.shape[2]
        if total < 1:
            raise ValueError("total thickness must be >= 1")
        if total > nk:
            raise ValueError(
                f"requested total layers {total} exceeds available samples nk={nk}"
            )
        return self.cube[:, :, :total].copy()

    @classmethod
    def from_time_domain(
        cls,
        seismogram_time: NDArray[Any],
        vp_depth: NDArray[Any],
        grid_spec: object,
        plan: Optional[object] = None,
    ) -> "SeismogramTopLayersExtractor":
        """Construct an extractor from a time-domain seismogram by converting
        it back to the depth domain using the project's resampler utilities.

        Args:
            seismogram_time: array shaped (ni, nj, nt)
            vp_depth: velocity cube shaped (ni, nj, nz) in depth domain (m/s)
            grid_spec: `GridSpec` instance describing dz/dt/shape used by resampler
            plan: optional ResamplePlan to avoid recomputing TWT

        Returns:
            SeismogramTopLayersExtractor instance wrapping the depth-domain cube
        """
        # Import lazily to avoid heavy imports at module import time
        from src.processing.resampling._resampler import resampler_factory

        # resampler_factory expects a GridSpec; cast here to satisfy static
        # analysis while preserving runtime behavior.
        resampler = resampler_factory.get_resampler(cast(Any, grid_spec))
        seis_depth = resampler.time_to_depth_cube(
            seismogram_time, vp_depth, plan=cast(Any, plan)
        )
        # If resampler returned a Quantity-like object, extract ndarray
        maybe_arr = seis_depth.array if isinstance(seis_depth, Quantity) else seis_depth
        # Ensure we always pass a plain numpy ndarray to the constructor and help static
        # type checkers by casting.
        arr = np.asarray(maybe_arr)
        return cls(arr)

    @classmethod
    def from_cache_or_generate(
        cls,
        cache_provider: Optional["CacheProvider"] = None,
        cache_dir: str = ".cache",
        prefer_latest: bool = True,
        generate_if_missing: bool = True,
        force_generate: bool = False,
    ) -> "SeismogramTopLayersExtractor":
        """Load the newest depth-domain AVO cache if present, otherwise generate it.

        Args:
            cache_dir: directory where AVO caches are stored
            prefer_latest: if True, use the newest matching cache file found
            generate_if_missing: if True, run `ModelingPipeline.run()` to create caches when missing

        Returns:
            SeismogramTopLayersExtractor instance wrapping the depth-domain full stack
        """
        # Create default provider if not supplied
        if cache_provider is None:
            cache_provider = DefaultCacheProvider(cache_dir=cache_dir)

        # Try to load latest cache via provider
        if prefer_latest:
            arr = cache_provider.load_latest_depth()
            if arr is not None:
                return cls(np.asarray(arr))

        # If missing, optionally generate
        if generate_if_missing:
            arr = cache_provider.generate_depth(force=force_generate)
            return cls(np.asarray(arr))

        raise FileNotFoundError("No avo_depth cache found and generation not allowed")

    @classmethod
    def from_npz(
        cls, path: str, key: Optional[str] = None
    ) -> "SeismogramTopLayersExtractor":
        """Load an ndarray from an NPZ file and construct an extractor.

        If `key` is provided, the array at that key will be used. Otherwise
        the method will pick the first ndarray value found in the archive.
        """
        data = np.load(path)
        arr: Optional[NDArray[Any]] = None
        if key is not None:
            arr = data[key]
        else:
            # pick first ndarray from the archive

            for v in data.values():
                if isinstance(v, np.ndarray):
                    arr = cast(NDArray[Any], v)
                    break

        if arr is None:
            raise ValueError(f"no ndarray found in npz: {path}")
        assert arr is not None
        return cls(arr)


__all__ = ["SeismogramTopLayersExtractor", "CacheProvider", "DefaultCacheProvider"]


class CacheProvider(Protocol):
    """Protocol describing an injectable cache provider for depth-domain AVO caches."""

    def load_latest_depth(self) -> Optional[NDArray[Any]]: ...

    def generate_depth(self, force: bool = False) -> Optional[NDArray[Any]]: ...


class DefaultCacheProvider:
    """Default implementation that uses `CacheManager` and `ModelingPipeline`.

    This class is small and injectable so callers can provide mocks in tests.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        cache_manager: Any | None = None,
        pipeline: Any | None = None,
    ):
        self.cache_dir = cache_dir
        self._cache_manager = cache_manager
        self._pipeline = pipeline

    def _find_latest(self) -> Optional[str]:
        from pathlib import Path

        d = Path(self.cache_dir)
        if not d.exists():
            return None
        # Prefer AVO depth caches but also accept top_layers NPZs created by this tool
        files = list(d.glob("avo_depth_*.npz")) + list(d.glob("top_layers_*.npz"))
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0]) if files else None

    def load_latest_depth(self) -> Optional[NDArray[Any]]:
        path = self._find_latest()
        if path is None:
            return None
        # lazy import
        from pathlib import Path
        import numpy as _np

        p = Path(path)
        name = p.name
        # If this is a top_layers file produced by the extractor, load the 'top' array
        if name.startswith("top_layers_"):
            try:
                data = _np.load(str(p), allow_pickle=True)
                if "top" in data:
                    return cast(NDArray[Any], data["top"])
            except Exception:
                pass

        # Fallback to CacheManager's AVO loader for full_stack_depth-style caches
        try:
            from src.modeling.model_cache import CacheManager

            cm = self._cache_manager or CacheManager(cache_dir=str(p.parent))
            _, full_stack_depth = cm.load_avo_synthetics(name)
            return cast(Optional[NDArray[Any]], full_stack_depth)
        except Exception:
            return None

    def generate_depth(self, force: bool = False) -> Optional[NDArray[Any]]:
        # If not forced and a cache exists, return it
        if not force:
            path = self._find_latest()
            if path:
                return self.load_latest_depth()

        # lazy import of pipeline
        from src.modeling.pipeline import ModelingPipeline

        mp = self._pipeline or ModelingPipeline()
        result = mp.run()
        if (
            isinstance(result, dict)
            and "full_stack_depth" in result
            and result["full_stack_depth"] is not None
        ):
            return cast(NDArray[Any], result["full_stack_depth"])
        # fallback: try to find cache file created by pipeline
        path = self._find_latest()
        if path:
            return self.load_latest_depth()
        raise RuntimeError("Failed to generate or locate avo_depth cache")
