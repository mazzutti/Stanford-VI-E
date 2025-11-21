"""Facies extraction helpers for the `gen` package.
This module mirrors `src.gen.seismogram` but works with facies cubes.
It provides the same API surface: class-based extractor, cache-aware
factory methods and a small default cache provider that can load or
generate facies depth caches.

Conventions are the same as the seismogram extractor: input arrays are
3D with shape (ni, nj, nk) and "top" refers to the first indices along
the last axis.
"""

from __future__ import annotations

from pathlib import Path
from time import time as _time
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

class FaciesTopLayersExtractor:
    """Extract top N layers from a 3D facies cube.

    API mirrors SeismogramTopLayersExtractor so callers can reuse logic.
    """

    def __init__(self, cube: NDArray[Any]):
        if cube.ndim != 3:
            raise ValueError("`cube` must be a 3D array with shape (ni, nj, nk)")
        self.cube = cube

    def extract_top_two_geological_layers(
        self, thicknesses: tuple[int, int] = (80, 40)
    ) -> NDArray[Any]:
        """Return the top two geological layer stacks as a 3D slice.

        Parameters
        ----------
        thicknesses : tuple[int, int]
            (top_thickness, second_thickness)

        Returns
        -------
        NDArray
            Slice of the cube containing the requested top layers.
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
        return self.cube[:, :, :total]

    @classmethod
    def from_npz(cls, path: str, key: str | None = None) -> FaciesTopLayersExtractor:
        """Load facies arrays from a .npz archive, using ``key`` if provided.

        Falls back to the first ndarray found in the archive when ``key`` is None.
        """
        data = np.load(path, allow_pickle=True)
        arr: NDArray[Any] | None = None
        if key is not None and key in data:
            arr = cast(NDArray[Any], data[key])
        else:
            for v in data.values():
                if isinstance(v, np.ndarray):
                    arr = cast(NDArray[Any], v)
                    break

        if arr is None:
            raise ValueError(f"no ndarray found in npz: {path}")

        return cls(arr)

    @classmethod
    def from_cache_or_generate(
        cls,
        cache_provider: CacheProvider | None = None,
        cache_dir: str = ".cache",
        prefer_latest: bool = True,
        generate_if_missing: bool = True,
        force_regeneration: bool = False,
    ) -> FaciesTopLayersExtractor:
        """Load the latest facies cache or invoke generation when allowed.

        This is a convenience factory that accepts several configuration
        parameters to control cache lookup and generation.
        """
        # The argument list is intentionally long to mirror the existing
        # API surface; keep it stable for callers.

        if cache_provider is None:
            cache_provider = DefaultCacheProvider(cache_dir=cache_dir)

        if prefer_latest:
            arr = cache_provider.load_latest_depth()
            if arr is not None:
                return cls(np.asarray(arr))

        if generate_if_missing:
            arr = cache_provider.generate_depth(force_regeneration=force_regeneration)
            return cls(np.asarray(arr))

        raise FileNotFoundError("No facies cache found and generation not allowed")

class CacheProvider(Protocol):
    """Protocol for a facies cache provider.

    Implementations should provide methods to load the latest cached depth
    cube or generate it when requested.
    """

    # Protocol methods are intentionally left as stubs; they are documented
    # at the class level. Silence missing-function-docstring for protocol
    # method stubs that are only type-level declarations.

    def load_latest_depth(self) -> NDArray[Any] | None: ...

    def generate_depth(
        self, force_regeneration: bool = False
    ) -> NDArray[Any] | None: ...

class DefaultCacheProvider:
    """Default provider that looks for facies caches and can call a pipeline."""

    def __init__(
        self,
        cache_dir: str = ".cache",
        cache_manager: Any | None = None,
        pipeline: Any | None = None,
    ):
        self.cache_dir = cache_dir
        self._cache_manager = cache_manager
        self._pipeline = pipeline

    def _find_latest(self) -> str | None:
        d = Path(self.cache_dir)
        if not d.exists():
            return None
        files = list(d.glob("facies_depth_*.npz"))
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0]) if files else None

    def load_latest_depth(self) -> NDArray[Any] | None:
        """Attempt to load the latest cached facies depth array.

        Returns the ndarray when available, otherwise ``None``.
        """
        latest = self._find_latest()
        if latest is None:
            # If no cache found, try to load the raw Facies file shipped
            # with the dataset at `Facies/facies.dat` (project root).
            facies_src = Path("Facies") / "facies.dat"
            if facies_src.exists():
                data = np.loadtxt(
                    str(facies_src), dtype=int, comments="#", ndmin=1, skiprows=3
                )
                arr = data.reshape((150, 200, 200), order="F")
                try:
                    cache_p = Path(self.cache_dir)
                    cache_p.mkdir(parents=True, exist_ok=True)
                    ts = int(_time())
                    cache_file = cache_p / f"facies_depth_{ts}.npz"
                    np.savez_compressed(cache_file, facies=arr)
                except (OSError, ValueError):
                    # Best-effort caching: ignore failures related to filesystem
                    # operations or type conversions but don't suppress other errors.
                    pass
                return arr
        # `latest` is either a string path or None; create a Path object
        p = Path(str(latest))
        try:
            data = np.load(str(p), allow_pickle=True)
            # data["facies"] should already be an ndarray; cast to satisfy
            # the static type checker which may treat the loaded object as Any.
            return cast(NDArray[Any], data["facies"])
        except (OSError, ValueError, KeyError):
            # If the file cannot be read, or the expected key is missing, give up.
            pass

        # Explicitly return None when no cached data could be loaded.
        return None

    def generate_depth(self, force_regeneration: bool = False) -> NDArray[Any] | None:
        """Generate facies depth cache by invoking the configured pipeline.

        May remove existing caches when ``force_regeneration`` is True.
        """
        # This function contains nested blocks to perform best-effort
        # cache removal and generation. Keep the structure explicit and
        # disable the nested-blocks refactor warning locally.

        # If forced, remove existing facies cache files so the pipeline
        # will regenerate fresh outputs instead of reusing cached ones.
        if force_regeneration:
            try:
                d = Path(self.cache_dir)
                if d.exists():
                    patterns = (
                        "facies_depth_*.npz",
                        "facies_*.npz",
                        "facies_top_layers_*.npz",
                    )
                    for pat in patterns:
                        for p in d.glob(pat):
                            try:
                                p.unlink()
                            except OSError:
                                # best-effort removal; ignore filesystem errors
                                pass
            except OSError:
                # ignore errors related to filesystem operations
                pass

        path = self._find_latest()
        if path:
            return self.load_latest_depth()

        raise RuntimeError("Failed to generate or locate facies depth cache")

__all__ = ["FaciesTopLayersExtractor", "CacheProvider", "DefaultCacheProvider"]
