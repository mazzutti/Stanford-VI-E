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
from typing import Optional, Protocol, Any, cast, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - only for static type checkers
    import plotly.graph_objects as go


class FaciesTopLayersExtractor:
    """Extract top N layers from a 3D facies cube.

    API mirrors SeismogramTopLayersExtractor so callers can reuse logic.
    """

    def __init__(self, cube: NDArray[Any]):
        if cube.ndim != 3:
            raise ValueError("`cube` must be a 3D array with shape (ni, nj, nk)")
        self.cube = cube

    def extract_top_layers(self, n_layers: int = 2) -> NDArray[Any]:
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        nk = self.cube.shape[2]
        if n_layers > nk:
            raise ValueError(
                "n_layers cannot be larger than number of samples (nk={})".format(nk)
            )
        return self.cube[:, :, :n_layers].copy()

    def extract_top_layer(self, index: int = 0) -> NDArray[Any]:
        nk = self.cube.shape[2]
        if index < 0 or index >= nk:
            raise IndexError("layer index out of range")
        return self.cube[:, :, index].copy()

    def top_layers_mean(self, n_layers: int = 2) -> NDArray[Any]:
        layers = self.extract_top_layers(n_layers)
        # Always return a 2D array (ni, nj). If there are no layers, return
        # an array of zeros with float dtype to keep the return type stable.
        if layers.size == 0:
            ni, nj = self.cube.shape[0], self.cube.shape[1]
            return np.zeros((ni, nj), dtype=np.float64)
        return cast(NDArray[Any], np.mean(layers, axis=2))

    def extract_top_two_geological_layers(
        self, thicknesses: tuple[int, int] = (80, 40)
    ) -> NDArray[Any]:
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

    def create_plotly_figure(
        self,
        slice_indices: tuple[int, int, int] | None = None,
        full_depth: NDArray[Any] | None = None,
        embed_in_full_depth: bool = False,
        is_categorical: bool = False,
        title: str = "",
    ) -> "go.Figure":
        """Create a Plotly Figure for this cube.

        If `embed_in_full_depth` is True, `full_depth` must be provided and the
        extractor's cube (typically containing just the top layers) will be
        placed into the top slices of a full-size cube (NaN elsewhere) so the
        Z-axis reflects the full depth.
        """
        import numpy as _np

        from src.plotting.plotly_plotter import PlotlyPlotter

        # Import Plotly types for better type-checker propagation when
        # running static analysis. The runtime import is guarded so this
        # function remains cheap at runtime.

        if embed_in_full_depth:
            if full_depth is None:
                raise ValueError(
                    "full_depth must be provided when embedding in full depth"
                )
            ni, nj, nk_full = full_depth.shape
            # Create a masked full cube and copy top layers to its first indices
            masked = _np.full((ni, nj, nk_full), _np.nan, dtype=float)
            n_layers = self.cube.shape[2]
            masked[:, :, :n_layers] = self.cube
            arr = masked
        else:
            arr = _np.asarray(self.cube)

        ni, nj, nk = arr.shape
        if slice_indices is None:
            slice_indices = (ni // 2, nj // 2, nk // 2)

        plotter = PlotlyPlotter()
        traces = plotter.create_3d_volume(
            arr, slice_indices, is_categorical=is_categorical
        )
        fig = plotter.create_figure(traces, title=title)
        return fig

    def save_plotly_html(
        self,
        filepath: str | Path,
        slice_indices: tuple[int, int, int] | None = None,
        full_depth: NDArray[Any] | None = None,
        embed_in_full_depth: bool = False,
        is_categorical: bool = False,
        title: str = "",
    ) -> str:
        """Create and save a Plotly HTML file for this cube. Returns the path."""
        from pathlib import Path as _Path

        fig = self.create_plotly_figure(
            slice_indices=slice_indices,
            full_depth=full_depth,
            embed_in_full_depth=embed_in_full_depth,
            is_categorical=is_categorical,
            title=title,
        )
        out = _Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Use the PlotlyPlotter saving (injects interaction script)
        from src.plotting.plotly_plotter import PlotlyPlotter

        plotter = PlotlyPlotter()
        plotter.save_figure(fig, str(out))
        return str(out)

    def save_matplotlib_png(
        self, output_path: str | Path, domain: str = "depth"
    ) -> str:
        """Save a static PNG using the Matplotlib renderer via SeismicPlotter.

        Returns the PNG path.
        """
        from pathlib import Path as _Path

        out = _Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            from src.plotting.seismic_plotter import SeismicPlotter

            sp = SeismicPlotter(cache_dir=".cache", out_dir=str(out.parent))
            sp.plot_full_stack(self.cube, output_path=out, domain=domain)
            return str(out)
        except Exception as e:
            raise RuntimeError(f"Matplotlib PNG export failed: {e}")

    @classmethod
    def from_npz(
        cls, path: str, key: Optional[str] = None
    ) -> "FaciesTopLayersExtractor":
        data = np.load(path, allow_pickle=True)
        if key is not None and key in data:
            arr = data[key]
        else:
            arr = None
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
    def load_latest_depth(self) -> Optional[NDArray[Any]]: ...

    def generate_depth(self, force: bool = False) -> Optional[NDArray[Any]]: ...


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

    def _find_latest(self) -> Optional[str]:
        from pathlib import Path

        d = Path(self.cache_dir)
        if not d.exists():
            return None
        files = (
            list(d.glob("facies_depth_*.npz"))
            + list(d.glob("facies_*.npz"))
            + list(d.glob("facies_top_layers_*.npz"))
        )
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0]) if files else None

    def load_latest_depth(self) -> Optional[NDArray[Any]]:
        from pathlib import Path
        import numpy as _np

        path = self._find_latest()
        if path is None:
            # If no cache found, try to load the raw Facies file shipped
            # with the dataset at `Facies/facies.dat` (project root).
            facies_src = Path("Facies") / "facies.dat"
            if facies_src.exists():
                try:
                    data = _np.loadtxt(
                        str(facies_src), dtype=int, comments="#", ndmin=1
                    )
                except Exception:
                    # Fallback: read tokens and parse ints (more robust)
                    txt = facies_src.read_text()
                    tokens: list[int] = []
                    for tok in txt.split():
                        try:
                            tokens.append(int(tok))
                        except Exception:
                            continue
                    data = _np.array(tokens, dtype=int)

                # The Stanford VI grid uses shape (150, 200, 200)
                expected = 150 * 200 * 200
                if data.size == expected:
                    arr = data.reshape((150, 200, 200))
                    return arr
                else:
                    # If sizes don't match, attempt to find a plausible reshape
                    if data.size >= 3:
                        # try (ni, nj, nk) where ni*nj*nk == data.size
                        # common candidate is (150,200,200)
                        try:
                            arr = data.reshape((150, 200, -1))
                            if arr.size == data.size:
                                return arr
                        except Exception:
                            pass
                    # give up and return None so callers fall back
                    return None
        p = Path(str(path))
        name = p.name
        try:
            data = _np.load(str(p), allow_pickle=True)
            # try to load known keys
            if "facies" in data:
                return cast(NDArray[Any], data["facies"])
            if "top" in data:
                return cast(NDArray[Any], data["top"])
            # fallback: first ndarray
            for v in data.values():
                if isinstance(v, _np.ndarray):
                    return cast(NDArray[Any], v)
        except Exception:
            pass

        # fallback: try cache manager if available
        try:
            from src.modeling.model_cache import CacheManager

            cm = self._cache_manager or CacheManager(cache_dir=str(p.parent))
            # try generic load (may not be available for facies)
            _, arr = cm.load_avo_synthetics(name)
            return cast(NDArray[Any], arr)
        except Exception:
            return None

    def generate_depth(self, force: bool = False) -> Optional[NDArray[Any]]:
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
                        return cast(NDArray[Any], result[k])
        except Exception:
            pass

        # fallback to cached file if pipeline didn't produce one
        path = self._find_latest()
        if path:
            return self.load_latest_depth()

        raise RuntimeError("Failed to generate or locate facies depth cache")


__all__ = ["FaciesTopLayersExtractor", "CacheProvider", "DefaultCacheProvider"]
