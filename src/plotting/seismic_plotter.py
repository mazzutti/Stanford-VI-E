"""Seismic full-stack Plotly generator (OOP integration).

Provides SeismicPlotter which uses the existing PlotlyPlotter helper to
create interactive 3D HTML visualizations for full-stack seismograms in
both time and depth domains. Designed to mirror the project's OOP style
and integrate with the CLI tool registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.components import ImageRenderer, SliceExtractor
from src.plotting.helpers.config import PlotConfig
from src.plotting.plotly_plotter import PlotlyPlotter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - static typing only
    pass


class SeismicPlotter(BasePlotter):
    """Unified SeismicPlotter providing both Plotly HTML and Matplotlib PNG outputs.

    Methods:
      - generate_from_caches(domain): create interactive Plotly HTML(s)
      - plot_from_cache(cache_file, output_dir, domain): create PNGs (angle stacks + full-stack)

    This merges the previous Plotly-only SeismicPlotter with the Matplotlib
    SeismogramPlotter functionality so consumers can use a single class.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        out_dir: str = "docs/images",
        backend: str = "qtagg",
        verbose: bool = False,
    ):
        # Initialize BasePlotter (may set Matplotlib backend)
        super().__init__(backend=backend)
        self.cache_dir = Path(cache_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._plotly = PlotlyPlotter()
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

    def _select_cache(self, domain: str) -> Path | None:
        """Return the first cache Path matching the requested domain or None.

        This helper looks for AVO cache files following the project's naming
        conventions and returns the first candidate or None when not found.
        """

        pattern = "avo_time*.npz" if domain == "time" else "avo_depth*.npz"
        candidates = list(self.cache_dir.glob(pattern))
        if not candidates:
            logger.warning(
                "No AVO cache found for pattern %s in %s", pattern, self.cache_dir
            )
            return None
        return candidates[0]

    def _load_full_stack(self, npz_path: Path) -> NDArray[np.floating[Any]]:
        """Load and return the 'full_stack' array from an NPZ file.

        Raises KeyError when the expected key is missing.
        """

        npz = np.load(npz_path, allow_pickle=True)
        if "full_stack" not in npz:
            raise KeyError(f"'full_stack' key not found in {npz_path}")
        return cast(NDArray[np.floating[Any]], npz["full_stack"])

    # --------- Plotly (interactive) -------------------------------------------------
    def generate_from_caches(self, domain: str = "time") -> list[Path]:
        """Generate interactive HTML(s) from cache for given domain and return generated paths."""
        cache_file = self._select_cache(domain)
        if cache_file is None:
            logger.error("No cache file available for domain: %s", domain)
            return []

        logger.info("Loading full_stack from %s", cache_file)
        cube = self._load_full_stack(cache_file)

        return self._generate_plotly_from_cube(cube, domain)

    # --------- Matplotlib (PNGs) ---------------------------------------------------
    def plot_angle_stack(
        self,
        seismogram: NDArray[np.floating[Any]],
        angle: float,
        output_path: Path,
        domain: str = "time",
        title_suffix: str = "",
    ) -> Path:
        """Plot a single angle stack with 3 orthogonal slices (PNG)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        domain_label = "Time Domain (TWT)" if domain == "time" else "Depth Domain"
        title = f"Seismic Angle Stack - {angle}° - {domain_label}{title_suffix}"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        ni, nj, nk = seismogram.shape
        mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2
        # Prepare extractor/config and render the three orthogonal slices.
        extractor, config, k_label = self._prepare_extractor_and_config(
            seismogram, domain
        )

        slice_label = f"{k_label.split()[0]} Slice"
        titles = (
            f"Inline (i={mid_i})",
            f"Crossline (j={mid_j})",
            f"{slice_label} (k={mid_k})",
        )
        self._render_three_slices(
            axes, seismogram, extractor, config, mid_i, mid_j, mid_k, k_label, titles
        )

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("  ✓ Generated: %s", output_path)
        return output_path

    def plot_full_stack(
        self,
        full_stack: NDArray[np.floating[Any]],
        output_path: Path,
        domain: str = "time",
        title_suffix: str = "",
    ) -> Path:
        """Plot full stack seismogram with 3 orthogonal slices (PNG)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        domain_label = "Time Domain (TWT)" if domain == "time" else "Depth Domain"
        title = f"Seismic Full Stack - {domain_label}{title_suffix}"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        ni, nj, nk = full_stack.shape
        mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2
        # Prepare extractor/config and render the three orthogonal slices.
        extractor, config, k_label = self._prepare_extractor_and_config(
            full_stack, domain
        )

        slice_label = f"{k_label.split()[0]} Slice at {mid_k}{k_label.split()[1][1:-1]}"
        titles = (f"Inline {mid_i}", f"Crossline {mid_j}", slice_label)
        self._render_three_slices(
            axes, full_stack, extractor, config, mid_i, mid_j, mid_k, k_label, titles
        )

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("  ✓ Generated: %s", output_path)
        return output_path

    def plot_from_cache(
        self,
        cache_file: Path,
        output_dir: Path,
        domain: str = "time",
        angles: list[float] | None = None,
    ) -> dict[str, list[Path]]:
        """Generate all seismogram plots from cache file (angle stacks + full stack)."""
        logger.info("Loading seismogram data from: %s", cache_file)
        data = np.load(cache_file, allow_pickle=True)
        return self._generate_plots_from_cache(data, output_dir, domain, angles)

    def _generate_plotly_from_cube(
        self, cube: NDArray[np.floating[Any]], domain: str
    ) -> list[Path]:
        """Generate and save the Plotly full-stack HTML for `cube` and
        return list of generated paths.
        """
        ni, nj, nk = cube.shape
        slice_indices = (ni // 2, nj // 2, nk // 2)

        title = (
            "Full-Stack AVO Seismogram (Time Domain)"
            if domain == "time"
            else "Full-Stack AVO Seismogram (Depth Domain)"
        )

        config = PlotConfig.for_seismic(k_unit="")
        cmap_name = config.cmap or "seismic"

        traces: list[go.Surface] = self._plotly.create_3d_volume(
            cube, slice_indices, colorscale=cmap_name, show_colorbar=True
        )

        fig: go.Figure = self._plotly.create_figure(traces, title=title)
        out_name = f"seismic_full_stack_{domain}_3d.html"
        out_path = self.out_dir / out_name
        self._plotly.save_figure(fig, str(out_path))
        logger.info("Saved seismic full-stack interactive HTML: %s", out_path)
        return [out_path]

    def _generate_plots_from_cache(
        self,
        data: Any,
        output_dir: Path,
        domain: str,
        angles: list[float] | None,
    ) -> dict[str, list[Path]]:
        """Generate angle-stack and full-stack PNGs from a loaded NPZ `data` object."""
        suffix = "_depth" if domain == "depth" else ""
        generated_files: dict[str, list[Path]] = {"angle_stacks": [], "full_stack": []}

        # Angle stacks
        if "angle_stacks" in data or any(k.startswith("angle_") for k in data.keys()):
            logger.info("Generating angle stack plots (%s domain)...", domain)

            if "angle_stacks" in data:
                angle_stacks = data["angle_stacks"]
                if angles is None:
                    angles = data.get("angles", list(range(len(angle_stacks))))
            else:
                angle_keys = sorted([k for k in data.keys() if k.startswith("angle_")])
                angle_stacks = [data[k] for k in angle_keys]
                if angles is None:
                    angles = [int(k.split("_")[1]) for k in angle_keys]

            angles_iter = cast(list[float], angles)
            for i, (angle_stack, angle) in enumerate(zip(angle_stacks, angles_iter)):
                output_file = output_dir / f"seismic_angle_{i}{suffix}.png"
                self.plot_angle_stack(
                    angle_stack,
                    angle=angle,
                    output_path=output_file,
                    domain=domain,
                )
                generated_files["angle_stacks"].append(output_file)

        # Full stack
        if "full_stack" in data:
            logger.info("Generating full stack plot (%s domain)...", domain)
            output_file = output_dir / f"seismic_full_stack{suffix}.png"
            self.plot_full_stack(
                data["full_stack"],
                output_path=output_file,
                domain=domain,
            )
            generated_files["full_stack"].append(output_file)

        total_files = len(generated_files["angle_stacks"]) + len(
            generated_files["full_stack"]
        )
        logger.info(
            "✓ Generated %d seismogram plot(s) (%s domain)", total_files, domain
        )
        return generated_files

    # --- Small helpers to reduce create-method complexity ---------------------
    def _prepare_extractor_and_config(
        self, arr: NDArray[np.floating[Any]], domain: str
    ) -> tuple["SliceExtractor", PlotConfig, str]:
        """Return (extractor, plot_config, k_label) for the provided array and domain.

        Keeps selection logic for categorical vs seismic data in one place.
        """
        ni, nj, nk = arr.shape
        extractor = SliceExtractor(shape=(ni, nj, nk))

        k_label = "Time (ms)" if domain == "time" else "Depth (m)"
        if np.issubdtype(arr.dtype, np.integer) or np.allclose(arr, arr.astype(int)):
            config = PlotConfig.for_categorical()
            config = config.update(show_colorbar=True)
        else:
            config = PlotConfig.for_seismic(k_unit=k_label)
            config = config.update(cmap="seismic", show_colorbar=True)

            vmax = np.percentile(np.abs(arr), 99)
            vmin = -vmax
            config = config.update(extra_kwargs={"vmin": vmin, "vmax": vmax})

        return extractor, config, k_label

    def _render_three_slices(
        self,
        axes: Sequence[Axes] | NDArray[Any],
        arr: NDArray[np.floating[Any]],
        extractor: SliceExtractor,
        config: PlotConfig,
        mid_i: int,
        mid_j: int,
        mid_k: int,
        k_label: str,
        titles: tuple[str, str, str],
    ) -> None:
        # `k_label` kept for compatibility; not used in this renderer.

        """Render inline, crossline and depth slices into `axes` using `config`.

        `titles` should be a tuple of three strings for each subplot.
        """
        inline_data, xlabel, ylabel = extractor.extract_inline(arr, mid_i)
        config_inline = config.update(xlabel=xlabel, ylabel=ylabel, title=titles[0])
        _ = cast(Any, ImageRenderer.render(axes[0], inline_data, config_inline))

        crossline_data, xlabel, ylabel = extractor.extract_crossline(arr, mid_j)
        config_crossline = config.update(xlabel=xlabel, ylabel=ylabel, title=titles[1])
        _ = cast(Any, ImageRenderer.render(axes[1], crossline_data, config_crossline))

        depth_data, xlabel, ylabel = extractor.extract_depthslice(arr, mid_k)
        config_depth = config.update(xlabel=xlabel, ylabel=ylabel, title=titles[2])
        _ = cast(Any, ImageRenderer.render(axes[2], depth_data, config_depth))
