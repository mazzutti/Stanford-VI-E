"""Seismic full-stack Plotly generator (OOP integration).

Provides SeismicPlotter which uses the existing PlotlyPlotter helper to
create interactive 3D HTML visualizations for full-stack seismograms in
both time and depth domains. Designed to mirror the project's OOP style
and integrate with the CLI tool registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Any, Optional

import numpy as np

from src.plotting.plotly_plotter import PlotlyPlotter
from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.config import PlotConfig
from src.plotting.helpers.components import ImageRenderer, SliceExtractor

import matplotlib
import matplotlib.pyplot as plt
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
        backend: str = "Agg",
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

    def _select_cache(self, domain: str) -> Optional[Path]:
        pattern = "avo_time*.npz" if domain == "time" else "avo_depth*.npz"
        candidates = list(self.cache_dir.glob(pattern))
        if not candidates:
            logger.warning(
                "No AVO cache found for pattern %s in %s", pattern, self.cache_dir
            )
            return None
        return candidates[0]

    def _load_full_stack(self, npz_path: Path) -> np.ndarray:
        npz = np.load(npz_path, allow_pickle=True)
        if "full_stack" not in npz:
            raise KeyError(f"'full_stack' key not found in {npz_path}")
        return npz["full_stack"]

    # --------- Plotly (interactive) -------------------------------------------------
    def generate_from_caches(self, domain: str = "time") -> List[Path]:
        """Generate interactive HTML(s) from cache for given domain and return generated paths."""
        results: List[Path] = []
        cache_file = self._select_cache(domain)
        if cache_file is None:
            logger.error("No cache file available for domain: %s", domain)
            return results

        logger.info("Loading full_stack from %s", cache_file)
        cube = self._load_full_stack(cache_file)

        # Use middle slices
        ni, nj, nk = cube.shape
        slice_indices = (ni // 2, nj // 2, nk // 2)

        title = (
            "Full-Stack AVO Seismogram (Time Domain)"
            if domain == "time"
            else "Full-Stack AVO Seismogram (Depth Domain)"
        )

        # Use the project's seismic colormap (matplotlib 'seismic') as default
        config = PlotConfig.for_seismic(k_unit="")
        cmap_name = config.cmap or "seismic"

        traces = self._plotly.create_3d_volume(
            cube,
            slice_indices=slice_indices,
            title=title,
            k_label="Time/Depth",
            k_unit="",
            colorscale=cmap_name,
            show_colorbar=True,
        )

        fig = self._plotly.create_figure(traces, title=title)
        out_name = f"seismic_full_stack_{domain}_3d.html"
        out_path = self.out_dir / out_name
        self._plotly.save_figure(fig, str(out_path))
        logger.info("Saved seismic full-stack interactive HTML: %s", out_path)
        results.append(out_path)

        return results

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

        extractor = SliceExtractor(shape=(ni, nj, nk))

        k_label = "Time (ms)" if domain == "time" else "Depth (m)"
        # If the provided data is integer (facies), use categorical plotting
        if np.issubdtype(seismogram.dtype, np.integer) or np.allclose(
            seismogram, seismogram.astype(int)
        ):
            config = PlotConfig.for_categorical()
            config = config.update(show_colorbar=True)
            # Do not compute vmin/vmax for categorical data
        else:
            config = PlotConfig.for_seismic(k_unit=k_label)
            config = config.update(cmap="seismic", show_colorbar=True)

            vmax = np.percentile(np.abs(seismogram), 99)
            vmin = -vmax
            config = config.update(extra_kwargs={"vmin": vmin, "vmax": vmax})

        inline_data, xlabel, ylabel = extractor.extract_inline(seismogram, mid_i)
        config_inline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Inline (i={mid_i})",
        )
        ImageRenderer.render(axes[0], inline_data, config_inline)

        crossline_data, xlabel, ylabel = extractor.extract_crossline(seismogram, mid_j)
        config_crossline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Crossline (j={mid_j})",
        )
        ImageRenderer.render(axes[1], crossline_data, config_crossline)

        depth_data, xlabel, ylabel = extractor.extract_depthslice(seismogram, mid_k)
        slice_label = f"{k_label.split()[0]} Slice"
        config_depth = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{slice_label} (k={mid_k})",
        )
        ImageRenderer.render(axes[2], depth_data, config_depth)

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"  ✓ Generated: {output_path}")
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

        extractor = SliceExtractor(shape=(ni, nj, nk))

        k_label = "Time (ms)" if domain == "time" else "Depth (m)"
        # If the provided data looks like categorical facies (integers), switch
        # to categorical PlotConfig so the ImageRenderer draws discrete colors
        if np.issubdtype(full_stack.dtype, np.integer) or np.allclose(
            full_stack, full_stack.astype(int)
        ):
            config = PlotConfig.for_categorical()
            config = config.update(show_colorbar=True)
        else:
            config = PlotConfig.for_seismic(k_unit=k_label)
            config = config.update(cmap="seismic", show_colorbar=True)

            vmax = np.percentile(np.abs(full_stack), 99)
            vmin = -vmax
            config = config.update(extra_kwargs={"vmin": vmin, "vmax": vmax})

        inline_data, xlabel, ylabel = extractor.extract_inline(full_stack, mid_i)
        config_inline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Inline {mid_i}",
        )
        ImageRenderer.render(axes[0], inline_data, config_inline)

        crossline_data, xlabel, ylabel = extractor.extract_crossline(full_stack, mid_j)
        config_crossline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Crossline {mid_j}",
        )
        ImageRenderer.render(axes[1], crossline_data, config_crossline)

        depth_data, xlabel, ylabel = extractor.extract_depthslice(full_stack, mid_k)
        slice_label = f"{k_label.split()[0]} Slice at {mid_k}{k_label.split()[1][1:-1]}"
        config_depth = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=slice_label,
        )
        ImageRenderer.render(axes[2], depth_data, config_depth)

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"  ✓ Generated: {output_path}")
        return output_path

    def plot_from_cache(
        self,
        cache_file: Path,
        output_dir: Path,
        domain: str = "time",
        angles: Optional[list[float]] = None,
    ) -> dict[str, list[Path]]:
        """Generate all seismogram plots from cache file (angle stacks + full stack)."""
        logger.info(f"Loading seismogram data from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)

        suffix = "_depth" if domain == "depth" else ""

        generated_files = {"angle_stacks": [], "full_stack": []}

        # Plot angle stacks
        if "angle_stacks" in data or any(k.startswith("angle_") for k in data.keys()):
            logger.info(f"Generating angle stack plots ({domain} domain)...")

            if "angle_stacks" in data:
                angle_stacks = data["angle_stacks"]
                if angles is None:
                    angles = data.get("angles", list(range(len(angle_stacks))))
            else:
                angle_keys = sorted([k for k in data.keys() if k.startswith("angle_")])
                angle_stacks = [data[k] for k in angle_keys]
                if angles is None:
                    angles = [int(k.split("_")[1]) for k in angle_keys]

            for i, (angle_stack, angle) in enumerate(zip(angle_stacks, angles)):
                output_file = output_dir / f"seismic_angle_{i}{suffix}.png"
                self.plot_angle_stack(
                    angle_stack,
                    angle=angle,
                    output_path=output_file,
                    domain=domain,
                )
                generated_files["angle_stacks"].append(output_file)

        # Plot full stack
        if "full_stack" in data:
            logger.info(f"Generating full stack plot ({domain} domain)...")
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
        logger.info(f"✓ Generated {total_files} seismogram plot(s) ({domain} domain)")

        return generated_files
