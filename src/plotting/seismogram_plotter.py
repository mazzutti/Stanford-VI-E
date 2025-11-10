"""Seismogram plotting module for AVO analysis.

Generates PNG visualizations for seismogram data in both time and depth domains.
Creates individual angle stack plots and full stack plots with 3-slice views.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.config import PlotConfig
from src.plotting.helpers.components import ImageRenderer, SliceExtractor

logger = logging.getLogger(__name__)


class SeismogramPlotter(BasePlotter):
    """Plotter for seismogram visualizations (time and depth domains).

    Generates 3-slice PNG plots for AVO angle stacks and full stack seismograms.
    Handles both time domain (TWT) and depth domain visualizations.
    """

    def __init__(self, backend: str = "Agg", verbose: bool = False):
        """Initialize seismogram plotter.

        Parameters
        ----------
        backend : str
            Matplotlib backend to use, default: Agg (non-interactive)
        verbose : bool
            Enable verbose logging, default: False
        """
        super().__init__(backend=backend)
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

    def plot_angle_stack(
        self,
        seismogram: NDArray[np.floating[Any]],
        angle: float,
        output_path: Path,
        domain: str = "time",
        title_suffix: str = "",
    ) -> Path:
        """Plot a single angle stack with 3 orthogonal slices.

        Parameters
        ----------
        seismogram : NDArray[np.floating[Any]]
            3D seismogram array (ni, nj, nk)
        angle : float
            Angle in degrees
        output_path : Path
            Output file path for PNG
        domain : str
            Domain type: "time" or "depth", default: time
        title_suffix : str
            Optional suffix for title, default: empty

        Returns
        -------
        Path
            Path to generated PNG file
        """
        # Setup figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        domain_label = "Time Domain (TWT)" if domain == "time" else "Depth Domain"
        title = f"Seismic Angle Stack - {angle}° - {domain_label}{title_suffix}"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Get data shape and middle indices
        ni, nj, nk = seismogram.shape
        mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2

        # Create slice extractor
        extractor = SliceExtractor(shape=(ni, nj, nk))

        # Configure plot settings
        k_label = "Time (ms)" if domain == "time" else "Depth (m)"
        config = PlotConfig.for_seismic(k_unit=k_label)
        config = config.update(cmap="seismic", show_colorbar=True)

        # Get symmetric color limits
        vmax = np.percentile(np.abs(seismogram), 99)
        vmin = -vmax
        config = config.update(extra_kwargs={"vmin": vmin, "vmax": vmax})

        # Inline slice
        inline_data, xlabel, ylabel = extractor.extract_inline(seismogram, mid_i)
        config_inline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Inline (i={mid_i})",
        )
        ImageRenderer.render(axes[0], inline_data, config_inline)

        # Crossline slice
        crossline_data, xlabel, ylabel = extractor.extract_crossline(seismogram, mid_j)
        config_crossline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Crossline (j={mid_j})",
        )
        ImageRenderer.render(axes[1], crossline_data, config_crossline)

        # Depth/Time slice
        depth_data, xlabel, ylabel = extractor.extract_depthslice(seismogram, mid_k)
        slice_label = f"{k_label.split()[0]} Slice"
        config_depth = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{slice_label} (k={mid_k})",
        )
        ImageRenderer.render(axes[2], depth_data, config_depth)

        plt.tight_layout()

        # Save
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
        """Plot full stack seismogram with 3 orthogonal slices.

        Parameters
        ----------
        full_stack : NDArray[np.floating[Any]]
            3D full stack array (ni, nj, nk)
        output_path : Path
            Output file path for PNG
        domain : str
            Domain type: "time" or "depth", default: time
        title_suffix : str
            Optional suffix for title, default: empty

        Returns
        -------
        Path
            Path to generated PNG file
        """
        # Setup figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        domain_label = "Time Domain (TWT)" if domain == "time" else "Depth Domain"
        title = f"Seismic Full Stack - {domain_label}{title_suffix}"
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Get data shape and middle indices
        ni, nj, nk = full_stack.shape
        mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2

        # Create slice extractor
        extractor = SliceExtractor(shape=(ni, nj, nk))

        # Configure plot settings
        k_label = "Time (ms)" if domain == "time" else "Depth (m)"
        config = PlotConfig.for_seismic(k_unit=k_label)
        config = config.update(cmap="seismic", show_colorbar=True)

        # Get symmetric color limits
        vmax = np.percentile(np.abs(full_stack), 99)
        vmin = -vmax
        config = config.update(extra_kwargs={"vmin": vmin, "vmax": vmax})

        # Inline slice
        inline_data, xlabel, ylabel = extractor.extract_inline(full_stack, mid_i)
        config_inline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Inline {mid_i}",
        )
        ImageRenderer.render(axes[0], inline_data, config_inline)

        # Crossline slice
        crossline_data, xlabel, ylabel = extractor.extract_crossline(full_stack, mid_j)
        config_crossline = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"Crossline {mid_j}",
        )
        ImageRenderer.render(axes[1], crossline_data, config_crossline)

        # Depth/Time slice
        depth_data, xlabel, ylabel = extractor.extract_depthslice(full_stack, mid_k)
        slice_label = f"{k_label.split()[0]} Slice at {mid_k}{k_label.split()[1][1:-1]}"
        config_depth = config.update(
            xlabel=xlabel,
            ylabel=ylabel,
            title=slice_label,
        )
        ImageRenderer.render(axes[2], depth_data, config_depth)

        plt.tight_layout()

        # Save
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
        """Generate all seismogram plots from cache file.

        Parameters
        ----------
        cache_file : Path
            Path to NPZ cache file containing seismogram data
        output_dir : Path
            Output directory for PNG files
        domain : str
            Domain type: "time" or "depth", default: time
        angles : Optional[list[float]]
            List of angles (if None, will be inferred from cache)

        Returns
        -------
        dict[str, list[Path]]
            Dictionary with 'angle_stacks' and 'full_stack' file paths
        """
        logger.info(f"Loading seismogram data from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)

        # Determine domain suffix for filenames
        suffix = "_depth" if domain == "depth" else ""

        generated_files = {"angle_stacks": [], "full_stack": []}

        # Plot angle stacks
        if "angle_stacks" in data or any(k.startswith("angle_") for k in data.keys()):
            logger.info(f"Generating angle stack plots ({domain} domain)...")

            # Try to load angle stacks
            if "angle_stacks" in data:
                angle_stacks = data["angle_stacks"]
                if angles is None:
                    angles = data.get("angles", list(range(len(angle_stacks))))
            else:
                # Load individual angle arrays
                angle_keys = sorted([k for k in data.keys() if k.startswith("angle_")])
                angle_stacks = [data[k] for k in angle_keys]
                if angles is None:
                    angles = [int(k.split("_")[1]) for k in angle_keys]

            # Plot each angle
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
