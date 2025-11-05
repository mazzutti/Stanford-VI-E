"""Facies overlay plotting.

Provides OverlayPlotter class for visualizing seismic data with facies overlays.
Uses PlotConfig and ImageRenderer for clean, modern design.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.config import PlotConfig
from src.plotting.helpers.components import ImageRenderer

logger = logging.getLogger(__name__)


class OverlayPlotter(BasePlotter):
    """Plotter for seismic data with facies boundary overlays.

    Detects facies boundaries and overlays them on seismic data.
    Uses PlotConfig for configuration and ImageRenderer for rendering.
    """

    def detect_facies_boundaries(
        self, facies_slice: np.ndarray, sigma: float = 0.5, threshold: float = 0.1
    ) -> np.ndarray:
        """Detect facies boundaries using gradient analysis.

        Args:
            facies_slice: 2D facies array
            sigma: Gaussian filter standard deviation
            threshold: Gradient magnitude threshold for boundary detection

        Returns:
            Binary array marking boundaries
        """
        smoothed = gaussian_filter(facies_slice.astype(float), sigma=sigma)
        grad_x = sobel(smoothed, axis=0)
        grad_y = sobel(smoothed, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        boundaries = gradient_magnitude > threshold

        self._log_debug(
            f"detected boundaries: {boundaries.sum()} pixels above threshold {threshold}"
        )

        return boundaries

    def plot_seismic_with_facies_overlay(
        self,
        ax: Axes,
        seismic_slice: np.ndarray,
        facies_slice: np.ndarray,
        config: Optional[PlotConfig] = None,
    ) -> Tuple[AxesImage, Colorbar | None]:
        """Plot seismic with facies boundaries overlaid.

        Args:
            ax: Matplotlib axis
            seismic_slice: 2D seismic data
            facies_slice: 2D facies data
            config: PlotConfig for styling (uses seismic default if None)

        Returns:
            Tuple of (image, colorbar)
        """
        config = config or PlotConfig.for_seismic()

        nj, nk = seismic_slice.shape

        # Plot seismic base image
        config = config.update(
            xlabel="Crossline (J)",
            colorbar_label="Amplitude",
        )
        im, cbar = ImageRenderer.render(ax, seismic_slice, config)

        # Detect and overlay facies boundaries
        boundaries = self.detect_facies_boundaries(facies_slice)

        nj_facies, nk_facies = facies_slice.shape
        J = np.arange(nj_facies)
        K = np.arange(nk_facies) * config.k_scale
        JJ, KK = np.meshgrid(J, K, indexing="ij")

        # Overlay boundary contours
        ax.contour(
            JJ.T,
            KK.T,
            boundaries.T,
            levels=[0.5],
            colors="lime",
            linewidths=1.5,
            linestyles="solid",
            alpha=0.8,
        )

        # Overlay facies transitions as dashed lines
        facies_levels = [0.5, 1.5, 2.5]
        ax.contour(
            JJ.T,
            KK.T,
            facies_slice.T,
            levels=facies_levels,
            colors="white",
            linewidths=0.5,
            linestyles="dashed",
            alpha=0.5,
        )

        self._log_debug(
            f"plotted overlay: seismic shape={seismic_slice.shape}, "
            f"facies shape={facies_slice.shape}"
        )

        return im, cbar

    def plot_facies_only(
        self,
        ax: Axes,
        facies_slice: np.ndarray,
        config: Optional[PlotConfig] = None,
    ) -> Tuple[AxesImage, Colorbar | None]:
        """Plot facies data only with discrete colormap.

        Args:
            ax: Matplotlib axis
            facies_slice: 2D facies array
            config: PlotConfig for styling (uses categorical default if None)

        Returns:
            Tuple of (image, colorbar)
        """
        config = config or PlotConfig.for_categorical()

        config = config.update(
            xlabel="Crossline (J)",
            colorbar_label="Facies",
        )

        im, cbar = ImageRenderer.render(ax, facies_slice, config)

        self._log_debug(
            f"plotted facies only: shape={facies_slice.shape}, unique={np.unique(facies_slice)}"
        )

        return im, cbar
