"""Facies overlay plotting.

Provides OverlayPlotter class for visualizing seismic data with facies overlays.
Uses PlotConfig and ImageRenderer for clean, modern design.
"""

import logging
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, sobel

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.components import ImageRenderer
from src.plotting.helpers.config import PlotConfig

logger = logging.getLogger(__name__)

class OverlayPlotter(BasePlotter):
    """Plotter for seismic data with facies boundary overlays.

    Detects facies boundaries and overlays them on seismic data.
    Uses PlotConfig for configuration and ImageRenderer for rendering.
    """

    def detect_facies_boundaries(
        self,
        facies_slice: NDArray[Any],
        sigma: float = 0.5,
        threshold: float = 0.1,
    ) -> NDArray[np.bool_]:
        """Detect facies boundaries using gradient analysis.

        Args:
            facies_slice: 2D facies array
            sigma: Gaussian filter standard deviation
            threshold: Gradient magnitude threshold for boundary detection

        Returns:
            Binary array marking boundaries
        """
        # Cast intermediate results to Any to reduce third-party typing noise
        smoothed = cast(Any, gaussian_filter(facies_slice.astype(float), sigma=sigma))
        grad_x = cast(Any, sobel(smoothed, axis=0))
        grad_y = cast(Any, sobel(smoothed, axis=1))
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        boundaries = cast(NDArray[np.bool_], gradient_magnitude > threshold)

        self._log_debug(
            f"detected boundaries: {boundaries.sum()} pixels above threshold {threshold}"
        )

        return boundaries

    def plot_seismic_with_facies_overlay(
        self,
        ax: Axes,
        seismic_slice: NDArray[np.floating[Any]],
        facies_slice: NDArray[np.floating[Any]],
        config: PlotConfig | None = None,
    ) -> tuple[AxesImage, Colorbar | None]:
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

        # Plot seismic base image
        config = config.update(
            xlabel="Crossline (J)",
            colorbar_label="Amplitude",
        )
        im, cbar = ImageRenderer.render(ax, seismic_slice, config)

        # Detect and overlay facies boundaries
        boundaries = self.detect_facies_boundaries(facies_slice)

        nj_facies, nk_facies = facies_slice.shape
        # meshgrid and coordinate arrays can produce partially-unknown types
        # Use explicit dtypes to give Pyright concrete array types
        j_coords: NDArray[Any] = np.arange(nj_facies, dtype=float)
        k_coords: NDArray[Any] = np.arange(nk_facies, dtype=float) * float(
            config.k_scale
        )
        jj, kk = np.meshgrid(j_coords, k_coords, indexing="ij")
        # Convert meshgrid outputs to concrete ndarrays
        jj = np.asarray(jj)
        kk = np.asarray(kk)

        # Overlay boundary contours
        # Overlay boundaries and facies contours via helper to reduce locals
        self._overlay_boundaries(ax, boundaries, facies_slice, config)

        self._log_debug(
            f"plotted overlay: seismic shape={seismic_slice.shape}, "
            f"facies shape={facies_slice.shape}"
        )

        return im, cbar

    def _overlay_boundaries(
        self,
        ax: Axes,
        boundaries: NDArray[np.bool_],
        facies_slice: NDArray[np.floating[Any]],
        config: PlotConfig,
    ) -> None:
        """Helper to overlay boundary contours and facies transition lines.

        Extracted from `plot_seismic_with_facies_overlay` to reduce local
        variable counts and improve readability.
        """
        nj_facies, nk_facies = facies_slice.shape

        j_coords: NDArray[Any] = np.arange(nj_facies, dtype=float)
        k_coords: NDArray[Any] = np.arange(nk_facies, dtype=float) * float(
            config.k_scale
        )
        jj, kk = np.meshgrid(j_coords, k_coords, indexing="ij")
        jj = np.asarray(jj)
        kk = np.asarray(kk)

        # Boundary contour
        ax.contour(
            jj,
            kk,
            boundaries,
            levels=np.asarray([0.5]),
            colors="lime",
            linewidths=1.5,
            linestyles="solid",
            alpha=0.8,
        )

        # Facies transition dashed lines
        ax.contour(
            jj,
            kk,
            facies_slice,
            levels=np.asarray([0.5, 1.5, 2.5]),
            colors="white",
            linewidths=0.5,
            linestyles="dashed",
            alpha=0.5,
        )

    def plot_facies_only(
        self,
        ax: Axes,
        facies_slice: NDArray[np.floating[Any]],
        config: PlotConfig | None = None,
        category_labels: dict[int, str] | None = None,
    ) -> tuple[AxesImage, Colorbar | None]:
        """Plot facies data only with discrete colormap.

        Args:
            ax: Matplotlib axis
            facies_slice: 2D facies array
            config: PlotConfig for styling (uses categorical default if None)

        Returns:
            Tuple of (image, colorbar)
        """
        config = config or PlotConfig.for_categorical()

        # Allow callers to provide human-friendly category labels
        if category_labels:
            # PlotConfig.update expects JSON-serializable mapping types;
            # cast to Any so the typed update accepts the dict[int,str].
            config = config.update(category_labels=cast(Any, category_labels))

        config = config.update(
            xlabel="Crossline (J)",
            colorbar_label="Facies",
        )

        im, cbar = ImageRenderer.render(ax, facies_slice, config)

        self._log_debug(
            f"plotted facies only: shape={facies_slice.shape}, unique={np.unique(facies_slice)}"
        )

        return im, cbar
