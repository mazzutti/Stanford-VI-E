"""Plotting package with clean OOP interface.

Provides plotters for all visualization tasks:
- SlicePlotter: 2D and 3D orthogonal slices
- OverlayPlotter: Seismic with facies overlay
- FaciesPlotter: Facies correlation summary figures
- RockPhysicsPlotter: Rock physics attributes
- SeismogramPlotter: Seismogram visualizations (time and depth)
- PlotlyPlotter: Interactive 3D visualizations

Plus composition-based components for maximum flexibility:
- PlotConfig: Type-safe configuration for plots
- SliceExtractor: Extract 2D slices from 3D cubes
- DataNormalizer: Compute data limits and colormaps
- AxisStyler: Apply consistent axis styling
- ImageRenderer: Render images with colorbars

All plotters inherit from BasePlotter and share common utilities through composition.

Example:
    from src.plotting import SlicePlotter
    from src.plotting.helpers import PlotConfig

    plotter = SlicePlotter()
    config = PlotConfig.for_seismic(k_unit="ms")
    plotter.plot_2d_slices(ax, cube, (50, 50, 50), config)

Or use components directly:
    from src.plotting.helpers import ImageRenderer, PlotConfig

    config = PlotConfig.for_categorical()
    im, cbar = ImageRenderer.render(ax, facies_data, config)
"""

import logging

from src.plotting.facies_plotter import FaciesPlotter

# Import helper components
from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.components import (
    AxisStyler,
    DataNormalizer,
    ImageRenderer,
    SliceExtractor,
)

# Initialize matplotlib and numpy
from src.plotting.helpers.config import PlotConfig, init_plotting
from src.plotting.overlay_plotter import OverlayPlotter
from src.plotting.plotly_plotter import PlotlyPlotter
from src.plotting.rock_physics_plotter import RockPhysicsPlotter
from src.plotting.seismic_plotter import SeismicPlotter

# Import main plotter classes
from src.plotting.slice_plotter import SlicePlotter

plt, np = init_plotting(backend="qtagg")

# Public API
__all__ = [
    # Core utilities
    "plt",
    "np",
    "BasePlotter",
    # Configuration and components
    "PlotConfig",
    "SliceExtractor",
    "DataNormalizer",
    "AxisStyler",
    "ImageRenderer",
    # Plotters
    "SlicePlotter",
    "OverlayPlotter",
    "FaciesPlotter",
    "RockPhysicsPlotter",
    "SeismicPlotter",
    "PlotlyPlotter",
]

logger = logging.getLogger(__name__)
