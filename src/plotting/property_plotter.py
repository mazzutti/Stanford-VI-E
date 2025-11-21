"""OOP-based property plotting framework.

Provides base class and specialized plotters for different property types
using template method pattern and inheritance to eliminate code duplication.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from src.io.loader import DatasetManager
from src.plotting.mixins import ImshowWithColorbarMixin
from src.plotting.plotly_plotter import PlotlyPlotter

logger = logging.getLogger(__name__)

# Some plotting helpers import heavy plotting backends and helpers inside
# methods to avoid import-time costs and optional dependencies. Prefer
# adding per-import suppression at the call site rather than a module-level
# disable.


class PropertyPlotter(ABC):
    """Abstract base class for property plotting.

    This class implements the template method pattern for property
    visualization workflows. Subclasses implement data loading and
    property metadata definition.

    Attributes
    ----------
    output_dir : Path
        Output directory for generated plots
    verbose : bool
        Enable verbose logging
    """

    def __init__(self, output_dir: str = "docs/images", verbose: bool = False):
        """Initialize the property plotter.

        Parameters
        ----------
        output_dir : str
            Output directory for plot files, default: docs/images
        verbose : bool
            Enable verbose logging, default: False
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", self.output_dir)

    @abstractmethod
    def load_data(self) -> None:
        """Load property data from source.

        This method should populate the internal data structures
        needed for plotting. Must be implemented by subclasses.
        """

    @abstractmethod
    def get_properties(self) -> dict[str, dict[str, Any]]:
        """Get property metadata dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping property keys to metadata dicts with:
            - name: str - Display name
            - units: str - Physical units
            - cmap: str - Matplotlib colormap name
            - data: NDArray - 3D data array

        Must be implemented by subclasses.
        """

    @abstractmethod
    def _compute_slice_indices_and_bounds(
        self, data: NDArray[np.floating[Any]], title: str, units: str
    ) -> tuple[int, int, int, float, float, str]:
        """Compute middle slice indices, percentile bounds and colorbar label.

        Subclasses must implement this to provide the middle inline/crossline/depth
        indices, the vmin/vmax bounds (as floats) and the colorbar label string.
        """
        raise NotImplementedError()

    @abstractmethod
    def _imshow_with_colorbar(
        self,
        ax: Axes,
        arr: NDArray[Any],
        title: str,
        xlabel: str,
        ylabel: str,
        cmap: str,
        vmin: float,
        vmax: float,
        colorbar_label: str,
    ) -> AxesImage:
        # Abstract plotting helpers necessarily take multiple configuration
        # parameters to support flexible rendering. Silence pylint's
        # argument-count noise for this abstract API, including positional
        # argument warnings which arise from flexible caller patterns.

        """Render an image on `ax` with labels and a colorbar.

        Subclasses should return the `AxesImage` created so callers can
        further manipulate the image if needed.
        """
        raise NotImplementedError()

    @abstractmethod
    def _make_plotly_traces(
        self, data: NDArray[np.floating[Any]], cmap: str, units: str
    ) -> tuple[list[go.Surface], int, int, int]:
        """Create Plotly Surface traces for inline, crossline and depth slices.

        Subclasses implement this to return a list of three `go.Surface` traces
        (inline, crossline, depth) along with the mid indices (i, j, k).
        """
        raise NotImplementedError()

    def plot_3d_slices(
        self,
        data: NDArray[np.floating[Any]],
        output_path: Path,
        title: str,
        units: str,
        cmap: str = "viridis",
        dpi: int = 300,
    ) -> Path:
        # Plotting glue necessarily carries many arguments for configuration.
        # Narrowly disable argument/locals checks for clarity and to avoid
        # noise from Pylint in this high-level orchestrator method, including
        # positional-argument checks which are common for plotting APIs.

        """Generate a 3-slice PNG plot (inline, crossline, depth) for 3D data.

        Parameters
        ----------
        data : NDArray[np.floating[Any]]
            3D data array with shape (ni, nj, nk)
        output_path : Path
            Output file path for the PNG image
        title : str
            Main title for the figure
        units : str
            Units for the colorbar label
        cmap : str
            Matplotlib colormap name, default: viridis
        dpi : int
            Resolution in dots per inch, default: 300

        Returns
        -------
        Path
            Path to the generated PNG file
        """
        # Create figure with 3 subplots (inline, crossline, depthslice)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig = cast(Figure, fig)
        axes = cast(Sequence[Axes], axes)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Compute indices, bounds and colorbar label via helper
        mid_i, mid_j, mid_k, vmin, vmax, colorbar_label = (
            self._compute_slice_indices_and_bounds(data, title, units)
        )

        # Inline slice (constant i)
        self._imshow_with_colorbar(
            axes[0],
            data[mid_i, :, :],
            title=f"Inline (i={mid_i})",
            xlabel="Crossline (j)",
            ylabel="Depth (k)",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
        )

        # Crossline slice (constant j)
        self._imshow_with_colorbar(
            axes[1],
            data[:, mid_j, :],
            title=f"Crossline (j={mid_j})",
            xlabel="Inline (i)",
            ylabel="Depth (k)",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
        )

        # Depth slice (constant k)
        self._imshow_with_colorbar(
            axes[2],
            data[:, :, mid_k],
            title=f"Depth Slice (k={mid_k})",
            xlabel="Inline (i)",
            ylabel="Crossline (j)",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
        )

        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.debug("Saved 3D slice plot: %s", output_path)

        return output_path

    def generate_all_plots(self, file_prefix: str = "property") -> list[str]:
        """Generate plots for all properties (template method).

        This method implements the common workflow:
        1. Load data
        2. Iterate through properties
        3. Generate plots
        4. Return list of generated files

        Parameters
        ----------
        file_prefix : str
            Prefix for output filenames, default: property

        Returns
        -------
        list[str]
            List of generated file paths
        """
        # Step 1: Load data
        logger.info("Loading property data...")
        self.load_data()

        # Step 2: Get properties
        properties: dict[str, dict[str, Any]] = self.get_properties()
        logger.info("Found %d properties to plot", len(properties))

        # Step 3: Generate plots
        generated_files: list[str] = []
        logger.info("Generating individual property plots...")

        for prop_key, prop_info in properties.items():
            data = prop_info.get("data")
            if data is None:
                logger.warning("Property '%s' has no data, skipping", prop_key)
                continue

            try:
                prop_name = prop_info["name"]
                prop_units = prop_info["units"]
                prop_cmap = prop_info.get("cmap", "viridis")

                logger.info("Plotting %s with shape %s", prop_name, data.shape)

                output_file = self.output_dir / f"{file_prefix}_{prop_key}.png"

                self.plot_3d_slices(
                    data=data,
                    output_path=output_file,
                    title=prop_name,
                    units=prop_units,
                    cmap=prop_cmap,
                    dpi=150,
                )

                generated_files.append(str(output_file))
                logger.info("  ✓ Generated: %s", output_file)

            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logger.error("Failed to plot %s: %s", prop_key, e)
                continue

        logger.info("✓ Generated %d property plots", len(generated_files))

        return generated_files


class RockPhysicsPropertyPlotter(ImshowWithColorbarMixin, PropertyPlotter):
    """Plotter for rock physics attributes.

    Loads attributes from NPZ cache files and generates visualizations
    for Lambda-Rho, Mu-Rho, AVO Intercept, and AVO Gradient.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        domain: str = "depth",
        output_dir: str = "docs/images",
        verbose: bool = False,
    ):
        """Initialize rock physics plotter.

        Parameters
        ----------
        cache_dir : str
            Cache directory path, default: .cache
        domain : str
            Domain for visualization (depth or time), default: depth
        output_dir : str
            Output directory for PNG files, default: docs/images
        verbose : bool
            Enable verbose logging, default: False
        """
        super().__init__(output_dir, verbose)
        self.cache_dir = Path(cache_dir)
        self.domain = domain
        self.data: Any = None

    def load_data(self) -> None:
        """Load rock physics attributes from NPZ cache file."""
        # Determine which cache file to use based on domain
        if self.domain == "time":
            cache_file = self.cache_dir / "rock_physics_attributes_time.npz"
        else:  # depth
            cache_file = self.cache_dir / "rock_physics_attributes.npz"

        if not cache_file.exists():
            error_msg = f"Cache file not found: {cache_file}"
            logger.error(error_msg)
            if self.domain == "time":
                logger.error(
                    "Hint: Run 'python resample_rock_physics_to_time.py' "
                    "to create time-domain attributes"
                )
            raise FileNotFoundError(error_msg)

        logger.info("Loading data from: %s", cache_file)
        self.data = np.load(cache_file, allow_pickle=True)
        # Guard against unexpected np.load return types (ndarray, NpzFile, None)
        if self.data is None:
            keys = []
        elif hasattr(self.data, "files"):
            try:
                keys = list(self.data.files)
            except (AttributeError, TypeError):
                keys = []
        elif hasattr(self.data, "keys"):
            try:
                keys = list(self.data.keys())
            except (AttributeError, TypeError):
                keys = []
        else:
            keys = []

        logger.info("Available attributes: %s", keys)

    def get_properties(self) -> dict[str, dict[str, Any]]:
        """Get rock physics attributes metadata.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary of rock physics attributes with metadata
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Define attributes with their metadata
        attributes = {
            "lambda_rho": {"name": "Lambda-Rho", "units": "(km/s)²·g/cm³"},
            "mu_rho": {"name": "Mu-Rho", "units": "(km/s)²·g/cm³"},
            "intercept": {"name": "AVO Intercept", "units": "dimensionless"},
            "gradient": {"name": "AVO Gradient", "units": "dimensionless"},
        }

        # Add data arrays and colormap to metadata
        properties: dict[str, dict[str, Any]] = {}
        for key, metadata in attributes.items():
            if key in self.data:
                properties[key] = {
                    **metadata,
                    "data": cast(NDArray[np.floating[Any]], self.data[key]),
                    "cmap": "viridis",
                }

        return properties

    def generate_comparison_plot(self) -> str | None:
        """Generate multi-attribute comparison plot.

        Returns
        -------
        str | None
            Path to generated comparison plot, or None if failed

        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # The importer is intentionally local to avoid heavy optional
        # dependencies during import-time. Silence pylint for this
        # intentional pattern at the function level.

        from src.plotting.rock_physics_plotter import (
            RockPhysicsPlotter,
        )

        properties = self.get_properties()
        attr_data_dict = {k: self.data[k] for k in properties if k in self.data}

        if not attr_data_dict:
            logger.warning("No attributes available for comparison plot")
            return None

        try:
            logger.info("Generating multi-attribute comparison plot...")

            # Create figure for grid of attributes and cast to Any immediately
            fig = plt.figure(figsize=(16, 10))

            # Get middle inline index
            first_data = next(iter(attr_data_dict.values()))
            mid_i = first_data.shape[0] // 2

            # Create a new dictionary with names that include units
            attr_data_dict_with_units = {
                f"{prop['name']}\n[{prop['units']}]": self.data[k]
                for k, prop in properties.items()
                if k in self.data
            }

            # Use RockPhysicsPlotter for the grid layout
            plotter = RockPhysicsPlotter(backend="Agg")
            plotter.plot_multiple_attributes(
                fig,
                attr_data_dict_with_units,
                idx=mid_i,
                slice_type="inline",
                cmap="viridis",
            )

            output_file = self.output_dir / "rock_physics_comparison.png"
            fig.savefig(str(output_file), dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.info("  ✓ Generated: %s", output_file)
            return str(output_file)

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error("Failed to generate comparison plot: %s", e)
            return None

    def generate_3d_plotly_visualizations(self) -> list[str]:
        """Generate 3D interactive Plotly visualizations for rock physics attributes.

        Returns
        -------
        list[str]
            List of generated HTML file paths
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Plotly is imported at module level

        logger.info(
            "Generating 3D interactive Plotly visualizations for rock physics attributes..."
        )

        properties = self.get_properties()
        generated_files: list[str] = []

        for prop_key, prop_info in properties.items():
            data = prop_info["data"]
            if data is None:
                logger.warning("Property '%s' not loaded, skipping", prop_key)
                continue

            logger.info(
                "Creating 3D plot for %s with shape %s", prop_info["name"], data.shape
            )

            # Create Plotly traces using shared helper
            fig: go.Figure = go.Figure()
            traces, _, _, _ = self._make_plotly_traces(
                data, prop_info["cmap"], prop_info["units"]
            )

            for t in traces:
                fig.add_trace(t)

            fig.update_layout(
                template=None,
                title={
                    "text": f"Rock Physics: {prop_info['name']}",
                    "x": 0.5,
                    "xanchor": "center",
                },
                scene={
                    "xaxis": {"title": "Inline (i)"},
                    "yaxis": {"title": "Crossline (j)"},
                    "zaxis": {"title": "Depth (k)", "autorange": "reversed"},
                    "aspectmode": "data",
                    "camera": {
                        "eye": {"x": 1.5, "y": 1.5, "z": 1.3},
                        "center": {"x": 0, "y": 0, "z": 0},
                    },
                },
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            # Save HTML file
            output_file = self.output_dir / f"rock_physics_{prop_key}_3d.html"
            fig.write_html(
                output_file,
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )

            # Inject optional CSS/JS for fullscreen display.
            # Failure is non-fatal; the injector logs issues.
            PlotlyPlotter.inject_3d_interaction_script(str(output_file))

            generated_files.append(str(output_file))
            logger.info("  ✓ Saved: %s", output_file)

        return generated_files


class OriginalPropertyPlotter(ImshowWithColorbarMixin, PropertyPlotter):
    """Plotter for original Stanford VI-E properties.

    Loads properties from GSLIB files and generates visualizations
    for P-wave velocity, S-wave velocity, and density.
    """

    def __init__(
        self,
        data_dir: str = ".",
        output_dir: str = "docs/images",
        verbose: bool = False,
    ):
        """Initialize original property plotter.

        Parameters
        ----------
        data_dir : str
            Root directory containing Stanford VI-E data folders, default: .
        output_dir : str
            Output directory for plot files, default: docs/images
        verbose : bool
            Enable verbose logging, default: False
        """
        super().__init__(output_dir, verbose)
        self.data_dir = data_dir
        # DatasetManager is imported lazily inside `load_data`; annotate
        # as Optional by forward reference so mypy understands assignments.
        self.manager: DatasetManager | None = None

    def load_data(self) -> None:
        """Load original properties from GSLIB files."""
        from src.io.grid import GridSpec

        logger.info("Loading original Stanford VI-E properties...")

        # Setup grid specification
        grid_spec = GridSpec(shape=(150, 200, 200), dz=1.0, dt=0.001)

        # File mapping for Stanford VI-E dataset
        file_map = {
            "vp": "P-wave Velocity",
            "vs": "S-wave Velocity",
            "rho": "Density",
        }

        # Load data using DatasetManager
        self.manager = DatasetManager.from_stanfordsix(
            data_path=self.data_dir,
            file_map=file_map,
            grid_spec=grid_spec,
        )

    def get_properties(self) -> dict[str, dict[str, Any]]:
        """Get original properties metadata.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary of original properties with metadata
        """
        if self.manager is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        return {
            "vp": {
                "name": "P-wave Velocity (Vp)",
                "data": self.manager.vp,
                "units": "km/s",
                "cmap": "viridis",
            },
            "vs": {
                "name": "S-wave Velocity (Vs)",
                "data": self.manager.vs,
                "units": "km/s",
                "cmap": "plasma",
            },
            "rho": {
                "name": "Density (ρ)",
                "data": self.manager.rho,
                "units": "g/cm³",
                "cmap": "RdYlBu_r",
            },
        }

    def generate_3d_plotly_visualizations(self) -> list[str]:
        """Generate 3D interactive Plotly visualizations.

        Returns
        -------
        list[str]
            List of generated HTML file paths
        """
        if self.manager is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Plotly is imported at module level

        logger.info("Generating 3D interactive Plotly visualizations...")

        properties = self.get_properties()
        generated_files: list[str] = []

        for prop_key, prop_info in properties.items():
            data = prop_info["data"]
            if data is None:
                logger.warning("Property '%s' not loaded, skipping", prop_key)
                continue

            logger.info(
                "Creating 3D plot for %s with shape %s", prop_info["name"], data.shape
            )

            # Create Plotly traces using shared helper
            fig: go.Figure = go.Figure()
            traces, _, _, _ = self._make_plotly_traces(
                data, prop_info["cmap"], prop_info["units"]
            )

            for t in traces:
                fig.add_trace(t)

            fig.update_layout(
                template=None,
                title={
                    "text": f"Original Data: {prop_info['name']}",
                    "x": 0.5,
                    "xanchor": "center",
                },
                scene={
                    "xaxis": {"title": "Inline (i)"},
                    "yaxis": {"title": "Crossline (j)"},
                    "zaxis": {"title": "Depth (k)", "autorange": "reversed"},
                    "aspectmode": "data",
                    "camera": {
                        "eye": {"x": 1.5, "y": 1.5, "z": 1.3},
                        "center": {"x": 0, "y": 0, "z": 0},
                    },
                },
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            # Save HTML file
            output_file = self.output_dir / f"original_{prop_key}_3d.html"
            fig.write_html(
                output_file,
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )

            # Inject optional CSS/JS for fullscreen display.
            # Failure is non-fatal; the injector logs issues.
            PlotlyPlotter.inject_3d_interaction_script(str(output_file))

            generated_files.append(str(output_file))
            logger.info("  ✓ Saved: %s", output_file)

        return generated_files
