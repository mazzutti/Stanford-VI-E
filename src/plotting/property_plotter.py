"""OOP-based property plotting framework.

Provides base class and specialized plotters for different property types
using template method pattern and inheritance to eliminate code duplication.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.plotting.plotly_plotter import PlotlyPlotter

logger = logging.getLogger(__name__)


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
        logger.info(f"Output directory: {self.output_dir}")

    @abstractmethod
    def load_data(self) -> None:
        """Load property data from source.

        This method should populate the internal data structures
        needed for plotting. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_properties(self) -> Dict[str, Dict[str, Any]]:
        """Get property metadata dictionary.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping property keys to metadata dicts with:
            - name: str - Display name
            - units: str - Physical units
            - cmap: str - Matplotlib colormap name
            - data: NDArray - 3D data array

        Must be implemented by subclasses.
        """
        pass

    def plot_3d_slices(
        self,
        data: NDArray[np.floating[Any]],
        output_path: Path,
        title: str,
        units: str,
        cmap: str = "viridis",
        dpi: int = 300,
    ) -> Path:
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
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Get middle indices for slicing
        ni, nj, nk = data.shape
        mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2

        # Get data range for consistent colorbar (2nd and 98th percentile)
        vmin, vmax = np.percentile(data, [2, 98])

        # Create colorbar label with units
        colorbar_label = f"{title}\n[{units}]"

        # Inline slice (constant i)
        im1 = axes[0].imshow(
            data[mid_i, :, :].T,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_title(f"Inline (i={mid_i})")
        axes[0].set_xlabel("Crossline (j)")
        axes[0].set_ylabel("Depth (k)")
        plt.colorbar(im1, ax=axes[0], label=colorbar_label)

        # Crossline slice (constant j)
        im2 = axes[1].imshow(
            data[:, mid_j, :].T,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title(f"Crossline (j={mid_j})")
        axes[1].set_xlabel("Inline (i)")
        axes[1].set_ylabel("Depth (k)")
        plt.colorbar(im2, ax=axes[1], label=colorbar_label)

        # Depth slice (constant k)
        im3 = axes[2].imshow(
            data[:, :, mid_k].T,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[2].set_title(f"Depth Slice (k={mid_k})")
        axes[2].set_xlabel("Inline (i)")
        axes[2].set_ylabel("Crossline (j)")
        plt.colorbar(im3, ax=axes[2], label=colorbar_label)

        plt.tight_layout()

        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Saved 3D slice plot: {output_path}")

        return output_path

    def generate_all_plots(self, file_prefix: str = "property") -> List[str]:
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
        List[str]
            List of generated file paths
        """
        # Step 1: Load data
        logger.info("Loading property data...")
        self.load_data()

        # Step 2: Get properties
        properties = self.get_properties()
        logger.info(f"Found {len(properties)} properties to plot")

        # Step 3: Generate plots
        generated_files = []
        logger.info("Generating individual property plots...")

        for prop_key, prop_info in properties.items():
            data = prop_info.get("data")
            if data is None:
                logger.warning(f"Property '{prop_key}' has no data, skipping")
                continue

            try:
                prop_name = prop_info["name"]
                prop_units = prop_info["units"]
                prop_cmap = prop_info.get("cmap", "viridis")

                logger.info(f"Plotting {prop_name} with shape {data.shape}")

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
                logger.info(f"  ✓ Generated: {output_file}")

            except Exception as e:
                logger.error(f"Failed to plot {prop_key}: {e}")
                continue

        logger.info(f"✓ Generated {len(generated_files)} property plots")

        return generated_files


class RockPhysicsPropertyPlotter(PropertyPlotter):
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
        self.data = None

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

        logger.info(f"Loading data from: {cache_file}")
        self.data = np.load(cache_file, allow_pickle=True)
        logger.info(f"Available attributes: {list(self.data.keys())}")

    def get_properties(self) -> Dict[str, Dict[str, Any]]:
        """Get rock physics attributes metadata.

        Returns
        -------
        Dict[str, Dict[str, Any]]
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
        properties = {}
        for key, metadata in attributes.items():
            if key in self.data:
                properties[key] = {
                    **metadata,
                    "data": self.data[key],
                    "cmap": "viridis",
                }

        return properties

    def generate_comparison_plot(self) -> Optional[str]:
        """Generate multi-attribute comparison plot.

        Returns
        -------
        Optional[str]
            Path to generated comparison plot, or None if failed
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        from src.plotting.rock_physics_plotter import RockPhysicsPlotter

        properties = self.get_properties()
        attr_data_dict = {k: self.data[k] for k in properties.keys() if k in self.data}

        if not attr_data_dict:
            logger.warning("No attributes available for comparison plot")
            return None

        try:
            logger.info("Generating multi-attribute comparison plot...")

            # Create figure for grid of attributes
            fig = plt.figure(figsize=(16, 10))

            # Get middle inline index
            first_data = next(iter(attr_data_dict.values()))
            mid_i = first_data.shape[0] // 2

            # Create a new dictionary with names that include units
            attr_data_dict_with_units = {
                f"{properties[k]['name']}\n[{properties[k]['units']}]": self.data[k]
                for k in properties.keys()
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
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"  ✓ Generated: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Failed to generate comparison plot: {e}")
            return None

    def generate_3d_plotly_visualizations(self) -> List[str]:
        """Generate 3D interactive Plotly visualizations for rock physics attributes.

        Returns
        -------
        List[str]
            List of generated HTML file paths
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error(
                "Plotly is required for 3D plots. Install with: pip install plotly"
            )
            return []

        logger.info(
            "Generating 3D interactive Plotly visualizations for rock physics attributes..."
        )

        properties = self.get_properties()
        generated_files = []

        for prop_key, prop_info in properties.items():
            data = prop_info["data"]
            if data is None:
                logger.warning(f"Property '{prop_key}' not loaded, skipping")
                continue

            logger.info(
                f"Creating 3D plot for {prop_info['name']} with shape {data.shape}"
            )

            ni, nj, nk = data.shape
            mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2

            # Create figure with 3D orthogonal slices
            fig = go.Figure()

            # Create coordinate arrays for the grid
            i_coords = np.arange(ni)
            j_coords = np.arange(nj)
            k_coords = np.arange(nk)[::-1]  # Reverse to make depth increase downward

            # Inline slice (constant i=mid_i) - YZ plane at x=mid_i
            J_yz, K_yz = np.meshgrid(j_coords, k_coords, indexing="ij")

            fig.add_trace(
                go.Surface(
                    x=np.full_like(J_yz, mid_i),
                    y=J_yz,
                    z=K_yz,
                    surfacecolor=data[mid_i, :, ::-1],  # Flip along k dimension
                    colorscale=prop_info["cmap"],
                    name=f"Inline (i={mid_i})",
                    showscale=True,
                    colorbar=dict(
                        title=prop_info["units"],
                        x=1.02,
                        len=0.75,
                    ),
                )
            )

            # Crossline slice (constant j=mid_j) - XZ plane at y=mid_j
            I_xz, K_xz = np.meshgrid(i_coords, k_coords, indexing="ij")

            fig.add_trace(
                go.Surface(
                    x=I_xz,
                    y=np.full_like(I_xz, mid_j),
                    z=K_xz,
                    surfacecolor=data[:, mid_j, ::-1],  # Flip along k dimension
                    colorscale=prop_info["cmap"],
                    name=f"Crossline (j={mid_j})",
                    showscale=False,
                )
            )

            # Depth slice (constant k=mid_k) - XY plane at z=mid_k
            I_xy, J_xy = np.meshgrid(i_coords, j_coords, indexing="ij")

            fig.add_trace(
                go.Surface(
                    x=I_xy,
                    y=J_xy,
                    z=np.full_like(I_xy, k_coords[mid_k]),
                    surfacecolor=data[:, :, mid_k],
                    colorscale=prop_info["cmap"],
                    name=f"Depth Slice (k={mid_k})",
                    showscale=False,
                )
            )

            fig.update_layout(
                template=None,
                title=dict(
                    text=f"Rock Physics: {prop_info['name']}",
                    x=0.5,
                    xanchor="center",
                ),
                scene=dict(
                    xaxis=dict(
                        title="Inline (i)",
                    ),
                    yaxis=dict(
                        title="Crossline (j)",
                    ),
                    zaxis=dict(
                        title="Depth (k)",
                        autorange="reversed",
                    ),
                    aspectmode="data",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.3),
                        center=dict(x=0, y=0, z=0),
                    ),
                ),
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            # Save HTML file
            output_file = self.output_dir / f"rock_physics_{prop_key}_3d.html"
            fig.write_html(
                str(output_file),
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )

            # Inject CSS and JavaScript for fullscreen display using centralized PlotlyPlotter method
            PlotlyPlotter.inject_3d_interaction_script(str(output_file))

            generated_files.append(str(output_file))
            logger.info(f"  ✓ Saved: {output_file}")

        return generated_files


class OriginalPropertyPlotter(PropertyPlotter):
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
        self.manager = None

    def load_data(self) -> None:
        """Load original properties from GSLIB files."""
        from src.io.loader import DatasetManager
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

    def get_properties(self) -> Dict[str, Dict[str, Any]]:
        """Get original properties metadata.

        Returns
        -------
        Dict[str, Dict[str, Any]]
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

    def generate_3d_plotly_visualizations(self) -> List[str]:
        """Generate 3D interactive Plotly visualizations.

        Returns
        -------
        List[str]
            List of generated HTML file paths
        """
        if self.manager is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error(
                "Plotly is required for 3D plots. Install with: pip install plotly"
            )
            return []

        logger.info("Generating 3D interactive Plotly visualizations...")

        properties = self.get_properties()
        generated_files = []

        for prop_key, prop_info in properties.items():
            data = prop_info["data"]
            if data is None:
                logger.warning(f"Property '{prop_key}' not loaded, skipping")
                continue

            logger.info(
                f"Creating 3D plot for {prop_info['name']} with shape {data.shape}"
            )

            ni, nj, nk = data.shape
            mid_i, mid_j, mid_k = ni // 2, nj // 2, nk // 2

            # Create figure with 3D orthogonal slices
            fig = go.Figure()

            # Create coordinate arrays for the grid
            i_coords = np.arange(ni)
            j_coords = np.arange(nj)
            k_coords = np.arange(nk)[::-1]  # Reverse to make depth increase downward

            # Inline slice (constant i=mid_i) - YZ plane at x=mid_i
            J_yz, K_yz = np.meshgrid(j_coords, k_coords, indexing="ij")

            fig.add_trace(
                go.Surface(
                    x=np.full_like(J_yz, mid_i),
                    y=J_yz,
                    z=K_yz,
                    surfacecolor=data[mid_i, :, ::-1],  # Flip along k dimension
                    colorscale=prop_info["cmap"],
                    name=f"Inline (i={mid_i})",
                    showscale=True,
                    colorbar=dict(
                        title=prop_info["units"],
                        x=1.02,
                        len=0.75,
                    ),
                )
            )

            # Crossline slice (constant j=mid_j) - XZ plane at y=mid_j
            I_xz, K_xz = np.meshgrid(i_coords, k_coords, indexing="ij")

            fig.add_trace(
                go.Surface(
                    x=I_xz,
                    y=np.full_like(I_xz, mid_j),
                    z=K_xz,
                    surfacecolor=data[:, mid_j, ::-1],  # Flip along k dimension
                    colorscale=prop_info["cmap"],
                    name=f"Crossline (j={mid_j})",
                    showscale=False,
                )
            )

            # Depth slice (constant k=mid_k) - XY plane at z=mid_k
            I_xy, J_xy = np.meshgrid(i_coords, j_coords, indexing="ij")

            fig.add_trace(
                go.Surface(
                    x=I_xy,
                    y=J_xy,
                    z=np.full_like(I_xy, k_coords[mid_k]),
                    surfacecolor=data[:, :, mid_k],
                    colorscale=prop_info["cmap"],
                    name=f"Depth Slice (k={mid_k})",
                    showscale=False,
                )
            )

            fig.update_layout(
                template=None,
                title=dict(
                    text=f"Original Data: {prop_info['name']}",
                    x=0.5,
                    xanchor="center",
                ),
                scene=dict(
                    xaxis=dict(
                        title="Inline (i)",
                    ),
                    yaxis=dict(
                        title="Crossline (j)",
                    ),
                    zaxis=dict(
                        title="Depth (k)",
                        autorange="reversed",
                    ),
                    aspectmode="data",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.3),
                        center=dict(x=0, y=0, z=0),
                    ),
                ),
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            # Save HTML file
            output_file = self.output_dir / f"original_{prop_key}_3d.html"
            fig.write_html(
                str(output_file),
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )

            # Inject CSS and JavaScript for fullscreen display using centralized PlotlyPlotter method
            PlotlyPlotter.inject_3d_interaction_script(str(output_file))

            generated_files.append(str(output_file))
            logger.info(f"  ✓ Saved: {output_file}")

        return generated_files
