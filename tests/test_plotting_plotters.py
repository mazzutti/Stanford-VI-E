"""Tests for plotting plotter render methods.

Tests verify SlicePlotter, OverlayPlotter, and RockPhysicsPlotter rendering.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from src.plotting.slice_plotter import SlicePlotter
from src.plotting.overlay_plotter import OverlayPlotter
from src.plotting.rock_physics_plotter import RockPhysicsPlotter
from src.plotting.helpers.config import PlotConfig


class TestSlicePlotterRendering:
    """Test SlicePlotter rendering methods."""

    @pytest.fixture
    def plotter(self):
        """Create a SlicePlotter instance."""
        return SlicePlotter(backend="Agg")

    @pytest.fixture
    def test_cube(self):
        """Create test 3D data."""
        return np.random.randn(10, 15, 20)

    def test_plot_2d_slices_returns_tuple(self, plotter, test_cube):
        """Test that plot_2d_slices returns (image, colorbar) tuple."""
        fig, ax = plt.subplots()
        config = PlotConfig.for_seismic()

        result = plotter.plot_2d_slices(ax, test_cube, (5, 7, 10), config)

        assert isinstance(result, tuple)
        assert len(result) == 2
        im, cbar = result
        assert im is not None
        plt.close(fig)

    def test_plot_2d_slices_without_config(self, plotter, test_cube):
        """Test plot_2d_slices uses default config when None."""
        fig, ax = plt.subplots()

        im, cbar = plotter.plot_2d_slices(ax, test_cube, (5, 7, 10))

        assert im is not None
        assert cbar is not None
        plt.close(fig)

    def test_plot_crossline(self, plotter, test_cube):
        """Test plotting a crossline slice."""
        fig, ax = plt.subplots()
        config = PlotConfig.for_seismic()

        im, cbar = plotter.plot_crossline(ax, test_cube, (5, 7, 10), config)

        assert im is not None
        assert ax.get_xlabel() == "Inline (I)"
        plt.close(fig)

    def test_plot_depthslice(self, plotter, test_cube):
        """Test plotting a depth slice."""
        fig, ax = plt.subplots()
        config = PlotConfig.for_seismic()

        im, cbar = plotter.plot_depthslice(ax, test_cube, (5, 7, 10), config)

        assert im is not None
        assert ax.get_xlabel() == "Inline (I)"
        plt.close(fig)

    def test_plot_3d_slices_returns_axis(self, plotter, test_cube):
        """Test that plot_3d_slices returns 3D axis."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        config = PlotConfig.for_seismic()

        result = plotter.plot_3d_slices(ax, test_cube, (5, 7, 10), config)

        assert result is not None
        plt.close(fig)

    def test_plot_2d_slices_with_title(self, plotter, test_cube):
        """Test that config title is applied."""
        fig, ax = plt.subplots()
        config = PlotConfig(title="Test 2D Slice")

        plotter.plot_2d_slices(ax, test_cube, (5, 7, 10), config)

        assert "Test 2D Slice" in ax.get_title() or ax.get_title() == "Test 2D Slice"
        plt.close(fig)

    def test_plot_different_slice_types(self, plotter, test_cube):
        """Test all slice orientation types."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        config = PlotConfig.for_seismic()

        # Inline
        im1, _ = plotter.plot_2d_slices(axes[0], test_cube, (5, 7, 10), config)

        # Crossline
        im2, _ = plotter.plot_crossline(axes[1], test_cube, (5, 7, 10), config)

        # Depth slice
        im3, _ = plotter.plot_depthslice(axes[2], test_cube, (5, 7, 10), config)

        assert im1 is not None
        assert im2 is not None
        assert im3 is not None
        plt.close(fig)


class TestOverlayPlotterRendering:
    """Test OverlayPlotter rendering methods."""

    @pytest.fixture
    def plotter(self):
        """Create an OverlayPlotter instance."""
        return OverlayPlotter(backend="Agg")

    @pytest.fixture
    def seismic_slice(self):
        """Create test seismic slice."""
        return np.random.randn(15, 20)

    @pytest.fixture
    def facies_slice(self):
        """Create test facies slice."""
        return np.random.randint(0, 4, (15, 20))

    def test_detect_facies_boundaries(self, plotter, facies_slice):
        """Test facies boundary detection."""
        boundaries = plotter.detect_facies_boundaries(facies_slice)

        assert boundaries.shape == facies_slice.shape
        assert boundaries.dtype == bool
        # Some boundaries should be detected
        assert boundaries.sum() > 0

    def test_detect_boundaries_with_sigma(self, plotter, facies_slice):
        """Test boundary detection with custom sigma."""
        boundaries1 = plotter.detect_facies_boundaries(facies_slice, sigma=0.5)
        boundaries2 = plotter.detect_facies_boundaries(facies_slice, sigma=2.0)

        # Different sigmas may produce different results
        assert boundaries1.shape == boundaries2.shape

    def test_detect_boundaries_with_threshold(self, plotter, facies_slice):
        """Test boundary detection with custom threshold."""
        boundaries1 = plotter.detect_facies_boundaries(facies_slice, threshold=0.1)
        boundaries2 = plotter.detect_facies_boundaries(facies_slice, threshold=0.5)

        # Higher threshold means fewer boundaries
        assert boundaries2.sum() <= boundaries1.sum()

    def test_plot_seismic_with_facies_overlay(
        self, plotter, seismic_slice, facies_slice
    ):
        """Test plotting seismic with facies overlay."""
        fig, ax = plt.subplots()
        config = PlotConfig.for_seismic()

        im, cbar = plotter.plot_seismic_with_facies_overlay(
            ax, seismic_slice, facies_slice, config
        )

        assert im is not None
        assert cbar is not None
        # Should have contours overlaid
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_plot_seismic_with_facies_without_config(
        self, plotter, seismic_slice, facies_slice
    ):
        """Test overlay uses default config when None."""
        fig, ax = plt.subplots()

        im, cbar = plotter.plot_seismic_with_facies_overlay(
            ax, seismic_slice, facies_slice
        )

        assert im is not None
        plt.close(fig)

    def test_plot_facies_only(self, plotter, facies_slice):
        """Test plotting facies data only."""
        fig, ax = plt.subplots()
        config = PlotConfig.for_categorical()

        im, cbar = plotter.plot_facies_only(ax, facies_slice, config)

        assert im is not None
        assert cbar is not None
        plt.close(fig)

    def test_plot_facies_only_without_config(self, plotter, facies_slice):
        """Test facies plotting uses categorical config when None."""
        fig, ax = plt.subplots()

        im, cbar = plotter.plot_facies_only(ax, facies_slice)

        assert im is not None
        plt.close(fig)


class TestRockPhysicsPlotterRendering:
    """Test RockPhysicsPlotter rendering methods."""

    @pytest.fixture
    def plotter(self):
        """Create a RockPhysicsPlotter instance."""
        return RockPhysicsPlotter(backend="Agg")

    @pytest.fixture
    def test_data(self):
        """Create test 3D attribute data."""
        return np.random.randn(8, 12, 15) * 100 + 1000

    def test_plot_attribute_inline(self, plotter, test_data):
        """Test plotting inline slice of attribute."""
        fig, ax = plt.subplots()

        im, cbar = plotter.plot_attribute(ax, test_data, idx=3, slice_type="inline")

        assert im is not None
        assert cbar is not None
        plt.close(fig)

    def test_plot_attribute_crossline(self, plotter, test_data):
        """Test plotting crossline slice of attribute."""
        fig, ax = plt.subplots()

        im, cbar = plotter.plot_attribute(ax, test_data, idx=5, slice_type="crossline")

        assert im is not None
        plt.close(fig)

    def test_plot_attribute_depthslice(self, plotter, test_data):
        """Test plotting depth slice of attribute."""
        fig, ax = plt.subplots()

        im, cbar = plotter.plot_attribute(ax, test_data, idx=7, slice_type="depthslice")

        assert im is not None
        plt.close(fig)

    def test_plot_attribute_without_config(self, plotter, test_data):
        """Test attribute plotting uses default config when None."""
        fig, ax = plt.subplots()

        im, cbar = plotter.plot_attribute(ax, test_data, idx=3, slice_type="inline")

        assert im is not None
        plt.close(fig)

    def test_plot_attribute_with_custom_config(self, plotter, test_data):
        """Test attribute plotting with custom config."""
        fig, ax = plt.subplots()
        config = PlotConfig.for_attributes("Vp", cmap="plasma")

        im, cbar = plotter.plot_attribute(
            ax, test_data, idx=3, slice_type="inline", config=config
        )

        assert im is not None
        plt.close(fig)

    def test_plot_multiple_attributes(self, plotter):
        """Test plotting multiple attributes in grid."""
        attributes = {
            "Vp": np.random.randn(8, 12, 15) + 5000,
            "Vs": np.random.randn(8, 12, 15) + 2800,
            "Rho": np.random.randn(8, 12, 15) + 2300,
        }

        fig = plt.figure(figsize=(12, 4))

        plotter.plot_multiple_attributes(fig, attributes, idx=5)

        # Should have created subplots for each attribute
        assert len(fig.get_axes()) >= 3
        plt.close(fig)

    def test_plot_multiple_attributes_with_slice_type(self, plotter):
        """Test multiple attributes with different slice types."""
        attributes = {
            "Vp": np.random.randn(8, 12, 15),
            "Vs": np.random.randn(8, 12, 15),
        }

        for slice_type in ["inline", "crossline", "depthslice"]:
            fig = plt.figure()

            plotter.plot_multiple_attributes(
                fig, attributes, idx=2, slice_type=slice_type
            )

            plt.close(fig)

    def test_plot_multiple_attributes_with_cmap(self, plotter):
        """Test multiple attributes with custom colormap."""
        attributes = {
            "Vp": np.random.randn(8, 12, 15),
            "Vs": np.random.randn(8, 12, 15),
        }

        fig = plt.figure()

        plotter.plot_multiple_attributes(fig, attributes, idx=2, cmap="viridis")

        plt.close(fig)


class TestPlotterIntegration:
    """Integration tests for plotter workflows."""

    def test_slice_plotter_complete_workflow(self):
        """Test complete SlicePlotter workflow."""
        plotter = SlicePlotter(backend="Agg")
        cube = np.random.randn(10, 15, 20)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        config = PlotConfig.for_seismic(k_unit="ms")

        # Plot different slices
        plotter.plot_2d_slices(axes[0, 0], cube, (5, 7, 10), config)
        plotter.plot_crossline(axes[0, 1], cube, (5, 7, 10), config)
        plotter.plot_depthslice(axes[1, 0], cube, (5, 7, 10), config)

        # 3D plot

        ax3d = fig.add_subplot(2, 2, 4, projection="3d")
        plotter.plot_3d_slices(ax3d, cube, (5, 7, 10), config)

        plt.close(fig)

    def test_overlay_plotter_complete_workflow(self):
        """Test complete OverlayPlotter workflow."""
        plotter = OverlayPlotter(backend="Agg")
        seismic = np.random.randn(15, 20)
        facies = np.random.randint(0, 4, (15, 20))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        config = PlotConfig.for_seismic()

        # Overlay plot
        plotter.plot_seismic_with_facies_overlay(axes[0], seismic, facies, config)

        # Facies only
        plotter.plot_facies_only(axes[1], facies)

        plt.close(fig)

    def test_rock_physics_plotter_complete_workflow(self):
        """Test complete RockPhysicsPlotter workflow."""
        plotter = RockPhysicsPlotter(backend="Agg")

        attributes = {
            "Vp": np.random.randn(8, 12, 15) + 5000,
            "Vs": np.random.randn(8, 12, 15) + 2800,
            "Rho": np.random.randn(8, 12, 15) + 2300,
            "Lam": np.random.randn(8, 12, 15) + 3000,
        }

        fig = plt.figure(figsize=(16, 4))
        plotter.plot_multiple_attributes(fig, attributes, idx=5)

        plt.close(fig)
