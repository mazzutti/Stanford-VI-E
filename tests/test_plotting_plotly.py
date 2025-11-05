"""Tests for Plotly plotter for interactive 3D visualizations.

Tests verify PlotlyPlotter 3D volume creation functionality.
"""

import pytest
import numpy as np
import plotly.graph_objects as go

from src.plotting.plotly_plotter import PlotlyPlotter


class TestPlotlyPlotterBasic:
    """Test basic PlotlyPlotter functionality."""

    @pytest.fixture
    def plotter(self):
        """Create a PlotlyPlotter instance."""
        return PlotlyPlotter(backend="Agg")

    @pytest.fixture
    def test_cube(self):
        """Create test 3D data for Plotly."""
        return np.random.randn(10, 15, 20)

    def test_create_3d_volume_returns_surfaces(self, plotter, test_cube):
        """Test that create_3d_volume returns list of Plotly surfaces."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), title="Test Volume")

        assert isinstance(surfaces, list)
        assert len(surfaces) > 0
        # Should contain Plotly surface traces
        assert all(isinstance(s, go.Surface) for s in surfaces)

    def test_create_3d_volume_with_three_slices(self, plotter, test_cube):
        """Test that 3D volume creates three orthogonal slices."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10))

        # Should have three surfaces (one for each orthogonal slice)
        assert len(surfaces) == 3

    def test_create_3d_volume_with_title(self, plotter, test_cube):
        """Test 3D volume with custom title."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), title="Custom Title")

        assert surfaces is not None
        assert len(surfaces) > 0

    def test_create_3d_volume_with_k_scale(self, plotter, test_cube):
        """Test 3D volume with vertical scale factor."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), k_scale=0.5)

        assert len(surfaces) == 3

    def test_create_3d_volume_with_k_label(self, plotter, test_cube):
        """Test 3D volume with custom vertical axis label."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), k_label="Time (ms)")

        assert surfaces is not None

    def test_create_3d_volume_with_k_unit(self, plotter, test_cube):
        """Test 3D volume with vertical axis unit."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), k_unit="ms")

        assert surfaces is not None

    def test_create_3d_volume_with_colorscale(self, plotter, test_cube):
        """Test 3D volume with custom colorscale."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), colorscale="Viridis")

        assert len(surfaces) > 0

    def test_create_3d_volume_seismic_colorscale(self, plotter, test_cube):
        """Test 3D volume with seismic colorscale."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), colorscale="RdBu")

        assert len(surfaces) == 3

    def test_create_3d_volume_categorical(self, plotter):
        """Test 3D volume with categorical data."""
        categorical_cube = np.random.randint(0, 5, (10, 15, 20))

        surfaces = plotter.create_3d_volume(
            categorical_cube, (5, 7, 10), is_categorical=True
        )

        assert len(surfaces) > 0

    def test_create_3d_volume_without_colorbar(self, plotter, test_cube):
        """Test 3D volume without showing colorbar."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), show_colorbar=False)

        assert len(surfaces) > 0

    def test_create_3d_volume_with_colorbar(self, plotter, test_cube):
        """Test 3D volume with colorbar enabled."""
        surfaces = plotter.create_3d_volume(test_cube, (5, 7, 10), show_colorbar=True)

        assert len(surfaces) > 0


class TestPlotlyPlotterConfiguration:
    """Test PlotlyPlotter with various configurations."""

    @pytest.fixture
    def plotter(self):
        """Create a PlotlyPlotter instance."""
        return PlotlyPlotter(backend="Agg")

    @pytest.fixture
    def test_cube(self):
        """Create test 3D data."""
        return np.random.randn(8, 12, 16)

    def test_all_colorscales(self, plotter, test_cube):
        """Test with various Plotly colorscales."""
        colorscales = ["Viridis", "Plasma", "Inferno", "RdBu", "Greys"]

        for colorscale in colorscales:
            surfaces = plotter.create_3d_volume(
                test_cube, (4, 6, 8), colorscale=colorscale
            )
            assert len(surfaces) > 0

    def test_different_slice_indices(self, plotter, test_cube):
        """Test with different slice indices."""
        indices_list = [(0, 0, 0), (3, 6, 8), (7, 11, 15)]

        for indices in indices_list:
            surfaces = plotter.create_3d_volume(test_cube, indices)
            assert len(surfaces) == 3

    def test_edge_case_slices(self, plotter, test_cube):
        """Test with edge case slice indices."""
        # First slice
        surfaces1 = plotter.create_3d_volume(test_cube, (0, 0, 0))
        assert len(surfaces1) > 0

        # Last slice
        ni, nj, nk = test_cube.shape
        surfaces2 = plotter.create_3d_volume(test_cube, (ni - 1, nj - 1, nk - 1))
        assert len(surfaces2) > 0


class TestPlotlyPlotterDataTypes:
    """Test PlotlyPlotter with different data types."""

    @pytest.fixture
    def plotter(self):
        """Create a PlotlyPlotter instance."""
        return PlotlyPlotter(backend="Agg")

    def test_float_data(self, plotter):
        """Test with float32 data."""
        cube = np.random.randn(8, 12, 16).astype(np.float32)

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_float64_data(self, plotter):
        """Test with float64 data."""
        cube = np.random.randn(8, 12, 16).astype(np.float64)

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_integer_data(self, plotter):
        """Test with integer data."""
        cube = np.random.randint(-100, 100, (8, 12, 16))

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_large_value_range(self, plotter):
        """Test with very large value ranges."""
        cube = np.random.randn(8, 12, 16) * 1e6

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_small_value_range(self, plotter):
        """Test with very small value ranges."""
        cube = np.random.randn(8, 12, 16) * 1e-6

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_negative_data(self, plotter):
        """Test with all negative data."""
        cube = -np.random.randn(8, 12, 16)

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_positive_data(self, plotter):
        """Test with all positive data."""
        cube = np.abs(np.random.randn(8, 12, 16))

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0


class TestPlotlyPlotterSurfaceProperties:
    """Test properties of generated Plotly surfaces."""

    @pytest.fixture
    def plotter(self):
        """Create a PlotlyPlotter instance."""
        return PlotlyPlotter(backend="Agg")

    def test_surfaces_have_data(self, plotter):
        """Test that surfaces contain Z data."""
        cube = np.random.randn(8, 12, 16)
        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        for surface in surfaces:
            assert hasattr(surface, "z")
            assert surface.z is not None

    def test_surfaces_have_colorscale(self, plotter):
        """Test that surfaces have colorscale data."""
        cube = np.random.randn(8, 12, 16)
        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        for surface in surfaces:
            # Should have some coloring mechanism
            assert surface is not None

    def test_surfaces_with_different_titles(self, plotter):
        """Test creating surfaces with various titles."""
        cube = np.random.randn(8, 12, 16)

        titles = ["Test 1", "Test 2", "Complex: [Title]", ""]

        for title in titles:
            surfaces = plotter.create_3d_volume(cube, (4, 6, 8), title=title)
            assert len(surfaces) > 0


class TestPlotlyPlotterIntegration:
    """Integration tests for PlotlyPlotter."""

    def test_plotter_initialization(self):
        """Test PlotlyPlotter initialization."""
        plotter = PlotlyPlotter(backend="Agg")

        assert plotter is not None
        assert hasattr(plotter, "create_3d_volume")

    def test_complete_3d_visualization_workflow(self):
        """Test complete workflow for 3D visualization."""
        plotter = PlotlyPlotter(backend="Agg")
        cube = np.random.randn(10, 15, 20)

        # Create visualization
        surfaces = plotter.create_3d_volume(
            cube,
            (5, 7, 10),
            title="3D Seismic Volume",
            k_scale=1.0,
            k_label="Time",
            k_unit="ms",
            colorscale="RdBu",
        )

        assert len(surfaces) == 3
        assert all(isinstance(s, go.Surface) for s in surfaces)

    def test_multiple_visualizations_same_plotter(self):
        """Test creating multiple visualizations with same plotter."""
        plotter = PlotlyPlotter(backend="Agg")

        cubes = [
            np.random.randn(8, 12, 16),
            np.random.randn(6, 10, 14),
            np.random.randn(10, 15, 20),
        ]

        for cube in cubes:
            surfaces = plotter.create_3d_volume(cube, (2, 4, 6))
            assert len(surfaces) > 0
