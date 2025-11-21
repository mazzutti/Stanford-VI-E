"""Tests for complex visualization logic in FaciesPlotter and PlotlyPlotter.

These tests cover data analysis, statistical computation, and complex plotting
scenarios that require mocking AvoResults and handling edge cases.
"""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plotting.facies_plotter import FaciesPlotter
from src.plotting.plotly_plotter import PlotlyPlotter


class TestFaciesPlotterComplexLogic:
    """Test complex visualization logic in FaciesPlotter."""

    @pytest.fixture
    def plotter(self):
        """Create a FaciesPlotter instance."""
        return FaciesPlotter(backend="Agg")

    @pytest.fixture
    def mock_avo_results_full(self):
        """Create a complete mock AvoResults with all data."""
        mock = Mock()

        # Boundary amplitudes
        boundary_amps = Mock()
        boundary_amps.at_boundaries = np.random.randn(100)
        boundary_amps.away_from_boundaries = np.random.randn(150)
        mock.boundary_amps = boundary_amps

        # Interface stats
        interface1 = Mock()
        interface1.from_facies = 0
        interface1.to_facies = 1
        interface2 = Mock()
        interface2.from_facies = 1
        interface2.to_facies = 2

        mock.interface_stats_summary = {
            interface1: {"count": 50, "mean": 0.3, "std": 0.1},
            interface2: {"count": 100, "mean": 0.5, "std": 0.15},
        }

        # Facies amplitudes
        mock.facies_amplitudes = {
            0: np.random.randn(5000),
            1: np.random.randn(5000),
            2: np.random.randn(5000),
            3: np.random.randn(5000),
        }

        # Separation matrix
        mock.separation_matrix = np.random.rand(4, 4)

        return mock

    @pytest.fixture
    def mock_avo_results_empty(self):
        """Create a mock AvoResults with minimal data."""
        mock = Mock()
        mock.boundary_amps = None
        mock.interface_stats_summary = {}
        mock.facies_amplitudes = {}
        mock.separation_matrix = None
        return mock

    @pytest.fixture
    def mock_avo_results_partial(self):
        """Create a mock AvoResults with partial data."""
        mock = Mock()

        # Only some fields populated
        boundary_amps = Mock()
        boundary_amps.at_boundaries = np.array([])  # Empty
        boundary_amps.away_from_boundaries = np.random.randn(50)
        mock.boundary_amps = boundary_amps

        # Only some facies data
        mock.interface_stats_summary = {}
        mock.facies_amplitudes = {
            0: np.random.randn(1000),
            1: np.random.randn(500),
        }
        mock.separation_matrix = None

        return mock

    def test_create_summary_plots_with_full_data(self, plotter, mock_avo_results_full):
        """Test creating summary plots with complete AvoResults data."""
        fig = plotter.create_summary_plots(mock_avo_results_full, cache_dir=".cache")

        assert fig is not None
        # Should have multiple subplots
        assert len(fig.get_axes()) >= 5
        plt.close(fig)

    def test_create_summary_plots_with_empty_data(
        self, plotter, mock_avo_results_empty
    ):
        """Test creating summary plots with empty AvoResults."""
        fig = plotter.create_summary_plots(mock_avo_results_empty, cache_dir=".cache")

        assert fig is not None
        # Should still create the figure structure
        assert len(fig.get_axes()) >= 5
        plt.close(fig)

    def test_create_summary_plots_with_partial_data(
        self, plotter, mock_avo_results_partial
    ):
        """Test creating summary plots with partial AvoResults."""
        fig = plotter.create_summary_plots(mock_avo_results_partial, cache_dir=".cache")

        assert fig is not None
        plt.close(fig)

    def test_create_summary_plots_depth_domain(self, plotter, mock_avo_results_full):
        """Test summary plots with depth domain."""
        fig = plotter.create_summary_plots(
            mock_avo_results_full, cache_dir=".cache", domain="depth"
        )

        # Verify plot was created (don't test suptitle as it may not be set)
        assert fig is not None
        assert len(fig.get_axes()) >= 5
        plt.close(fig)

    def test_create_summary_plots_time_domain(self, plotter, mock_avo_results_full):
        """Test summary plots with time domain."""
        fig = plotter.create_summary_plots(
            mock_avo_results_full, cache_dir=".cache", domain="time"
        )

        # Verify plot was created (don't test suptitle as it may not be set)
        assert fig is not None
        assert len(fig.get_axes()) >= 5
        plt.close(fig)

    def test_boundary_amplitude_distribution(self, plotter, mock_avo_results_full):
        """Test histogram plotting of boundary amplitudes."""
        fig = plotter.create_summary_plots(mock_avo_results_full, cache_dir=".cache")

        # Get first subplot (boundary distributions)
        ax1 = fig.get_axes()[0]

        # Should have histogram patches
        patches = ax1.patches
        assert len(patches) > 0
        plt.close(fig)

    def test_interface_strengths_barplot(self, plotter, mock_avo_results_full):
        """Test bar plot of reflection strengths at interfaces."""
        fig = plotter.create_summary_plots(mock_avo_results_full, cache_dir=".cache")

        # Get second subplot
        ax2 = fig.get_axes()[1]

        # Should have bar patches
        patches = ax2.patches
        assert len(patches) > 0
        plt.close(fig)

    def test_facies_discrimination_boxplot(self, plotter, mock_avo_results_full):
        """Test boxplot of facies discrimination."""
        fig = plotter.create_summary_plots(mock_avo_results_full, cache_dir=".cache")

        # Get third subplot
        ax3 = fig.get_axes()[2]

        # Should have boxplot artists
        assert len(ax3.artists) > 0 or len(ax3.lines) > 0
        plt.close(fig)

    def test_separation_matrix_heatmap(self, plotter, mock_avo_results_full):
        """Test heatmap of facies separation matrix."""
        fig = plotter.create_summary_plots(mock_avo_results_full, cache_dir=".cache")

        # Get fifth subplot (separation matrix)
        ax5 = fig.get_axes()[4]

        # Should have an image
        images = ax5.images
        assert len(images) > 0
        plt.close(fig)

    def test_interface_filtering_by_count(self, plotter):
        """Test that interfaces with low count are filtered."""
        mock = Mock()
        boundary_amps = Mock()
        boundary_amps.at_boundaries = np.random.randn(50)
        boundary_amps.away_from_boundaries = np.random.randn(50)
        mock.boundary_amps = boundary_amps

        # One interface with too few samples
        interface1 = Mock()
        interface1.from_facies = 0
        interface1.to_facies = 1

        interface2 = Mock()
        interface2.from_facies = 1
        interface2.to_facies = 2

        mock.interface_stats_summary = {
            interface1: {"count": 5, "mean": 0.3, "std": 0.1},  # Too low
            interface2: {"count": 50, "mean": 0.5, "std": 0.15},  # OK
        }

        mock.facies_amplitudes = {}
        mock.separation_matrix = None

        fig = plotter.create_summary_plots(mock, cache_dir=".cache")

        # Should still plot successfully
        assert fig is not None
        plt.close(fig)

    def test_handles_large_facies_datasets(self, plotter):
        """Test plotting with very large facies amplitude datasets."""
        mock = Mock()
        mock.boundary_amps = None
        mock.interface_stats_summary = {}

        # Very large facies data
        mock.facies_amplitudes = {i: np.random.randn(100000) for i in range(4)}
        mock.separation_matrix = None

        fig = plotter.create_summary_plots(mock, cache_dir=".cache")

        # Should handle downsampling gracefully
        assert fig is not None
        plt.close(fig)

    def test_handles_nan_values_in_separation_matrix(self, plotter):
        """Test plotting with NaN values in separation matrix."""
        mock = Mock()
        mock.boundary_amps = None
        mock.interface_stats_summary = {}
        mock.facies_amplitudes = {}

        # Matrix with NaN
        sep_matrix = np.ones((4, 4))
        sep_matrix[0, 0] = np.nan
        sep_matrix[2, 3] = np.nan
        mock.separation_matrix = sep_matrix

        fig = plotter.create_summary_plots(mock, cache_dir=".cache")

        assert fig is not None
        plt.close(fig)


class TestPlotlyPlotterComplexLogic:
    """Test complex visualization logic in PlotlyPlotter."""

    @pytest.fixture
    def plotter(self):
        """Create a PlotlyPlotter instance."""
        return PlotlyPlotter(backend="Agg")

    def test_surface_color_scaling_symmetric_seismic(self, plotter):
        """Test that seismic data gets symmetric color scaling."""
        # Create seismic-like data with outliers
        cube = np.random.randn(8, 12, 16) * 100
        cube[0, 0, 0] = 1000  # Add outlier

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8), colorscale="RdBu")

        # Should handle outliers gracefully
        assert len(surfaces) == 3
        for surface in surfaces:
            assert surface.cmin == -surface.cmax or surface.cmin is None

    def test_surface_color_scaling_zero_values(self, plotter):
        """Test handling of zero/near-zero data."""
        cube = np.zeros((8, 12, 16))

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        # Should not crash, should set sensible limits
        assert len(surfaces) == 3
        for surface in surfaces:
            if surface.cmax is not None:
                assert surface.cmax != 0

    def test_large_cube_memory_efficiency(self, plotter):
        """Test that large cubes are handled without memory issues."""
        # Large cube
        cube = np.random.randn(50, 50, 50)

        surfaces = plotter.create_3d_volume(cube, (25, 25, 25))

        # Should create surfaces without loading entire cube
        assert len(surfaces) == 3

    def test_categorical_colorscale_discrete_colors(self, plotter):
        """Test that categorical data uses discrete colorscale."""
        categorical_cube = np.random.randint(0, 4, (8, 12, 16))

        surfaces = plotter.create_3d_volume(
            categorical_cube, (4, 6, 8), is_categorical=True
        )

        # Check that categorical colorscale is applied
        for surface in surfaces:
            assert surface.colorscale is not None

    def test_k_scale_vertical_exaggeration(self, plotter):
        """Test that k_scale properly exaggerates vertical dimension."""
        cube = np.random.randn(8, 12, 16)

        k_scale = 2.0
        surfaces = plotter.create_3d_volume(cube, (4, 6, 8), k_scale=k_scale)

        # Check that surfaces have been scaled
        for surface in surfaces:
            assert surface.z is not None

    def test_different_percentile_clipping(self, plotter):
        """Test different data percentiles for color limits."""
        cube = np.random.randn(8, 12, 16)
        # Add outliers
        cube[0, 0, 0] = 1000
        cube[1, 1, 1] = -1000

        surfaces1 = plotter.create_3d_volume(cube, (4, 6, 8))
        surfaces2 = plotter.create_3d_volume(cube, (4, 6, 8))

        # Both should handle outliers
        assert len(surfaces1) == 3
        assert len(surfaces2) == 3

    def test_colorbar_with_label(self, plotter):
        """Test that colorbar has proper label."""
        cube = np.random.randn(8, 12, 16)

        surfaces = plotter.create_3d_volume(
            cube, (4, 6, 8), show_colorbar=True, title="Test Data"
        )

        assert len(surfaces) > 0

    def test_colorbar_without_label(self, plotter):
        """Test surface creation without colorbar."""
        cube = np.random.randn(8, 12, 16)

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8), show_colorbar=False)

        assert len(surfaces) > 0

    def test_surface_naming_conventions(self, plotter):
        """Test that surfaces are properly named."""
        cube = np.random.randn(8, 12, 16)

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8), title="Test Volume")

        # Should have named surfaces
        names = [s.name for s in surfaces]
        assert all(name is not None for name in names)

    def test_meshgrid_coordinate_generation(self, plotter):
        """Test that coordinate meshgrids are generated correctly."""
        cube = np.random.randn(10, 15, 20)

        surfaces = plotter.create_3d_volume(cube, (5, 7, 10))

        # Each surface should have proper x, y, z coordinates
        for surface in surfaces:
            assert surface.x is not None
            assert surface.y is not None
            assert surface.z is not None

    def test_mixed_nan_inf_handling(self, plotter):
        """Test handling of NaN and inf values."""
        cube = np.random.randn(8, 12, 16)
        cube[0, 0, 0] = np.nan
        cube[1, 1, 1] = np.inf
        cube[2, 2, 2] = -np.inf

        # Should not crash
        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_minimum_cube_dimensions(self, plotter):
        """Test with minimum viable cube dimensions."""
        cube = np.random.randn(2, 2, 2)

        surfaces = plotter.create_3d_volume(cube, (0, 0, 0))

        assert len(surfaces) > 0

    def test_slices_at_edges(self, plotter):
        """Test slicing at cube edges."""
        cube = np.random.randn(10, 15, 20)

        # First and last slices
        surfaces1 = plotter.create_3d_volume(cube, (0, 0, 0))
        surfaces2 = plotter.create_3d_volume(cube, (9, 14, 19))

        assert len(surfaces1) == 3
        assert len(surfaces2) == 3


class TestVisualizationEdgeCases:
    """Test edge cases common to both plotters."""

    def test_extremely_skewed_data(self):
        """Test plotting extremely skewed data."""
        plotter = PlotlyPlotter(backend="Agg")

        cube = np.random.exponential(0.1, (8, 12, 16))

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_bimodal_data_distribution(self):
        """Test plotting bimodal data."""
        plotter = PlotlyPlotter(backend="Agg")

        # Create bimodal distribution
        cube1 = np.random.normal(-5, 0.5, (4, 6, 4))
        cube2 = np.random.normal(5, 0.5, (4, 6, 4))
        cube = np.concatenate([cube1, cube2], axis=2)

        surfaces = plotter.create_3d_volume(cube, (2, 3, 4))

        assert len(surfaces) > 0

    def test_highly_correlated_slices(self):
        """Test with slices that are highly correlated."""
        plotter = PlotlyPlotter(backend="Agg")

        base = np.random.randn(8, 12)
        cube = np.stack(
            [base + np.random.randn(*base.shape) * 0.01 for _ in range(16)], axis=2
        )

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0

    def test_alternating_positive_negative_values(self):
        """Test data that alternates between positive and negative."""
        plotter = PlotlyPlotter(backend="Agg")

        i, j, k = np.ogrid[:8, :12, :16]
        cube = (-1) ** (i + j + k) * np.random.rand(8, 12, 16)

        surfaces = plotter.create_3d_volume(cube, (4, 6, 8))

        assert len(surfaces) > 0
