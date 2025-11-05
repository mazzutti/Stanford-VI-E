"""Tests for plotting components (SliceExtractor, DataNormalizer, AxisStyler, ImageRenderer).

Tests verify component functionality for rendering, normalization, and data extraction.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from src.plotting.helpers.components import (
    SliceExtractor,
    DataNormalizer,
    AxisStyler,
    ImageRenderer,
)
from src.plotting.helpers.config import PlotConfig


class TestSliceExtractor:
    """Test SliceExtractor component for extracting 2D slices from 3D cubes."""

    @pytest.fixture
    def extractor(self):
        """Create a SliceExtractor instance."""
        return SliceExtractor(shape=(10, 15, 20))

    def test_extract_inline_slice(self, extractor):
        """Test extracting an inline slice."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        slice_data, xlabel, ylabel = extractor.extract_inline(cube, idx=0)

        assert slice_data.shape == (15, 20)
        assert xlabel == "Crossline (J)"
        assert ylabel == "Depth Index (K)"
        np.testing.assert_array_equal(slice_data, cube[0, :, :])

    def test_extract_crossline_slice(self, extractor):
        """Test extracting a crossline slice."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        slice_data, xlabel, ylabel = extractor.extract_crossline(cube, idx=1)

        assert slice_data.shape == (10, 20)
        assert xlabel == "Inline (I)"
        assert ylabel == "Depth Index (K)"
        np.testing.assert_array_equal(slice_data, cube[:, 1, :])

    def test_extract_depthslice(self, extractor):
        """Test extracting a depth/time slice."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        slice_data, xlabel, ylabel = extractor.extract_depthslice(cube, idx=2)

        assert slice_data.shape == (10, 15)
        assert xlabel == "Inline (I)"
        assert ylabel == "Crossline (J)"
        np.testing.assert_array_equal(slice_data, cube[:, :, 2])

    def test_extract_by_orientation_inline(self, extractor):
        """Test extract_by_orientation with inline."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        slice_data, xlabel, ylabel = extractor.extract_by_orientation(cube, 3, "inline")

        assert slice_data.shape == (15, 20)

    def test_extract_by_orientation_crossline(self, extractor):
        """Test extract_by_orientation with crossline."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        slice_data, xlabel, ylabel = extractor.extract_by_orientation(
            cube, 5, "crossline"
        )

        assert slice_data.shape == (10, 20)

    def test_extract_by_orientation_depthslice(self, extractor):
        """Test extract_by_orientation with depthslice."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        slice_data, xlabel, ylabel = extractor.extract_by_orientation(
            cube, 7, "depthslice"
        )

        assert slice_data.shape == (10, 15)

    def test_extract_by_orientation_timeslice(self, extractor):
        """Test extract_by_orientation with timeslice (alias for depthslice)."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        slice_data, xlabel, ylabel = extractor.extract_by_orientation(
            cube, 8, "timeslice"
        )

        assert slice_data.shape == (10, 15)

    def test_extract_by_orientation_invalid(self, extractor):
        """Test extract_by_orientation with invalid orientation."""
        cube = np.arange(10 * 15 * 20).reshape(10, 15, 20)

        with pytest.raises(ValueError):
            extractor.extract_by_orientation(cube, 0, "invalid")


class TestDataNormalizer:
    """Test DataNormalizer component for computing limits and colormaps."""

    def test_compute_limits_default(self):
        """Test computing limits with default percentile."""
        data = np.array([1, 2, 3, 4, 5])

        vmin, vmax = DataNormalizer.compute_limits(data)

        # Should be symmetric around 0
        assert vmin == -vmax
        assert vmax > 0

    def test_compute_limits_categorical(self):
        """Test computing limits for categorical data."""
        data = np.array([0, 1, 2, 3])

        vmin, vmax = DataNormalizer.compute_limits(data, is_categorical=True)

        assert vmin == 0.0
        assert vmax == 3.0

    def test_compute_limits_custom_percentile(self):
        """Test computing limits with custom percentile."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier

        vmin, vmax = DataNormalizer.compute_limits(data, percentile=90)

        # Should exclude the outlier at 100
        assert vmax < 100

    def test_compute_limits_handles_zeros(self):
        """Test computing limits with all zero data."""
        data = np.zeros(10)

        vmin, vmax = DataNormalizer.compute_limits(data)

        # Should still have different vmin/vmax
        assert vmin != vmax

    def test_get_discrete_colormap(self):
        """Test getting a discrete colormap."""
        cmap = DataNormalizer.get_discrete_colormap(n_colors=4)

        assert cmap is not None
        # Should have 4 colors
        assert len(cmap.colors) == 4

    def test_get_discrete_colormap_different_sizes(self):
        """Test discrete colormaps with different numbers of colors."""
        for n_colors in [2, 4, 8]:
            cmap = DataNormalizer.get_discrete_colormap(n_colors=n_colors)
            assert len(cmap.colors) == n_colors


class TestAxisStyler:
    """Test AxisStyler component for applying axis styling."""

    def test_style_axis_basic(self):
        """Test basic axis styling."""
        fig, ax = plt.subplots()

        AxisStyler.style_axis(ax, title="Test Title", xlabel="X", ylabel="Y")

        assert ax.get_title() == "Test Title"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        plt.close(fig)

    def test_style_axis_with_grid(self):
        """Test axis styling with grid."""
        fig, ax = plt.subplots()

        AxisStyler.style_axis(ax, grid=True, grid_alpha=0.5)

        plt.close(fig)

    def test_style_axis_without_grid(self):
        """Test axis styling without grid."""
        fig, ax = plt.subplots()

        AxisStyler.style_axis(ax, grid=False)

        plt.close(fig)

    def test_style_axis_font_sizes(self):
        """Test axis styling with custom font sizes."""
        fig, ax = plt.subplots()

        AxisStyler.style_axis(ax, title="Title", fontsize_title=14, fontsize_labels=11)

        assert ax.get_title() == "Title"
        plt.close(fig)

    def test_add_colorbar(self):
        """Test adding a colorbar."""
        fig, ax = plt.subplots()
        data = np.random.randn(10, 10)
        im = ax.imshow(data)

        cbar = AxisStyler.add_colorbar(im, ax, label="Test Label")

        assert cbar is not None
        plt.close(fig)


class TestImageRenderer:
    """Test ImageRenderer component for rendering images with colorbars."""

    def test_render_basic_image(self):
        """Test rendering a basic image."""
        fig, ax = plt.subplots()
        data = np.random.randn(10, 10)
        config = PlotConfig()

        im, cbar = ImageRenderer.render(ax, data, config)

        assert im is not None
        assert cbar is not None
        plt.close(fig)

    def test_render_with_title(self):
        """Test rendering with title."""
        fig, ax = plt.subplots()
        data = np.random.randn(10, 10)
        config = PlotConfig(title="Test Image")

        im, cbar = ImageRenderer.render(ax, data, config)

        assert ax.get_title() == "Test Image"
        plt.close(fig)

    def test_render_categorical_data(self):
        """Test rendering categorical data."""
        fig, ax = plt.subplots()
        data = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        config = PlotConfig(is_categorical=True)

        im, cbar = ImageRenderer.render(ax, data, config)

        assert im is not None
        plt.close(fig)

    def test_render_without_colorbar(self):
        """Test rendering without colorbar."""
        fig, ax = plt.subplots()
        data = np.random.randn(10, 10)
        config = PlotConfig(show_colorbar=False)

        im, cbar = ImageRenderer.render(ax, data, config)

        assert im is not None
        assert cbar is None
        plt.close(fig)

    def test_render_with_custom_limits(self):
        """Test rendering with custom vmin/vmax."""
        fig, ax = plt.subplots()
        data = np.random.randn(10, 10)
        config = PlotConfig()

        im, cbar = ImageRenderer.render(ax, data, config, vmin=-1, vmax=1)

        assert im is not None
        plt.close(fig)

    def test_render_with_custom_cmap(self):
        """Test rendering with custom colormap."""
        fig, ax = plt.subplots()
        data = np.random.randn(10, 10)
        config = PlotConfig(cmap="viridis")

        im, cbar = ImageRenderer.render(ax, data, config)

        assert im is not None
        plt.close(fig)

    def test_render_returns_tuple(self):
        """Test that render returns a tuple."""
        fig, ax = plt.subplots()
        data = np.random.randn(10, 10)
        config = PlotConfig()

        result = ImageRenderer.render(ax, data, config)

        assert isinstance(result, tuple)
        assert len(result) == 2
        plt.close(fig)


class TestComponentIntegration:
    """Integration tests for component interactions."""

    def test_extract_and_render_workflow(self):
        """Test typical workflow: extract slice and render."""
        # Create test data
        cube = np.random.randn(5, 10, 8)

        # Extract slice
        extractor = SliceExtractor(shape=(5, 10, 8))
        slice_data, xlabel, ylabel = extractor.extract_inline(cube, idx=2)

        # Render slice
        fig, ax = plt.subplots()
        config = PlotConfig.for_seismic()
        im, cbar = ImageRenderer.render(ax, slice_data, config)

        assert im is not None
        assert slice_data.shape == (10, 8)
        plt.close(fig)

    def test_normalize_and_render_workflow(self):
        """Test normalizing data and rendering."""
        data = np.random.randn(20, 20) * 1000  # Large values

        # Normalize
        vmin, vmax = DataNormalizer.compute_limits(data, percentile=98)

        # Render
        fig, ax = plt.subplots()
        config = PlotConfig()
        im, cbar = ImageRenderer.render(ax, data, config, vmin=vmin, vmax=vmax)

        assert im is not None
        plt.close(fig)

    def test_full_rendering_pipeline(self):
        """Test complete rendering pipeline."""
        # 3D cube
        cube = np.random.randn(10, 15, 20)

        # Extract
        extractor = SliceExtractor(shape=(10, 15, 20))
        slice_data, xlabel, ylabel = extractor.extract_crossline(cube, idx=7)

        # Normalize
        vmin, vmax = DataNormalizer.compute_limits(slice_data)

        # Render with styling
        fig, ax = plt.subplots()
        config = PlotConfig(
            title="Pipeline Test",
            xlabel=xlabel,
            ylabel=ylabel,
            cmap="seismic",
        )
        im, cbar = ImageRenderer.render(ax, slice_data, config, vmin=vmin, vmax=vmax)

        assert im is not None
        assert ax.get_title() == "Pipeline Test"
        plt.close(fig)
