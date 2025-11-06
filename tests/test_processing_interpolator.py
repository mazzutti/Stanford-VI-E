"""Tests for BatchedInterpolator for vectorized depth-to-time interpolation.

Tests the block-wise interpolation logic for both uniform and irregular
two-way travel times.
"""

import pytest
import numpy as np
from src.processing.interpolator import BatchedInterpolator


class TestBatchedInterpolatorInitialization:
    """Test BatchedInterpolator initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with time axis."""
        time_axis = np.linspace(0, 10, 100)
        interp = BatchedInterpolator(time_axis=time_axis)

        assert np.array_equal(interp.time_axis, time_axis)
        assert interp.kind == "linear"
        assert interp.block_size == 65536

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom kind and block size."""
        time_axis = np.linspace(0, 10, 100)
        interp = BatchedInterpolator(
            time_axis=time_axis, kind="nearest", block_size=1024
        )

        assert interp.kind == "nearest"
        assert interp.block_size == 1024

    def test_time_axis_property(self):
        """Test that time_axis is properly stored."""
        time_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        interp = BatchedInterpolator(time_axis=time_axis)

        assert len(interp.time_axis) == 5


class TestBatchedInterpolatorUniformTWT:
    """Test BatchedInterpolator with uniform (1D) TWT."""

    def test_interpolate_1d_twt_single_trace(self):
        """Test interpolation with 1D TWT and single trace."""
        time_axis = np.array([0.0, 0.5, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear")

        # 1D TWT: uniform across traces
        twt_padded = np.array([0.0, 1.0, 2.0])
        # Single trace padded (shape: (3, 1))
        depth_padded_flat = np.array([[0.0], [10.0], [20.0]], dtype=np.float64)

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (3, 1)
        assert result[0, 0] == 0.0
        assert np.isclose(result[1, 0], 5.0)
        assert result[2, 0] == 10.0

    def test_interpolate_1d_twt_multiple_traces_no_blocking(self):
        """Test interpolation with 1D TWT and multiple traces (no blocking)."""
        time_axis = np.array([0.0, 1.0, 2.0])
        interp = BatchedInterpolator(
            time_axis=time_axis,
            kind="linear",
            block_size=100,  # Larger than trace count
        )

        twt_padded = np.array([0.0, 1.0, 2.0])
        # 3 traces padded (shape: (3, 3))
        depth_padded_flat = np.array(
            [[0.0, 10.0, 100.0], [1.0, 11.0, 101.0], [2.0, 12.0, 102.0]],
            dtype=np.float64,
        )

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (3, 3)
        # Check first trace: linear from 0 to 1 to 2
        np.testing.assert_array_almost_equal(result[:, 0], [0.0, 1.0, 2.0])

    def test_interpolate_1d_twt_nearest_neighbor(self):
        """Test 1D TWT with nearest-neighbor interpolation."""
        time_axis = np.array([0.0, 0.5, 1.0])
        interp = BatchedInterpolator(
            time_axis=time_axis, kind="nearest", block_size=100
        )

        twt_padded = np.array([0.0, 1.0, 2.0])
        depth_padded_flat = np.array([[10], [20], [30]], dtype=np.float64)

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (3, 1)
        # With nearest-neighbor, should pick nearest depth value


class TestBatchedInterpolator2DTWT:
    """Test BatchedInterpolator with 2D (irregular) TWT."""

    def test_interpolate_2d_twt_single_block(self):
        """Test interpolation with 2D TWT in single block."""
        time_axis = np.array([0.0, 1.0, 2.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        # 2D TWT: different per trace (shape: (3, 2))
        twt_padded = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]], dtype=np.float64)
        # Depth data matching TWT shape
        depth_padded_flat = np.array(
            [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float64
        )

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (3, 2)

    def test_interpolate_2d_twt_with_blocking(self):
        """Test 2D TWT interpolation with block processing."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=2)

        # 2D TWT (shape: (2, 4))
        twt_padded = np.array(
            [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=np.float64
        )
        depth_padded_flat = np.array(
            [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]], dtype=np.float64
        )

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (2, 4)

    def test_interpolate_2d_twt_uniform_columns(self):
        """Test 2D TWT where columns are actually uniform (optimization path)."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=2)

        # 2D TWT where all columns are identical
        twt_padded = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        depth_padded_flat = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (2, 3)
        # Since columns have same twt, should get linear interpolation
        np.testing.assert_array_almost_equal(result[0, :], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result[1, :], [4.0, 5.0, 6.0])


class TestBatchedInterpolatorEdgeCases:
    """Test BatchedInterpolator edge cases."""

    def test_empty_time_axis(self):
        """Test with empty time axis."""
        time_axis = np.array([], dtype=np.float64)
        interp = BatchedInterpolator(time_axis=time_axis)

        twt_padded = np.array([0.0, 1.0])
        depth_padded_flat = np.array([[0.0], [10.0]])

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (0, 1)

    def test_single_time_sample(self):
        """Test with single time sample."""
        time_axis = np.array([0.5])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear")

        twt_padded = np.array([0.0, 1.0])
        depth_padded_flat = np.array([[0.0], [10.0]], dtype=np.float64)

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 5.0)

    def test_single_depth_sample(self):
        """Test with single depth sample."""
        time_axis = np.array([0.0, 0.5, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear")

        twt_padded = np.array([0.0])  # Single depth
        depth_padded_flat = np.array([[10.0]], dtype=np.float64)

        # Suppress RuntimeWarning from scipy when interpolating with single point
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (3, 1)
        # All time samples should extrapolate/use the single value

    def test_very_large_block_size(self):
        """Test with block size larger than number of traces."""
        time_axis = np.array([0.0, 1.0, 2.0])
        interp = BatchedInterpolator(
            time_axis=time_axis, kind="linear", block_size=1000000
        )

        twt_padded = np.array([0.0, 1.0, 2.0])
        depth_padded_flat = np.array(
            [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float64
        )

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (3, 2)

    def test_very_small_block_size(self):
        """Test with very small block size."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=1)

        twt_padded = np.array([0.0, 1.0])
        depth_padded_flat = np.array(
            [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=np.float64
        )

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.shape == (2, 3)


class TestBatchedInterpolatorDataTypes:
    """Test BatchedInterpolator with different data types."""

    def test_float32_data(self):
        """Test with float32 data."""
        time_axis = np.array([0.0, 1.0], dtype=np.float32)
        interp = BatchedInterpolator(time_axis=time_axis)

        twt_padded = np.array([0.0, 1.0], dtype=np.float32)
        depth_padded_flat = np.array([[0.0], [10.0]], dtype=np.float32)

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.dtype == np.float32

    def test_float64_data(self):
        """Test with float64 data."""
        time_axis = np.array([0.0, 1.0], dtype=np.float64)
        interp = BatchedInterpolator(time_axis=time_axis)

        twt_padded = np.array([0.0, 1.0], dtype=np.float64)
        depth_padded_flat = np.array([[0.0], [10.0]], dtype=np.float64)

        result = interp.interpolate(twt_padded, depth_padded_flat)

        assert result.dtype == np.float64


class TestBatchedInterpolatorMonotonicity:
    """Test interpolation behavior with monotonic data."""

    def test_monotonically_increasing_data(self):
        """Test interpolation with monotonically increasing values."""
        time_axis = np.linspace(0, 2, 5)
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear")

        twt_padded = np.array([0.0, 1.0, 2.0])
        depth_padded_flat = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

        result = interp.interpolate(twt_padded, depth_padded_flat)

        # Result should be monotonically increasing
        assert np.all(np.diff(result[:, 0]) >= -1e-10)

    def test_constant_data(self):
        """Test interpolation with constant values."""
        time_axis = np.array([0.0, 1.0, 2.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear")

        twt_padded = np.array([0.0, 1.0, 2.0])
        depth_padded_flat = np.array([[5.0], [5.0], [5.0]], dtype=np.float64)

        result = interp.interpolate(twt_padded, depth_padded_flat)

        # Should be all 5.0
        np.testing.assert_array_almost_equal(result[:, 0], 5.0)


class TestBatchedInterpolatorNearest:
    """Test BatchedInterpolator.nearest() method."""

    def test_nearest_1d_twt_single_trace(self):
        """Test nearest with 1D TWT and single trace."""
        time_axis = np.array([0.0, 0.5, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear")

        twt_padded = np.array([0.0, 1.0, 2.0])
        depth_padded_flat = np.array([[0.0], [10.0], [20.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (3, 1)
        # At t=0.0, nearest to twt=[0, 1, 2] is index 0 -> depth 0
        assert result[0, 0] == 0.0
        # At t=0.5, equidistant from 0 and 1, chooses lower -> 0
        # At t=1.0, nearest to twt[1]=1.0 -> depth 10
        assert result[2, 0] == 10.0

    def test_nearest_1d_twt_multiple_traces_no_blocking(self):
        """Test nearest with 1D TWT and multiple traces (no blocking)."""
        time_axis = np.array([0.0, 1.0, 2.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        twt_padded = np.array([0.0, 1.0, 2.0])
        depth_padded_flat = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64
        )

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (3, 2)
        # First trace should have values [1, 2, 3]
        np.testing.assert_array_almost_equal(result[:, 0], [1.0, 2.0, 3.0])

    def test_nearest_1d_twt_with_blocking(self):
        """Test nearest with 1D TWT and trace blocking."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=2)

        twt_padded = np.array([0.0, 1.0])
        depth_padded_flat = np.array(
            [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=np.float64
        )

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 3)

    def test_nearest_2d_twt_single_column(self):
        """Test nearest with 2D TWT (single column per block)."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        # 2D TWT (shape: (2, 1))
        twt_padded = np.array([[0.0], [1.0]], dtype=np.float64)
        depth_padded_flat = np.array([[5.0], [15.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 1)

    def test_nearest_2d_twt_multiple_columns(self):
        """Test nearest with 2D TWT and multiple columns."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        # 2D TWT (shape: (2, 2)) with different values per column
        twt_padded = np.array([[0.0, 0.1], [1.0, 1.1]], dtype=np.float64)
        depth_padded_flat = np.array([[5.0, 6.0], [15.0, 16.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 2)

    def test_nearest_2d_twt_with_blocking(self):
        """Test nearest with 2D TWT and blocking."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=2)

        # 2D TWT (shape: (2, 4))
        twt_padded = np.array(
            [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=np.float64
        )
        depth_padded_flat = np.array(
            [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]], dtype=np.float64
        )

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 4)

    def test_nearest_distance_selection(self):
        """Test that nearest selects closest value correctly."""
        time_axis = np.array([0.25, 0.75])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        # TWT at 0.0 and 1.0
        twt_padded = np.array([0.0, 1.0])
        depth_padded_flat = np.array([[1.0], [2.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 1)
        # At t=0.25: distance to 0.0 is 0.25, to 1.0 is 0.75 -> choose 0.0 -> depth 1.0
        assert result[0, 0] == 1.0
        # At t=0.75: distance to 0.0 is 0.75, to 1.0 is 0.25 -> choose 1.0 -> depth 2.0
        assert result[1, 0] == 2.0

    def test_nearest_edge_values(self):
        """Test nearest at edge time values."""
        time_axis = np.array([0.0, 2.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        twt_padded = np.array([0.0, 1.0, 2.0])
        depth_padded_flat = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 1)
        # At t=0, exact match with twt[0] -> depth 0
        assert result[0, 0] == 0.0
        # At t=2, exact match with twt[2] -> depth 2
        assert result[1, 0] == 2.0

    def test_nearest_out_of_bounds_low(self):
        """Test nearest with time value below minimum TWT."""
        time_axis = np.array([-1.0, 0.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        twt_padded = np.array([0.0, 1.0])
        depth_padded_flat = np.array([[10.0], [20.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 1)
        # At t=-1.0, closest is twt[0]=0.0 -> depth 10
        assert result[0, 0] == 10.0

    def test_nearest_out_of_bounds_high(self):
        """Test nearest with time value above maximum TWT."""
        time_axis = np.array([1.0, 2.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        twt_padded = np.array([0.0, 1.0])
        depth_padded_flat = np.array([[10.0], [20.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 1)
        # At t=2.0, closest is twt[1]=1.0 -> depth 20
        assert result[1, 0] == 20.0

    def test_nearest_with_uniform_2d_twt_columns(self):
        """Test nearest where 2D TWT columns are uniform."""
        time_axis = np.array([0.0, 1.0])
        interp = BatchedInterpolator(time_axis=time_axis, kind="linear", block_size=100)

        # All columns have same TWT
        twt_padded = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        depth_padded_flat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        result = interp.nearest(twt_padded, depth_padded_flat)

        assert result.shape == (2, 2)
        # Should behave like 1D TWT case
        np.testing.assert_array_almost_equal(result[:, 0], [1.0, 3.0])
        np.testing.assert_array_almost_equal(result[:, 1], [2.0, 4.0])
