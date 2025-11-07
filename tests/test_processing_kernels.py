"""Tests for resampling kernels with Numba JIT compilation.

Tests the depth-to-time and time-to-depth resampling kernels including
nearest-neighbor and linear interpolation modes.
"""

import pytest
import numpy as np

from src.processing.resampling._kernels import (
    resample_depth_to_time_nearest,
    resample_depth_to_time_linear,
    resample_depth_to_time_from_irregular_nearest,
    resample_depth_to_time_from_irregular_linear,
)


class TestResampleDepthToTimeNearest:
    """Test nearest-neighbor depth-to-time resampling."""

    def test_resample_depth_to_time_nearest_single_trace(self):
        """Test nearest-neighbor with single trace."""
        # Setup: single trace with 3 depths
        ni, nj, nz = 1, 1, 3
        nt = 4

        twt_irregular = np.array([[[0.0, 0.5, 1.0]]])
        data_depth = np.array([[[1, 2, 3]]], dtype=np.int32)
        time_axis = np.array([0.0, 0.3, 0.7, 1.0])
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_nearest(twt_irregular, data_depth, time_axis, out_array)

        assert out_array.shape == (ni, nj, nt)
        assert out_array[0, 0, 0] == 1  # t=0.0 -> nearest to twt[0]=0.0
        assert (
            out_array[0, 0, 1] == 2
        )  # t=0.3 -> nearest to twt[1]=0.5 (dist 0.2 vs 0.3)
        assert (
            out_array[0, 0, 2] == 2
        )  # t=0.7 -> nearest to twt[1]=0.5 (dist 0.2 vs 0.3)
        assert out_array[0, 0, 3] == 3  # t=1.0 -> nearest to twt[2]=1.0

    def test_resample_depth_to_time_nearest_multiple_traces(self):
        """Test nearest-neighbor with multiple traces."""
        ni, nj, nz = 2, 2, 4
        nt = 3

        twt_irregular = np.array(
            [
                [[0.0, 0.2, 0.4, 0.6], [0.1, 0.3, 0.5, 0.7]],
                [[0.0, 0.25, 0.5, 0.75], [0.05, 0.35, 0.65, 0.95]],
            ]
        )
        data_depth = np.ones((ni, nj, nz), dtype=np.int32)
        data_depth[0, 0, :] = [10, 20, 30, 40]
        data_depth[0, 1, :] = [100, 200, 300, 400]
        data_depth[1, 0, :] = [11, 21, 31, 41]
        data_depth[1, 1, :] = [101, 201, 301, 401]

        time_axis = np.array([0.0, 0.5, 1.0])
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_nearest(twt_irregular, data_depth, time_axis, out_array)

        assert out_array.shape == (ni, nj, nt)
        # All traces should have same values at each time sample
        assert out_array[0, 0, 0] == 10
        assert out_array[0, 1, 0] == 100

    def test_resample_depth_to_time_nearest_boundary_below(self):
        """Test nearest-neighbor when time is below all depths."""
        ni, nj, nz = 1, 1, 3
        nt = 2

        twt_irregular = np.array([[[1.0, 2.0, 3.0]]])
        data_depth = np.array([[[100, 200, 300]]], dtype=np.int32)
        time_axis = np.array([0.0, 0.5])  # Both below first depth
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_nearest(twt_irregular, data_depth, time_axis, out_array)

        # Both should return first value
        assert out_array[0, 0, 0] == 100
        assert out_array[0, 0, 1] == 100

    def test_resample_depth_to_time_nearest_boundary_above(self):
        """Test nearest-neighbor when time is above all depths."""
        ni, nj, nz = 1, 1, 3
        nt = 2

        twt_irregular = np.array([[[0.1, 0.2, 0.3]]])
        data_depth = np.array([[[10, 20, 30]]], dtype=np.int32)
        time_axis = np.array([1.0, 2.0])  # Both above last depth
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_nearest(twt_irregular, data_depth, time_axis, out_array)

        # Both should return last value
        assert out_array[0, 0, 0] == 30
        assert out_array[0, 0, 1] == 30


class TestResampleDepthToTimeLinear:
    """Test linear interpolation depth-to-time resampling."""

    def test_resample_depth_to_time_linear_single_trace(self):
        """Test linear interpolation with single trace."""
        ni, nj, nz = 1, 1, 3
        nt = 3

        twt_irregular = np.array([[[0.0, 1.0, 2.0]]])
        data_depth = np.array([[[0.0, 10.0, 20.0]]])
        time_axis = np.array([0.0, 0.5, 1.0])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_linear(twt_irregular, data_depth, time_axis, out_array)

        assert out_array.shape == (ni, nj, nt)
        assert out_array[0, 0, 0] == 0.0  # t=0.0 -> v=0.0
        assert np.isclose(
            out_array[0, 0, 1], 5.0
        )  # t=0.5 -> interpolated between 0 and 10
        assert out_array[0, 0, 2] == 10.0  # t=1.0 -> v=10.0

    def test_resample_depth_to_time_linear_interpolation(self):
        """Test linear interpolation values."""
        ni, nj, nz = 1, 1, 4
        nt = 5

        # Linear function: f(t) = 2*t
        twt_irregular = np.array([[[0.0, 1.0, 2.0, 3.0]]])
        data_depth = np.array([[[0.0, 2.0, 4.0, 6.0]]])
        time_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_linear(twt_irregular, data_depth, time_axis, out_array)

        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(out_array[0, 0, :], expected)

    def test_resample_depth_to_time_linear_degenerate_interval(self):
        """Test linear interpolation with equal twt values (degenerate interval)."""
        ni, nj, nz = 1, 1, 4
        nt = 3

        # Create monotonic TWT: 0, 1, 2, 3
        twt_irregular = np.array([[[0.0, 1.0, 2.0, 3.0]]])
        data_depth = np.array([[[10.0, 20.0, 30.0, 40.0]]])
        time_axis = np.array([0.5, 1.5, 2.5])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_linear(twt_irregular, data_depth, time_axis, out_array)

        # At each time, should interpolate linearly
        assert np.isclose(out_array[0, 0, 0], 15.0)  # t=0.5 between 0 and 10
        assert np.isclose(out_array[0, 0, 1], 25.0)  # t=1.5 between 10 and 20
        assert np.isclose(out_array[0, 0, 2], 35.0)  # t=2.5 between 30 and 40

    def test_resample_depth_to_time_linear_boundary(self):
        """Test linear interpolation at boundaries."""
        ni, nj, nz = 1, 1, 3
        nt = 3

        twt_irregular = np.array([[[1.0, 2.0, 3.0]]])
        data_depth = np.array([[[100.0, 200.0, 300.0]]])
        time_axis = np.array([0.0, 1.0, 4.0])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_linear(twt_irregular, data_depth, time_axis, out_array)

        assert out_array[0, 0, 0] == 100.0  # Below first -> first value
        assert out_array[0, 0, 1] == 100.0  # At first -> first value
        assert out_array[0, 0, 2] == 300.0  # Above last -> last value


class TestResampleDepthToTimeFromIrregularNearest:
    """Test nearest-neighbor with irregular TWT (per-trace)."""

    def test_irregular_nearest_basic(self):
        """Test basic nearest-neighbor with irregular TWT."""
        ni, nj, nz = 1, 1, 3
        nt = 3

        twt_irregular = np.array([[[0.0, 0.5, 1.0]]])
        data_depth = np.array([[[10, 20, 30]]], dtype=np.int32)
        time_axis = np.array([0.0, 0.3, 0.7])
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_from_irregular_nearest(
            twt_irregular, data_depth, time_axis, out_array
        )

        assert out_array[0, 0, 0] == 10  # t=0.0 nearest 0.0
        assert out_array[0, 0, 1] == 20  # t=0.3 nearest 0.5 (dist 0.2 vs 0.3)
        assert out_array[0, 0, 2] == 20  # t=0.7 nearest 0.5 (dist 0.2 vs 0.3)

    def test_irregular_nearest_varying_twt_per_trace(self):
        """Test irregular nearest with different TWT per trace."""
        ni, nj, nz = 1, 2, 3
        nt = 2

        twt_irregular = np.array([[[0.0, 0.5, 1.0], [0.1, 0.6, 1.1]]])
        data_depth = np.array([[[10, 20, 30], [100, 200, 300]]], dtype=np.int32)
        time_axis = np.array([0.0, 0.5])
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_from_irregular_nearest(
            twt_irregular, data_depth, time_axis, out_array
        )

        # Trace 0: t=0.5 is exact match
        assert out_array[0, 0, 1] == 20
        # Trace 1: t=0.5 is between 0.1 and 0.6, nearest is 0.6
        assert out_array[0, 1, 1] == 200


class TestResampleDepthToTimeFromIrregularLinear:
    """Test linear interpolation with irregular TWT (per-trace)."""

    def test_irregular_linear_basic(self):
        """Test basic linear interpolation with irregular TWT."""
        ni, nj, nz = 1, 1, 3
        nt = 3

        twt_irregular = np.array([[[0.0, 1.0, 2.0]]])
        data_depth = np.array([[[0.0, 10.0, 20.0]]])
        time_axis = np.array([0.0, 0.5, 1.0])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_from_irregular_linear(
            twt_irregular, data_depth, time_axis, out_array
        )

        assert out_array[0, 0, 0] == 0.0
        assert np.isclose(out_array[0, 0, 1], 5.0)
        assert out_array[0, 0, 2] == 10.0

    def test_irregular_linear_varying_per_trace(self):
        """Test irregular linear with different interpolation per trace."""
        ni, nj, nz = 1, 2, 3
        nt = 2

        twt_irregular = np.array([[[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]])
        data_depth = np.array([[[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]])
        time_axis = np.array([0.5, 1.0])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_from_irregular_linear(
            twt_irregular, data_depth, time_axis, out_array
        )

        # Trace 0: t=0.5 linear between 0 and 10 -> 5.0
        assert np.isclose(out_array[0, 0, 0], 5.0)
        # Trace 1: t=0.5 linear between 0 and 10 (at twt 0 and 2) -> 2.5
        assert np.isclose(out_array[0, 1, 0], 2.5)

    def test_irregular_linear_degenerate_interval(self):
        """Test irregular linear with degenerate intervals."""
        ni, nj, nz = 1, 1, 4
        nt = 2

        twt_irregular = np.array([[[0.0, 1.0, 2.0, 3.0]]])  # No degenerate interval
        data_depth = np.array([[[10.0, 20.0, 30.0, 40.0]]])
        time_axis = np.array([1.5, 2.5])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_from_irregular_linear(
            twt_irregular, data_depth, time_axis, out_array
        )

        # Should interpolate correctly without degenerate issues
        assert np.isclose(out_array[0, 0, 0], 25.0)  # t=1.5 between 20 and 30
        assert np.isclose(out_array[0, 0, 1], 35.0)  # t=2.5 between 30 and 40


class TestKernelsLargeArrays:
    """Test kernels with larger arrays to ensure parallel efficiency."""

    def test_large_array_nearest(self):
        """Test nearest-neighbor with large array."""
        ni, nj, nz = 10, 10, 50
        nt = 100

        np.random.seed(42)
        twt_irregular = np.sort(np.random.rand(ni, nj, nz) * 10.0, axis=2)
        data_depth = np.random.randint(0, 1000, (ni, nj, nz), dtype=np.int32)
        time_axis = np.linspace(0, 10, nt)
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_nearest(twt_irregular, data_depth, time_axis, out_array)

        assert out_array.shape == (ni, nj, nt)
        assert not np.all(out_array == 0)  # Should have non-zero values

    def test_large_array_linear(self):
        """Test linear interpolation with large array."""
        ni, nj, nz = 10, 10, 50
        nt = 100

        np.random.seed(42)
        twt_irregular = np.sort(np.random.rand(ni, nj, nz) * 10.0, axis=2)
        data_depth = np.random.rand(ni, nj, nz)
        time_axis = np.linspace(0, 10, nt)
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_linear(twt_irregular, data_depth, time_axis, out_array)

        assert out_array.shape == (ni, nj, nt)
        assert not np.all(out_array == 0)


class TestKernelsDataTypes:
    """Test kernels with different data types."""

    def test_int32_data(self):
        """Test with int32 data."""
        ni, nj, nz = 1, 1, 3
        nt = 2

        twt_irregular = np.array([[[0.0, 1.0, 2.0]]])
        data_depth = np.array([[[10, 20, 30]]], dtype=np.int32)
        time_axis = np.array([0.5, 1.5])
        out_array = np.zeros((ni, nj, nt), dtype=np.int32)

        resample_depth_to_time_nearest(twt_irregular, data_depth, time_axis, out_array)
        assert out_array.dtype == np.int32

    def test_kernels_float64_data(self):
        """Test kernels with float64 data."""
        ni, nj, nz = 1, 1, 3
        nt = 2

        twt_irregular = np.array([[[0.0, 1.0, 2.0]]])
        data_depth = np.array([[[10.5, 20.5, 30.5]]], dtype=np.float64)
        time_axis = np.array([0.5, 1.5])
        out_array = np.zeros((ni, nj, nt), dtype=np.float64)

        resample_depth_to_time_linear(twt_irregular, data_depth, time_axis, out_array)
        assert out_array.dtype == np.float64
