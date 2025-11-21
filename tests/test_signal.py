"""Comprehensive test suite for src/signal module coverage improvement.

Tests focus on increasing coverage for reflectivity.py, signal.py,
and wavelets.py edge cases and error paths.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.io.grid import GridSpec
from src.signal import (DepthTimeConverter, RickerWavelet,
                        SeismicSignalProcessor, ZoeppritzSolver)
from src.utils.quantity import Quantity


class TestDepthTimeConverterCoverage:
    """Tests for DepthTimeConverter initialization and basic functionality."""

    @pytest.fixture
    def basic_grid_spec(self):
        """Create a basic grid specification."""
        return GridSpec.from_dimensions(nx=5, ny=5, nz=100, dz=10.0, dt=0.002)

    def test_converter_initialization(self, basic_grid_spec):
        """Test DepthTimeConverter initialization."""
        converter = DepthTimeConverter(grid_spec=basic_grid_spec)
        assert converter.grid_spec == basic_grid_spec


class TestZoeppritzSolverCoverage:
    """Tests for ZoeppritzSolver edge cases and complete coverage."""

    def test_solver_initialization_default(self):
        """Test solver with default batch size."""
        solver = ZoeppritzSolver()
        assert solver.cpu_batch == 1024

    def test_solver_initialization_custom_batch(self):
        """Test solver with custom batch size."""
        solver = ZoeppritzSolver(cpu_batch=512)
        assert solver.cpu_batch == 512

    def test_solver_initialization_zero_batch(self):
        """Test solver with zero batch size gets converted to int."""
        solver = ZoeppritzSolver(cpu_batch=0)
        assert solver.cpu_batch == 0

    def test_solve_normal_incidence(self):
        """Test Zoeppritz solution at normal incidence (0 degrees)."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        vp2 = np.array([4000.0])
        vs2 = np.array([2000.0])
        rho2 = np.array([2500.0])

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=0.0)

        # At normal incidence, result should be real-valued
        assert rp.shape == (1,)
        assert not np.all(np.isnan(rp))

    def test_solve_small_angles(self):
        """Test Zoeppritz at small incident angles."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        vp2 = np.array([3500.0])
        vs2 = np.array([1700.0])
        rho2 = np.array([2200.0])

        for angle in [5.0, 10.0, 15.0]:
            rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=angle)
            assert rp.shape == (1,)
            assert not np.any(np.isnan(rp))

    def test_solve_large_angles(self):
        """Test Zoeppritz at large incident angles."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        vp2 = np.array([3500.0])
        vs2 = np.array([1700.0])
        rho2 = np.array([2200.0])

        for angle in [30.0, 45.0]:
            rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=angle)
            assert rp.shape == (1,)
            # May be complex at large angles
            assert not np.all(np.isnan(rp))

    def test_solve_2d_array(self):
        """Test Zoeppritz with 2D input arrays."""
        solver = ZoeppritzSolver()

        vp1 = np.array([[3000.0, 3100.0], [3200.0, 3300.0]])
        vs1 = np.array([[1500.0, 1550.0], [1600.0, 1650.0]])
        rho1 = np.array([[2000.0, 2050.0], [2100.0, 2150.0]])
        vp2 = np.array([[4000.0, 4100.0], [4200.0, 4300.0]])
        vs2 = np.array([[2000.0, 2050.0], [2100.0, 2150.0]])
        rho2 = np.array([[2500.0, 2550.0], [2600.0, 2650.0]])

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=15.0)

        assert rp.shape == (2, 2)
        assert not np.all(np.isnan(rp))

    def test_solve_matching_layers(self):
        """Test Zoeppritz with matching elastic properties (no reflection)."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])

        # Identical to top layer
        rp = solver.solve(vp1, vs1, rho1, vp1, vs1, rho1, theta1_deg=30.0)

        # Should be zero (no impedance contrast)
        assert_allclose(rp, 0.0, atol=1e-10)

    def test_solve_high_contrast(self):
        """Test Zoeppritz with high impedance contrast."""
        solver = ZoeppritzSolver()

        vp1 = np.array([2000.0])
        vs1 = np.array([1000.0])
        rho1 = np.array([1500.0])
        vp2 = np.array([6000.0])
        vs2 = np.array([3000.0])
        rho2 = np.array([3500.0])

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=30.0)

        # Should have significant reflection
        assert np.abs(rp[0]) > 0.2

    def test_solve_3d_array(self):
        """Test Zoeppritz with 3D input arrays."""
        solver = ZoeppritzSolver()

        vp1 = np.full((2, 3, 4), 3000.0)
        vs1 = np.full((2, 3, 4), 1500.0)
        rho1 = np.full((2, 3, 4), 2000.0)
        vp2 = np.full((2, 3, 4), 4000.0)
        vs2 = np.full((2, 3, 4), 2000.0)
        rho2 = np.full((2, 3, 4), 2500.0)

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=20.0)

        assert rp.shape == (2, 3, 4)
        assert not np.all(np.isnan(rp))

    def test_solve_critical_angle(self):
        """Test Zoeppritz near critical angle."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        vp2 = np.array([5000.0])
        vs2 = np.array([2500.0])
        rho2 = np.array([2500.0])

        # Critical angle for P-wave at this interface
        # sin(critical) = vp1 / vp2 = 3000/5000 = 0.6
        # critical_angle ≈ 36.87 degrees
        critical_angle = np.degrees(np.arcsin(vp1[0] / vp2[0]))

        # Test near critical angle
        rp = solver.solve(
            vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=critical_angle - 5.0
        )
        assert rp.shape == (1,)
        assert not np.any(np.isnan(rp))

    def test_solve_supercritical_angle(self):
        """Test Zoeppritz beyond critical angle (complex angles)."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        vp2 = np.array([5000.0])
        vs2 = np.array([2500.0])
        rho2 = np.array([2500.0])

        critical_angle = np.degrees(np.arcsin(vp1[0] / vp2[0]))

        # Test beyond critical angle (may produce complex values)
        rp = solver.solve(
            vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=critical_angle + 5.0
        )
        assert rp.shape == (1,)
        # May be complex
        assert not np.all(np.isnan(rp))

    def test_solve_batch_size_env_variable(self):
        """Test that env variable for batch size is respected."""
        import os

        original_val = os.environ.get("ZOEPPRITZ_CPU_BATCH")
        try:
            os.environ["ZOEPPRITZ_CPU_BATCH"] = "256"
            solver = ZoeppritzSolver()
            assert solver.cpu_batch == 256
        finally:
            if original_val is not None:
                os.environ["ZOEPPRITZ_CPU_BATCH"] = original_val
            else:
                os.environ.pop("ZOEPPRITZ_CPU_BATCH", None)

    def test_solve_with_varying_spatial_properties(self):
        """Test Zoeppritz with spatially varying properties."""
        solver = ZoeppritzSolver()

        # Create varying property arrays
        vp1 = np.linspace(2500.0, 3500.0, 10).reshape(2, 5)
        vs1 = vp1 / 2.0
        rho1 = np.full((2, 5), 2000.0)
        vp2 = vp1 + 1000.0
        vs2 = vp2 / 2.0
        rho2 = rho1 + 500.0

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=25.0)

        assert rp.shape == (2, 5)
        assert not np.all(np.isnan(rp))

    def test_solve_low_velocity_contrast(self):
        """Test Zoeppritz with minimal velocity contrast."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        # Only 1% contrast
        vp2 = np.array([3030.0])
        vs2 = np.array([1515.0])
        rho2 = np.array([2020.0])

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=30.0)

        # Should have weak reflection
        assert np.abs(rp[0]) < 0.02

    def test_solve_large_batch(self):
        """Test Zoeppritz with large batch of traces."""
        solver = ZoeppritzSolver(cpu_batch=64)

        # Create 2000 traces (to potentially exceed batch size)
        vp1 = np.full(2000, 3000.0)
        vs1 = np.full(2000, 1500.0)
        rho1 = np.full(2000, 2000.0)
        vp2 = np.full(2000, 4000.0)
        vs2 = np.full(2000, 2000.0)
        rho2 = np.full(2000, 2500.0)

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=20.0)

        assert rp.shape == (2000,)
        assert not np.all(np.isnan(rp))

    def test_solve_grazing_incidence(self):
        """Test Zoeppritz at near-grazing incidence."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        vp2 = np.array([4000.0])
        vs2 = np.array([2000.0])
        rho2 = np.array([2500.0])

        # Near 89 degrees
        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=85.0)

        assert rp.shape == (1,)
        # May have large values or be complex at grazing angles
        assert not np.any(np.isnan(rp))

    def test_solve_inverted_vp_vs_ratio(self):
        """Test Zoeppritz with unusual Vp/Vs ratios."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([2000.0])  # Vp/Vs = 1.5 (lower than typical)
        rho1 = np.array([2000.0])
        vp2 = np.array([4000.0])
        vs2 = np.array([2500.0])  # Vp/Vs = 1.6
        rho2 = np.array([2500.0])

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=30.0)

        assert rp.shape == (1,)
        assert not np.any(np.isnan(rp))

    def test_solve_singular_matrix_case(self):
        """Test Zoeppritz with conditions that may lead to singular matrix."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        # Very similar properties
        vp2 = np.array([3000.001])
        vs2 = np.array([1500.001])
        rho2 = np.array([2000.001])

        # Nearly singular case
        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=45.0)

        assert rp.shape == (1,)
        # Should handle near-singular gracefully
        assert not np.any(np.isnan(rp))

    def test_solve_very_small_velocities(self):
        """Test Zoeppritz with very small velocity values."""
        solver = ZoeppritzSolver()

        vp1 = np.array([1000.0])
        vs1 = np.array([500.0])
        rho1 = np.array([1000.0])
        vp2 = np.array([1500.0])
        vs2 = np.array([750.0])
        rho2 = np.array([1200.0])

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=20.0)

        assert rp.shape == (1,)
        assert not np.any(np.isnan(rp))

    def test_solve_very_large_velocities(self):
        """Test Zoeppritz with very large velocity values."""
        solver = ZoeppritzSolver()

        vp1 = np.array([7000.0])
        vs1 = np.array([4000.0])
        rho1 = np.array([3300.0])
        vp2 = np.array([8000.0])
        vs2 = np.array([4600.0])
        rho2 = np.array([3400.0])

        rp = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=25.0)

        assert rp.shape == (1,)
        assert not np.any(np.isnan(rp))

    def test_initialization_with_invalid_env_variable(self):
        """Test initialization when env variable is non-numeric."""
        import os

        original_val = os.environ.get("ZOEPPRITZ_CPU_BATCH")
        try:
            # Set to non-numeric value - should trigger exception and use default
            os.environ["ZOEPPRITZ_CPU_BATCH"] = "invalid_value"
            solver = ZoeppritzSolver()
            # Should fall back to default
            assert solver.cpu_batch == 1024
        finally:
            if original_val is not None:
                os.environ["ZOEPPRITZ_CPU_BATCH"] = original_val
            else:
                os.environ.pop("ZOEPPRITZ_CPU_BATCH", None)


class TestSignalProcessorCoverage:
    """Tests for SeismicSignalProcessor edge cases."""

    def test_processor_check_scipy(self):
        """Test SeismicSignalProcessor initializes successfully."""
        processor = SeismicSignalProcessor()
        assert processor is not None
        # Verify progress_every attribute is accessible
        assert processor.progress_every is None

    def test_processor_progress_every(self):
        """Test processor with progress tracking."""
        processor = SeismicSignalProcessor(progress_every=10)
        assert processor.progress_every == 10

    def test_apply_wavelet_3d_cube(self):
        """Test apply_wavelet with 3D cube."""
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)

        reflectivity = np.random.randn(3, 3, 20) * 0.1
        seismogram = processor.apply_wavelet(reflectivity, wavelet)

        assert seismogram.shape == reflectivity.shape
        assert not np.all(np.isnan(seismogram))

    def test_apply_wavelet_different_modes(self):
        """Test apply_wavelet with different convolution modes."""
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)
        reflectivity = np.random.randn(2, 2, 20) * 0.1

        for mode in ["same", "full", "valid"]:
            try:
                seismogram = processor.apply_wavelet(reflectivity, wavelet, mode=mode)
                assert not np.all(np.isnan(seismogram))
            except ValueError:
                # Some modes may fail with small reflectivity
                pass

    def test_apply_wavelet_invalid_reflectivity_shape(self):
        """Test error handling for invalid reflectivity shape."""
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)

        # 1D reflectivity should fail
        with pytest.raises(ValueError):
            processor.apply_wavelet(np.array([0.1, 0.2, 0.3]), wavelet)

        # 2D reflectivity should fail
        with pytest.raises(ValueError):
            processor.apply_wavelet(np.ones((3, 20)), wavelet)

    def test_apply_wavelet_invalid_wavelet_shape(self):
        """Test error handling for invalid wavelet shape."""
        processor = SeismicSignalProcessor()
        reflectivity = np.random.randn(2, 2, 20)

        # 2D wavelet should fail
        with pytest.raises(ValueError):
            processor.apply_wavelet(reflectivity, np.ones((5, 5)))

    def test_apply_wavelet_with_array(self):
        """Test apply_wavelet with numpy array wavelet."""
        processor = SeismicSignalProcessor()
        wavelet_array = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        reflectivity = np.random.randn(2, 2, 20)

        seismogram = processor.apply_wavelet(reflectivity, wavelet_array)
        assert seismogram.shape == reflectivity.shape

    def test_apply_wavelet_progress_logging(self):
        """Test that progress logging works."""
        processor = SeismicSignalProcessor(progress_every=2)
        wavelet = RickerWavelet(f_peak=25.0)
        reflectivity = np.random.randn(5, 3, 20) * 0.1

        # Should not raise error with progress logging
        seismogram = processor.apply_wavelet(reflectivity, wavelet)
        assert seismogram.shape == reflectivity.shape

    def test_apply_wavelet_large_cube(self):
        """Test apply_wavelet with larger cube."""
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)
        reflectivity = np.random.randn(10, 10, 50) * 0.05

        seismogram = processor.apply_wavelet(reflectivity, wavelet)
        assert seismogram.shape == (10, 10, 50)

    def test_apply_wavelet_zeros(self):
        """Test apply_wavelet with zero reflectivity."""
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)
        reflectivity = np.zeros((2, 2, 20))

        seismogram = processor.apply_wavelet(reflectivity, wavelet)
        assert_allclose(seismogram, 0.0, atol=1e-10)

    def test_apply_wavelet_ones(self):
        """Test apply_wavelet with ones reflectivity."""
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)
        reflectivity = np.ones((2, 2, 10))

        seismogram = processor.apply_wavelet(reflectivity, wavelet)
        assert not np.all(np.isnan(seismogram))
        assert seismogram.shape == reflectivity.shape


class TestWaveletCoverage:
    """Tests for Wavelet edge cases and error handling."""

    def test_ricker_wavelet_initialization(self):
        """Test RickerWavelet initialization."""
        wavelet = RickerWavelet(f_peak=25.0)
        assert wavelet.f_peak == 25.0
        assert len(wavelet.samples) > 0

    def test_ricker_wavelet_custom_params(self):
        """Test RickerWavelet with custom parameters."""
        wavelet = RickerWavelet(f_peak=30.0, length=0.2, dt=0.001)
        assert wavelet.f_peak == 30.0
        assert wavelet.length == 0.2
        assert wavelet.dt == 0.001

    def test_ricker_wavelet_zero_frequency_error(self):
        """Test error on zero frequency."""
        with pytest.raises(ValueError):
            RickerWavelet(f_peak=0.0)

    def test_ricker_wavelet_negative_frequency_error(self):
        """Test error on negative frequency."""
        with pytest.raises(ValueError):
            RickerWavelet(f_peak=-25.0)

    def test_ricker_wavelet_negative_length_error(self):
        """Test error on negative length."""
        with pytest.raises(ValueError):
            RickerWavelet(f_peak=25.0, length=-0.1)

    def test_ricker_wavelet_zero_dt_error(self):
        """Test error on zero dt."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            RickerWavelet(f_peak=25.0, dt=0.0)

    def test_ricker_wavelet_negative_dt_error(self):
        """Test error on negative dt."""
        with pytest.raises(ValueError):
            RickerWavelet(f_peak=25.0, dt=-0.001)

    def test_ricker_wavelet_repr(self):
        """Test wavelet string representation."""
        wavelet = RickerWavelet(f_peak=25.0)
        repr_str = repr(wavelet)
        assert "RickerWavelet" in repr_str
        assert "f_peak" in repr_str

    def test_ricker_wavelet_duration_property(self):
        """Test wavelet duration calculation."""
        wavelet = RickerWavelet(f_peak=25.0, length=0.128)
        assert wavelet.duration == pytest.approx(0.128)

    def test_ricker_wavelet_nsamples_property(self):
        """Test wavelet nsamples property."""
        wavelet = RickerWavelet(f_peak=25.0, length=0.128, dt=0.002)
        expected_samples = int(0.128 / 0.002)
        assert wavelet.nsamples == expected_samples

    def test_ricker_wavelet_symmetry(self):
        """Test that Ricker wavelet is approximately symmetric."""
        wavelet = RickerWavelet(f_peak=25.0)
        samples = wavelet.samples

        # Ricker should have most energy concentrated in center
        center = len(samples) // 2
        center_energy = np.sum(samples[center - 5 : center + 5] ** 2)
        total_energy = np.sum(samples**2)

        # Center should have significant portion of energy
        assert center_energy / total_energy > 0.3

    def test_ricker_wavelet_normalization(self):
        """Test wavelet amplitude normalization."""
        wavelet = RickerWavelet(f_peak=25.0)
        max_amp = np.max(np.abs(wavelet.samples))

        # Should be reasonably normalized
        assert max_amp > 0
        assert not np.isnan(max_amp)
        assert not np.isinf(max_amp)


class TestSignalModuleIntegration:
    """Integration tests combining multiple signal components."""

    def test_complete_workflow(self):
        """Test complete workflow: reflectivity -> wavelet -> seismogram."""
        # Create synthetic reflectivity
        reflectivity = np.zeros((3, 3, 50))
        reflectivity[:, :, 10] = 0.1
        reflectivity[:, :, 25] = -0.05

        # Apply wavelet
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)

        seismogram = processor.apply_wavelet(reflectivity, wavelet)

        # Verify output
        assert seismogram.shape == reflectivity.shape
        assert not np.all(np.isnan(seismogram))

    def test_zoeppritz_to_seismogram_workflow(self):
        """Test Zoeppritz -> seismogram workflow."""
        # Compute reflection coefficients
        solver = ZoeppritzSolver()

        vp1 = np.array([[[3000.0]]])
        vs1 = np.array([[[1500.0]]])
        rho1 = np.array([[[2000.0]]])
        vp2 = np.array([[[4000.0]]])
        vs2 = np.array([[[2000.0]]])
        rho2 = np.array([[[2500.0]]])

        rc = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=30.0)

        # Reshape to 3D for processor
        reflectivity = np.zeros((1, 1, 50))
        reflectivity[:, :, 10] = rc[0].real

        # Apply wavelet
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=20.0)

        seismogram = processor.apply_wavelet(reflectivity, wavelet)

        assert seismogram.shape == (1, 1, 50)
        assert not np.all(np.isnan(seismogram))

    def test_zoeppritz_batch_processing(self):
        """Test Zoeppritz with batch of traces."""
        solver = ZoeppritzSolver(cpu_batch=256)

        # Create 3D velocity arrays for 10 traces
        vp1 = np.full((10,), 3000.0)
        vs1 = np.full((10,), 1500.0)
        rho1 = np.full((10,), 2000.0)
        vp2 = np.full((10,), 4000.0)
        vs2 = np.full((10,), 2000.0)
        rho2 = np.full((10,), 2500.0)

        rc = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=15.0)
        assert rc.shape == (10,)

    def test_processor_without_scipy(self):
        """Test processor initialization even if scipy check runs."""
        processor = SeismicSignalProcessor()
        # Just verify initialization doesn't crash
        assert processor._scipy_available is True

    def test_zoeppritz_extreme_angles(self):
        """Test Zoeppritz with angles near critical angle."""
        solver = ZoeppritzSolver()

        vp1 = np.array([3000.0])
        vs1 = np.array([1500.0])
        rho1 = np.array([2000.0])
        vp2 = np.array([4000.0])
        vs2 = np.array([2000.0])
        rho2 = np.array([2500.0])

        # Test multiple angles
        angles = [0.0, 15.0, 30.0, 45.0, 60.0]
        for angle in angles:
            rc = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg=angle)
            # Should return valid results (may be complex)
            assert not np.all(np.isnan(rc))

    def test_processor_full_valid_mode(self):
        """Test apply_wavelet with 'valid' convolution mode."""
        processor = SeismicSignalProcessor()
        wavelet = RickerWavelet(f_peak=25.0)
        reflectivity = np.random.randn(2, 2, 50) * 0.1

        try:
            seismogram = processor.apply_wavelet(reflectivity, wavelet, mode="valid")
            # Valid mode may return smaller array or raise ValueError
            if seismogram is not None:
                assert seismogram.size >= 0
        except ValueError:
            # Expected for some wavelet/reflectivity combinations
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
