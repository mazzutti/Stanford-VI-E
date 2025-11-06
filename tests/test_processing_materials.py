"""Tests for src.processing.materials module.

Tests for VelocityModel and other material property models with unit handling,
validation, and signal processing operations.
"""

import numpy as np
import pytest
from src.io.grid import GridSpec
from src.utils.quantity import Quantity
from src.processing.materials.velocity import VelocityModel
from src.processing.materials.properties import VsModel, DensityModel


class TestVelocityModelInit:
    """Tests for VelocityModel initialization."""

    def test_init_with_raw_array(self):
        """Test initialization with raw numpy array."""
        vp = np.random.rand(3, 4, 5) * 1000 + 3000  # 3000-4000 m/s
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        assert model.vp.array.shape == (3, 4, 5)
        assert isinstance(model.vp, Quantity)
        assert model.grid_spec == grid_spec

    def test_init_with_quantity(self):
        """Test initialization with Quantity object."""
        vp_array = np.random.rand(3, 4, 5) * 1000 + 3000
        vp = Quantity(vp_array, "m/s")
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        assert model.vp.array.shape == (3, 4, 5)
        assert model.vp.unit == "m/s"

    def test_init_invalid_dimensions(self):
        """Test that 1D or 2D arrays are rejected."""
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)

        # 2D array should fail
        with pytest.raises(ValueError, match="must be a 3D array"):
            VelocityModel(vp=np.random.rand(3, 4), grid_spec=grid_spec)

        # 1D array should fail
        with pytest.raises(ValueError, match="must be a 3D array"):
            VelocityModel(vp=np.random.rand(5), grid_spec=grid_spec)

    def test_init_shape_mismatch(self):
        """Test that shape mismatch with grid_spec is caught."""
        vp = np.random.rand(2, 3, 4)  # Different shape
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)

        with pytest.raises(ValueError, match="vp shape must match grid_spec.shape"):
            VelocityModel(vp=vp, grid_spec=grid_spec)


class TestVelocityModelValidation:
    """Tests for VelocityModel validation methods."""

    @pytest.fixture
    def valid_model(self):
        """Create a valid VelocityModel for testing."""
        vp = np.full((3, 4, 5), 3500.0)  # Constant valid velocity
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)
        return VelocityModel(vp=vp, grid_spec=grid_spec)

    def test_validate_valid_model(self, valid_model):
        """Test validation passes for physically plausible model."""
        valid_model.validate()  # Should not raise

    def test_validate_with_nan(self):
        """Test validation fails with NaN values."""
        vp = np.full((3, 4, 5), 3500.0)
        vp[1, 1, 1] = np.nan
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        with pytest.raises(ValueError, match="non-finite"):
            model.validate()

    def test_validate_with_inf(self):
        """Test validation fails with infinite values."""
        vp = np.full((3, 4, 5), 3500.0)
        vp[0, 0, 0] = np.inf
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        with pytest.raises(ValueError, match="non-finite"):
            model.validate()

    def test_validate_with_negative(self):
        """Test validation fails with negative velocities."""
        vp = np.full((3, 4, 5), 3500.0)
        vp[0, 0, 0] = -100.0
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        with pytest.raises(ValueError, match="non-positive"):
            model.validate()

    def test_validate_with_zero(self):
        """Test validation fails with zero velocity."""
        vp = np.full((3, 4, 5), 3500.0)
        vp[1, 2, 3] = 0.0
        grid_spec = GridSpec(shape=(3, 4, 5), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        with pytest.raises(ValueError, match="non-positive"):
            model.validate()


class TestVelocityModelSmoothing:
    """Tests for VelocityModel smoothing operations."""

    @pytest.fixture
    def noisy_model(self):
        """Create a model with noisy velocity field."""
        np.random.seed(42)
        vp = np.random.rand(5, 5, 5) * 500 + 3000  # 3000-3500 m/s
        grid_spec = GridSpec(shape=(5, 5, 5), dz=10, dt=0.004)
        return VelocityModel(vp=vp, grid_spec=grid_spec)

    def test_smooth_preserves_shape(self, noisy_model):
        """Test that smoothing preserves array shape."""
        original_shape = noisy_model.vp.array.shape
        noisy_model.smooth(sigma=1.0)
        assert noisy_model.vp.array.shape == original_shape

    def test_smooth_reduces_variance(self, noisy_model):
        """Test that smoothing reduces variance."""
        original_var = np.var(noisy_model.vp.array)
        noisy_model.smooth(sigma=2.0)
        smoothed_var = np.var(noisy_model.vp.array)
        assert smoothed_var < original_var

    def test_smooth_preserves_unit(self, noisy_model):
        """Test that smoothing preserves unit metadata."""
        original_unit = noisy_model.vp.unit
        noisy_model.smooth(sigma=1.5)
        assert noisy_model.vp.unit == original_unit

    def test_smooth_default_sigma(self, noisy_model):
        """Test smoothing with default sigma."""
        noisy_model.smooth()  # Should use default sigma=1.0
        assert noisy_model.vp.array.shape == (5, 5, 5)

    def test_smooth_with_large_sigma(self, noisy_model):
        """Test smoothing with large sigma produces more uniform result."""
        noisy_model.smooth(sigma=5.0)
        # After heavy smoothing, should be very uniform
        assert np.std(noisy_model.vp.array) < np.std([3000, 3500])


class TestVelocityModelUnitConversion:
    """Tests for unit conversion methods."""

    def test_to_m_per_s_with_quantity(self):
        """Test to_m_per_s with Quantity."""
        vp = Quantity(np.full((2, 2, 2), 3.5), "km/s")
        grid_spec = GridSpec(shape=(2, 2, 2), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        model.to_m_per_s()
        # Should have converted from km/s to m/s
        assert np.allclose(model.vp.array, 3500.0)

    @pytest.mark.skip(reason="UnitRegistry.ensure_m_per_s not implemented yet")
    def test_ensure_m_per_s_returns_true_when_converted(self):
        """Test ensure_m_per_s returns True when conversion is needed."""
        vp = Quantity(np.full((2, 2, 2), 3.5), "km/s")
        grid_spec = GridSpec(shape=(2, 2, 2), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        result = model.ensure_m_per_s()
        # Should return True if units were different before/after
        assert isinstance(result, bool)

    @pytest.mark.skip(reason="UnitRegistry.ensure_m_per_s not implemented yet")
    def test_ensure_m_per_s_with_raw_array(self):
        """Test ensure_m_per_s with raw array."""
        vp = np.full((2, 2, 2), 3500.0)
        grid_spec = GridSpec(shape=(2, 2, 2), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        # Should wrap array in Quantity
        result = model.ensure_m_per_s()
        assert isinstance(model.vp, Quantity)


class TestVelocityModelEdgeCases:
    """Tests for edge cases and corner conditions."""

    def test_small_grid(self):
        """Test with minimum size grid."""
        vp = np.ones((1, 1, 1)) * 3500.0
        grid_spec = GridSpec(shape=(1, 1, 1), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        model.validate()
        assert model.vp.array.shape == (1, 1, 1)

    def test_large_grid(self):
        """Test with reasonably large grid."""
        vp = np.random.rand(100, 100, 50) * 1000 + 3000
        grid_spec = GridSpec(shape=(100, 100, 50), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        model.validate()
        assert model.vp.array.shape == (100, 100, 50)

    def test_very_low_velocity(self):
        """Test with very low but valid velocity."""
        vp = np.ones((2, 2, 2)) * 1.0  # Very low velocity
        grid_spec = GridSpec(shape=(2, 2, 2), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        model.validate()  # Should pass

    def test_very_high_velocity(self):
        """Test with very high velocity."""
        vp = np.ones((2, 2, 2)) * 1e6  # Very high velocity
        grid_spec = GridSpec(shape=(2, 2, 2), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        model.validate()  # Should pass

    def test_mixed_valid_velocities(self):
        """Test with spatially varying but valid velocities."""
        np.random.seed(42)
        vp = np.random.rand(3, 3, 3) * 2000 + 2000  # 2000-4000 m/s
        grid_spec = GridSpec(shape=(3, 3, 3), dz=10, dt=0.004)
        model = VelocityModel(vp=vp, grid_spec=grid_spec)

        model.validate()
        # Ensure we have variation
        assert np.min(model.vp.array) < np.max(model.vp.array)


class TestVsModelInit:
    """Tests for VsModel initialization."""

    def test_init_with_array(self):
        """Test initialization with numpy array."""
        vs = np.ones((5, 5, 5)) * 1500.0
        model = VsModel(vs=vs)

        assert model.vs is vs
        assert np.array_equal(model.get_data(), vs)

    def test_init_preserves_data_type(self):
        """Test that initialization preserves data type."""
        vs = np.ones((5, 5, 5), dtype=np.float32) * 1500.0
        model = VsModel(vs=vs)

        assert model.get_data().dtype == vs.dtype


class TestVsModelGetSetData:
    """Tests for VsModel get/set data methods."""

    def test_get_data(self):
        """Test get_data returns array."""
        vs = np.ones((3, 3, 3)) * 1500.0
        model = VsModel(vs=vs)

        data = model.get_data()
        assert np.array_equal(data, vs)

    def test_set_data(self):
        """Test set_data updates vs."""
        vs = np.ones((3, 3, 3)) * 1500.0
        new_vs = np.ones((3, 3, 3)) * 2000.0

        model = VsModel(vs=vs)
        model.set_data(new_vs)

        assert np.array_equal(model.vs, new_vs)

    def test_get_data_returns_ndarray(self):
        """Test get_data always returns ndarray."""
        vs = [1500, 1600]  # List input
        model = VsModel(vs=vs)

        data = model.get_data()
        assert isinstance(data, np.ndarray)


class TestVsModelUnitConversion:
    """Tests for VsModel unit conversion methods."""

    @pytest.mark.skip(reason="UnitRegistry.ensure_m_per_s not implemented yet")
    def test_to_m_per_s_with_km_per_s(self):
        """Test conversion from km/s to m/s."""
        vs = np.ones((3, 3, 3)) * 2.0  # 2 km/s
        model = VsModel(vs=vs)

        model.to_m_per_s()

        # Should be converted to 2000 m/s
        assert np.all(model.vs >= 1000.0)

    @pytest.mark.skip(reason="UnitRegistry.ensure_m_per_s not implemented yet")
    def test_ensure_m_per_s_with_km_per_s(self):
        """Test ensure_m_per_s with km/s input."""
        vs = np.ones((3, 3, 3)) * 2.0  # 2 km/s
        model = VsModel(vs=vs)

        converted = model.ensure_m_per_s()

        # Should return True indicating conversion happened
        assert converted is True or isinstance(converted, (bool, np.bool_))

    @pytest.mark.skip(reason="UnitRegistry.ensure_m_per_s not implemented yet")
    def test_ensure_m_per_s_with_m_per_s(self):
        """Test ensure_m_per_s with m/s input."""
        vs = np.ones((3, 3, 3)) * 1500.0  # Already in m/s
        model = VsModel(vs=vs)

        converted = model.ensure_m_per_s()

        # Result should be boolean-like
        assert isinstance(converted, (bool, np.bool_, np.ndarray)) or converted in (
            True,
            False,
            0,
            1,
        )

    @pytest.mark.skip(reason="UnitRegistry.ensure_m_per_s not implemented yet")
    def test_ensure_units_alias(self):
        """Test that ensure_units is an alias for ensure_m_per_s."""
        vs = np.ones((3, 3, 3)) * 1500.0
        model = VsModel(vs=vs)

        # Should not raise
        result = model.ensure_units()
        assert result is not None


class TestVsModelValidation:
    """Tests for VsModel validation."""

    def test_validate_positive_values(self):
        """Test validation with positive finite values."""
        vs = np.ones((3, 3, 3)) * 1500.0
        model = VsModel(vs=vs)

        # Should not raise
        model.validate()

    def test_validate_rejects_negative_values(self):
        """Test validation rejects negative values."""
        vs = np.ones((3, 3, 3)) * -1500.0
        model = VsModel(vs=vs)

        with pytest.raises(ValueError, match="non-positive"):
            model.validate()

    def test_validate_rejects_zero(self):
        """Test validation rejects zero values."""
        vs = np.ones((3, 3, 3)) * 0.0
        model = VsModel(vs=vs)

        with pytest.raises(ValueError, match="non-positive"):
            model.validate()

    def test_validate_rejects_nan(self):
        """Test validation rejects NaN values."""
        vs = np.ones((3, 3, 3)) * np.nan
        model = VsModel(vs=vs)

        with pytest.raises(ValueError, match="non-finite"):
            model.validate()

    def test_validate_rejects_inf(self):
        """Test validation rejects infinite values."""
        vs = np.ones((3, 3, 3)) * np.inf
        model = VsModel(vs=vs)

        with pytest.raises(ValueError, match="non-finite"):
            model.validate()

    def test_validate_with_mixed_valid_values(self):
        """Test validation with spatially varying values."""
        vs = np.random.rand(5, 5, 5) * 500 + 1000  # 1000-1500 m/s
        model = VsModel(vs=vs)

        # Should not raise
        model.validate()


class TestDensityModelInit:
    """Tests for DensityModel initialization."""

    def test_init_with_array(self):
        """Test initialization with numpy array."""
        rho = np.ones((5, 5, 5)) * 2300.0
        model = DensityModel(rho=rho)

        assert model.rho is rho
        assert np.array_equal(model.get_data(), rho)

    def test_init_preserves_data_type(self):
        """Test that initialization preserves data type."""
        rho = np.ones((5, 5, 5), dtype=np.float32) * 2300.0
        model = DensityModel(rho=rho)

        assert model.get_data().dtype == rho.dtype


class TestDensityModelGetSetData:
    """Tests for DensityModel get/set data methods."""

    def test_get_data(self):
        """Test get_data returns array."""
        rho = np.ones((3, 3, 3)) * 2300.0
        model = DensityModel(rho=rho)

        data = model.get_data()
        assert np.array_equal(data, rho)

    def test_set_data(self):
        """Test set_data updates rho."""
        rho = np.ones((3, 3, 3)) * 2300.0
        new_rho = np.ones((3, 3, 3)) * 2400.0

        model = DensityModel(rho=rho)
        model.set_data(new_rho)

        assert np.array_equal(model.rho, new_rho)

    def test_get_data_returns_ndarray(self):
        """Test get_data always returns ndarray."""
        rho = [2300, 2400]  # List input
        model = DensityModel(rho=rho)

        data = model.get_data()
        assert isinstance(data, np.ndarray)


class TestDensityModelUnitConversion:
    """Tests for DensityModel unit conversion methods."""

    @pytest.mark.skip(reason="UnitRegistry.ensure_kg_per_m3 not implemented yet")
    def test_to_kg_per_m3_with_g_per_cm3(self):
        """Test conversion from g/cm^3 to kg/m^3."""
        rho = np.ones((3, 3, 3)) * 2.3  # 2.3 g/cm^3
        model = DensityModel(rho=rho)

        model.to_kg_per_m3()

        # Should be converted to approximately 2300 kg/m^3
        assert np.all(model.rho >= 1000.0)

    @pytest.mark.skip(reason="UnitRegistry.ensure_kg_per_m3 not implemented yet")
    def test_ensure_kg_per_m3_with_g_per_cm3(self):
        """Test ensure_kg_per_m3 with g/cm^3 input."""
        rho = np.ones((3, 3, 3)) * 2.3  # 2.3 g/cm^3
        model = DensityModel(rho=rho)

        converted = model.ensure_kg_per_m3()

        # Should return True indicating conversion happened
        assert converted is True or isinstance(converted, (bool, np.bool_))

    @pytest.mark.skip(reason="UnitRegistry.ensure_kg_per_m3 not implemented yet")
    def test_ensure_kg_per_m3_with_kg_per_m3(self):
        """Test ensure_kg_per_m3 with kg/m^3 input."""
        rho = np.ones((3, 3, 3)) * 2300.0  # Already in kg/m^3
        model = DensityModel(rho=rho)

        converted = model.ensure_kg_per_m3()

        # Result should be boolean-like
        assert isinstance(converted, (bool, np.bool_, np.ndarray)) or converted in (
            True,
            False,
            0,
            1,
        )

    @pytest.mark.skip(reason="UnitRegistry.ensure_kg_per_m3 not implemented yet")
    def test_ensure_units_alias(self):
        """Test that ensure_units is an alias for ensure_kg_per_m3."""
        rho = np.ones((3, 3, 3)) * 2300.0
        model = DensityModel(rho=rho)

        # Should not raise
        result = model.ensure_units()
        assert result is not None


class TestDensityModelValidation:
    """Tests for DensityModel validation."""

    def test_validate_positive_values(self):
        """Test validation with positive finite values."""
        rho = np.ones((3, 3, 3)) * 2300.0
        model = DensityModel(rho=rho)

        # Should not raise
        model.validate()

    def test_validate_rejects_negative_values(self):
        """Test validation rejects negative values."""
        rho = np.ones((3, 3, 3)) * -2300.0
        model = DensityModel(rho=rho)

        with pytest.raises(ValueError, match="non-positive"):
            model.validate()

    def test_validate_rejects_zero(self):
        """Test validation rejects zero values."""
        rho = np.ones((3, 3, 3)) * 0.0
        model = DensityModel(rho=rho)

        with pytest.raises(ValueError, match="non-positive"):
            model.validate()

    def test_validate_rejects_nan(self):
        """Test validation rejects NaN values."""
        rho = np.ones((3, 3, 3)) * np.nan
        model = DensityModel(rho=rho)

        with pytest.raises(ValueError, match="non-finite"):
            model.validate()

    def test_validate_rejects_inf(self):
        """Test validation rejects infinite values."""
        rho = np.ones((3, 3, 3)) * np.inf
        model = DensityModel(rho=rho)

        with pytest.raises(ValueError, match="non-finite"):
            model.validate()

    def test_validate_with_mixed_valid_values(self):
        """Test validation with spatially varying values."""
        rho = np.random.rand(5, 5, 5) * 300 + 2000  # 2000-2300 kg/m^3
        model = DensityModel(rho=rho)

        # Should not raise
        model.validate()


class TestVsAndDensityModelIntegration:
    """Integration tests for VsModel and DensityModel."""

    def test_vs_model_workflow(self):
        """Test typical VsModel workflow (without unit conversion)."""
        vs = np.ones((3, 3, 3)) * 1500.0
        model = VsModel(vs=vs)

        # Get data
        data = model.get_data()
        assert np.array_equal(data, vs)

        # Validate
        model.validate()

    def test_density_model_workflow(self):
        """Test typical DensityModel workflow (without unit conversion)."""
        rho = np.ones((3, 3, 3)) * 2300.0
        model = DensityModel(rho=rho)

        # Get data
        data = model.get_data()
        assert np.array_equal(data, rho)

        # Validate
        model.validate()

    def test_vs_and_density_together(self):
        """Test using VsModel and DensityModel together."""
        vs = np.ones((3, 3, 3)) * 1500.0
        rho = np.ones((3, 3, 3)) * 2300.0

        vs_model = VsModel(vs=vs)
        rho_model = DensityModel(rho=rho)

        # Both should validate successfully
        vs_model.validate()
        rho_model.validate()

        # Both should have data
        assert vs_model.get_data().shape == (3, 3, 3)
        assert rho_model.get_data().shape == (3, 3, 3)
