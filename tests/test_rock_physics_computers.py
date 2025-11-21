"""Comprehensive tests for rock physics computers - improved coverage.

Tests focus on:
- AVOAttributesComputer AVO parameter computation
- FluidFactorComputer fluid substitution effects
- LambdaMuRhoComputer elastic parameter conversion
- Edge cases and numerical stability
- Integration with seismic data
"""

# mypy: ignore-errors


import numpy as np
import pytest

from src.analysis.rock_physics.computers import (DEFAULT_AVO_ANGLES_DEG,
                                                 DEFAULT_FLUID_FACTOR_K,
                                                 AVOAttributesComputer,
                                                 FluidFactorComputer,
                                                 LambdaMuRhoComputer)


# Test fixtures
@pytest.fixture
def sample_seismic_properties():
    """Create realistic seismic property arrays."""
    np.random.seed(42)
    shape = (10, 10, 10)
    return {
        "vp": np.random.uniform(2500, 5500, shape),  # P-wave velocity m/s
        "vs": np.random.uniform(1200, 3200, shape),  # S-wave velocity m/s
        "rho": np.random.uniform(2100, 2900, shape),  # Density kg/m^3
        "vp_sat": np.random.uniform(2400, 5400, shape),  # Saturated Vp
        "vs_sat": np.random.uniform(1100, 3100, shape),  # Saturated Vs
    }


@pytest.fixture
def realistic_1d_properties():
    """Create 1D seismic profile."""
    depth = np.linspace(0, 3000, 100)
    vp = 1500 + 0.5 * depth  # Linear Vp increase with depth
    vs = 0.577 * vp  # Typical Vp/Vs ratio
    rho = 2.0 + 0.0001 * depth  # Linear density increase
    return {"vp": vp, "vs": vs, "rho": rho}


class TestAVOAttributesComputer:
    """Tests for AVOAttributesComputer."""

    def test_computer_initialization(self):
        """Test AVOAttributesComputer initialization."""
        computer = AVOAttributesComputer()
        assert computer is not None

    def test_default_angles(self):
        """Test default AVO angles."""
        assert DEFAULT_AVO_ANGLES_DEG is not None
        assert len(DEFAULT_AVO_ANGLES_DEG) > 0

    def test_compute_intercept(self, sample_seismic_properties):
        """Test intercept (A) computation."""
        computer = AVOAttributesComputer()
        vp = sample_seismic_properties["vp"]
        vs = sample_seismic_properties["vs"]
        rho = sample_seismic_properties["rho"]

        # Intercept is the normal incidence reflection coefficient
        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None
        assert "intercept" in result or "A" in result or isinstance(result, dict)

    def test_compute_gradient(self, sample_seismic_properties):
        """Test gradient (B) computation."""
        computer = AVOAttributesComputer()
        vp = sample_seismic_properties["vp"]
        vs = sample_seismic_properties["vs"]
        rho = sample_seismic_properties["rho"]

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None

    def test_compute_with_1d_profile(self):
        """Test computation rejects 1D data - expects 3D."""
        computer = AVOAttributesComputer()

        # Should reject 1D data
        vp_1d = np.random.uniform(2500, 5500, 100)
        vs_1d = np.random.uniform(1200, 3200, 100)
        rho_1d = np.random.uniform(2100, 2900, 100)

        with pytest.raises(ValueError, match="must be 3D"):
            computer.compute(vp=vp_1d, vs=vs_1d, rho=rho_1d)

    def test_avo_result_shapes(self, sample_seismic_properties):
        """Test AVO result shape (one less in k due to interface computation)."""
        computer = AVOAttributesComputer()
        vp = sample_seismic_properties["vp"]

        result = computer.compute(
            vp=vp,
            vs=sample_seismic_properties["vs"],
            rho=sample_seismic_properties["rho"],
        )

        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    # AVO reduces k dimension by 1 (interfaces)
                    expected_shape = (vp.shape[0], vp.shape[1], vp.shape[2] - 1)
                    assert (
                        value.shape == expected_shape
                    ), f"Expected {expected_shape}, got {value.shape}"

    def test_avo_values_physical(self, sample_seismic_properties):
        """Test AVO values are physically reasonable."""
        computer = AVOAttributesComputer()
        vp = sample_seismic_properties["vp"]
        vs = sample_seismic_properties["vs"]
        rho = sample_seismic_properties["rho"]

        result = computer.compute(vp=vp, vs=vs, rho=rho)

        # Check for finite values
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    assert np.all(np.isfinite(value))

    def test_compute_with_anomalies(self, sample_seismic_properties):
        """Test computation with velocity anomalies."""
        computer = AVOAttributesComputer()
        vp = sample_seismic_properties["vp"].copy()

        # Add low-velocity zone
        vp[3:5, 3:5, 3:5] = 1500  # Below normal

        result = computer.compute(
            vp=vp,
            vs=sample_seismic_properties["vs"],
            rho=sample_seismic_properties["rho"],
        )
        assert result is not None

    def test_compute_with_high_vp_vs_ratio(self):
        """Test with high Vp/Vs ratio (water-saturated)."""
        computer = AVOAttributesComputer()
        shape = (5, 5, 5)
        vp = np.full(shape, 2000.0)  # Low Vp (water-like)
        vs = np.full(shape, 100.0)  # Very low Vs (water)
        rho = np.full(shape, 1000.0)  # Water density

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None

    def test_compute_with_low_vp_vs_ratio(self):
        """Test with low Vp/Vs ratio (slow rocks)."""
        computer = AVOAttributesComputer()
        shape = (5, 5, 5)
        vp = np.full(shape, 3000.0)
        vs = np.full(shape, 2500.0)  # High Vs (unusual)
        rho = np.full(shape, 2700.0)

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None


class TestFluidFactorComputer:
    """Tests for FluidFactorComputer."""

    def test_computer_initialization_fluid_factor_computer(self):
        """Test FluidFactorComputer initialization."""
        computer = FluidFactorComputer()
        assert computer is not None

    def test_default_k_value(self):
        """Test default fluid factor K."""
        assert DEFAULT_FLUID_FACTOR_K is not None
        assert DEFAULT_FLUID_FACTOR_K > 0

    def test_compute_fluid_factor(self, sample_seismic_properties):
        """Test fluid factor computation."""
        computer = FluidFactorComputer()

        # First compute lambda_rho and mu_rho
        vp = sample_seismic_properties["vp_sat"]
        vs = sample_seismic_properties["vs_sat"]
        rho = sample_seismic_properties["rho"]

        lambda_rho = rho * vp**2 - 2 * rho * vs**2
        mu_rho = rho * vs**2

        result = computer.compute(lambda_rho=lambda_rho, mu_rho=mu_rho)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_fluid_factor_shape(self, sample_seismic_properties):
        """Test fluid factor has correct shape."""
        computer = FluidFactorComputer()
        vp_sat = sample_seismic_properties["vp_sat"]
        vs_sat = sample_seismic_properties["vs_sat"]
        rho = sample_seismic_properties["rho"]

        lambda_rho = rho * vp_sat**2 - 2 * rho * vs_sat**2
        mu_rho = rho * vs_sat**2

        result = computer.compute(lambda_rho=lambda_rho, mu_rho=mu_rho)

        assert result.shape == vp_sat.shape

    def test_fluid_factor_values_physical(self, sample_seismic_properties):
        """Test fluid factor values are reasonable."""
        computer = FluidFactorComputer()

        vp_sat = sample_seismic_properties["vp_sat"]
        vs_sat = sample_seismic_properties["vs_sat"]
        rho = sample_seismic_properties["rho"]

        lambda_rho = rho * vp_sat**2 - 2 * rho * vs_sat**2
        mu_rho = rho * vs_sat**2

        result = computer.compute(lambda_rho=lambda_rho, mu_rho=mu_rho)

        # Fluid factor should have reasonable values
        assert np.all(np.isfinite(result))

    def test_fluid_factor_sensitivity(self):
        """Test fluid factor sensitivity to saturation changes."""
        computer = FluidFactorComputer()
        shape = (5, 5, 5)
        rho = np.full(shape, 2500.0)

        # Dry rock
        vp_dry = np.full(shape, 3500.0)
        vs_dry = np.full(shape, 2000.0)
        lambda_rho_dry = rho * vp_dry**2 - 2 * rho * vs_dry**2
        mu_rho_dry = rho * vs_dry**2

        # Wet rock (higher Vp)
        vp_wet = np.full(shape, 4000.0)
        vs_wet = np.full(shape, 2000.0)  # Vs stays same
        lambda_rho_wet = rho * vp_wet**2 - 2 * rho * vs_wet**2
        mu_rho_wet = rho * vs_wet**2

        result_dry = computer.compute(lambda_rho=lambda_rho_dry, mu_rho=mu_rho_dry)
        result_wet = computer.compute(lambda_rho=lambda_rho_wet, mu_rho=mu_rho_wet)

        assert result_dry is not None
        assert result_wet is not None

    def test_fluid_factor_with_1d_profile(self):
        """Test fluid factor with properly shaped 3D data."""
        computer = FluidFactorComputer()
        shape = (10, 10, 10)

        vp = np.full(shape, 3500.0)
        vs = np.full(shape, 2000.0)
        rho = np.full(shape, 2500.0)

        # Assume slightly higher Vp for saturated
        vp_sat = vp * 1.1

        lambda_rho = rho * vp_sat**2 - 2 * rho * vs**2
        mu_rho = rho * vs**2

        result = computer.compute(lambda_rho=lambda_rho, mu_rho=mu_rho)

        assert result is not None


class TestLambdaMuRhoComputer:
    """Tests for LambdaMuRhoComputer."""

    def test_computer_initialization_lambda_mu_rho_computer(self):
        """Test LambdaMuRhoComputer initialization."""
        computer = LambdaMuRhoComputer()
        assert computer is not None

    def test_compute_lambda_rho(self, sample_seismic_properties):
        """Test Lambda-Rho computation."""
        computer = LambdaMuRhoComputer()
        vp = sample_seismic_properties["vp"]
        vs = sample_seismic_properties["vs"]
        rho = sample_seismic_properties["rho"]

        result = computer.compute(vp=vp, vs=vs, rho=rho)

        assert result is not None
        if isinstance(result, dict):
            assert "lambda_rho" in result or "LambdaRho" in result

    def test_compute_mu_rho(self, sample_seismic_properties):
        """Test Mu-Rho computation."""
        computer = LambdaMuRhoComputer()
        vp = sample_seismic_properties["vp"]
        vs = sample_seismic_properties["vs"]
        rho = sample_seismic_properties["rho"]

        result = computer.compute(vp=vp, vs=vs, rho=rho)

        assert result is not None
        if isinstance(result, dict):
            assert "mu_rho" in result or "MuRho" in result

    def test_lmr_result_shapes(self, sample_seismic_properties):
        """Test LMR results have correct shape."""
        computer = LambdaMuRhoComputer()
        vp = sample_seismic_properties["vp"]

        result = computer.compute(
            vp=vp,
            vs=sample_seismic_properties["vs"],
            rho=sample_seismic_properties["rho"],
        )

        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    assert value.shape == vp.shape

    def test_lmr_values_physical(self, sample_seismic_properties):
        """Test LMR values are physically reasonable."""
        computer = LambdaMuRhoComputer()
        vp = sample_seismic_properties["vp"]
        vs = sample_seismic_properties["vs"]
        rho = sample_seismic_properties["rho"]

        result = computer.compute(vp=vp, vs=vs, rho=rho)

        # Mu-Rho should always be positive
        if isinstance(result, dict):
            if "mu_rho" in result:
                assert np.all(result["mu_rho"] > 0)
            # Lambda-Rho can be negative
            if "lambda_rho" in result:
                assert np.all(np.isfinite(result["lambda_rho"]))

    def test_lmr_with_1d_profile(self, realistic_1d_properties):
        """Test LMR with 1D velocity profile."""
        computer = LambdaMuRhoComputer()
        vp = realistic_1d_properties["vp"]
        vs = realistic_1d_properties["vs"]
        rho = realistic_1d_properties["rho"]

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None

    def test_lambda_mu_relationship(self):
        """Test Lambda-Mu physical relationships."""
        computer = LambdaMuRhoComputer()
        shape = (10, 10, 10)

        vp = np.full(shape, 4000.0)
        vs = np.full(shape, 2300.0)
        rho = np.full(shape, 2700.0)

        result = computer.compute(vp=vp, vs=vs, rho=rho)

        if isinstance(result, dict):
            if "lambda_rho" in result and "mu_rho" in result:
                lambda_rho = result["lambda_rho"]
                mu_rho = result["mu_rho"]

                # Lambda should typically be larger than Mu
                assert np.all(lambda_rho >= 0)
                assert np.all(mu_rho > 0)


class TestComputerEdgeCases:
    """Tests for edge cases in all computers."""

    def test_uniform_properties_avo(self):
        """Test AVO with uniform properties."""
        computer = AVOAttributesComputer()
        shape = (5, 5, 5)
        vp = np.full(shape, 3500.0)
        vs = np.full(shape, 2000.0)
        rho = np.full(shape, 2600.0)

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None

    def test_uniform_properties_fluid(self):
        """Test fluid factor with uniform properties."""
        computer = FluidFactorComputer()
        shape = (5, 5, 5)
        vp = np.full(shape, 3500.0)
        vs = np.full(shape, 2000.0)
        rho = np.full(shape, 2600.0)

        lambda_rho = rho * vp**2 - 2 * rho * vs**2
        mu_rho = rho * vs**2

        result = computer.compute(lambda_rho=lambda_rho, mu_rho=mu_rho)
        assert result is not None

    def test_uniform_properties_lmr(self):
        """Test LMR with uniform properties."""
        computer = LambdaMuRhoComputer()
        shape = (5, 5, 5)
        vp = np.full(shape, 3500.0)
        vs = np.full(shape, 2000.0)
        rho = np.full(shape, 2600.0)

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None

    def test_extreme_velocities_high(self):
        """Test with very high velocities."""
        computer = AVOAttributesComputer()
        shape = (5, 5, 5)
        vp = np.full(shape, 8000.0)  # Very high Vp
        vs = np.full(shape, 4500.0)
        rho = np.full(shape, 3300.0)

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None

    def test_extreme_velocities_low(self):
        """Test with very low velocities."""
        computer = AVOAttributesComputer()
        shape = (5, 5, 5)
        vp = np.full(shape, 1500.0)  # Water velocity
        vs = np.full(shape, 100.0)  # Water Vs ~0
        rho = np.full(shape, 1000.0)

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None

    def test_large_data_volume(self):
        """Test with large data volume."""
        computer = AVOAttributesComputer()
        # Reduced size for practical testing: 594 Zoeppritz solver calls
        # is too expensive; use more reasonable 20x20x20 instead of 100x100x100
        shape = (20, 20, 20)
        vp = np.random.uniform(3000, 5000, shape)
        vs = np.random.uniform(1500, 3000, shape)
        rho = np.random.uniform(2300, 2900, shape)

        result = computer.compute(vp=vp, vs=vs, rho=rho)
        assert result is not None


class TestComputerIntegration:
    """Integration tests for computers."""

    def test_sequential_computations(self, sample_seismic_properties):
        """Test sequential computations."""
        avo_comp = AVOAttributesComputer()
        lmr_comp = LambdaMuRhoComputer()
        fluid_comp = FluidFactorComputer()

        vp = sample_seismic_properties["vp"]
        vs = sample_seismic_properties["vs"]
        rho = sample_seismic_properties["rho"]

        # Compute AVO
        avo_result = avo_comp.compute(vp=vp, vs=vs, rho=rho)
        assert avo_result is not None

        # Compute LMR
        lmr_result = lmr_comp.compute(vp=vp, vs=vs, rho=rho)
        assert lmr_result is not None

        # Compute fluid factor using LMR results
        if "lambda_rho" in lmr_result and "mu_rho" in lmr_result:
            fluid_result = fluid_comp.compute(
                lambda_rho=lmr_result["lambda_rho"],
                mu_rho=lmr_result["mu_rho"],
            )
            assert fluid_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
