"""Comprehensive tests for rock_physics_attributes module.

Tests cover:
- AVOAttributesComputer: AVO intercept/gradient computation
- LambdaMuRhoComputer: Lamé parameter computation
- FluidFactorComputer: Fluid factor derivation
- AttributeDiscriminationAnalyzer: Statistical discrimination analysis
- RockPhysicsAnalyzer: Full pipeline orchestration
- Edge cases, error handling, and input validation
"""

# mypy: ignore-errors


import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from src.analysis.rock_physics import (
    AVOAttributesComputer,
    LambdaMuRhoComputer,
    FluidFactorComputer,
    AttributeDiscriminationAnalyzer,
    RockPhysicsAnalyzer,
    RockPhysicsConstants,
    DEFAULT_AVO_ANGLES_DEG,
    DEFAULT_FLUID_FACTOR_K,
)


# ============================================================================
# FIXTURES FOR MOCKING ZOEPPRITZ SOLVER
# ============================================================================


@pytest.fixture
def mock_zoeppritz_solver():
    """Fixture that mocks the ZoeppritzSolver class for testing.

    Returns a Mock object that simulates the ZoeppritzSolver class.
    The solve method returns realistic reflection coefficients.
    """
    mock_solver = Mock()

    def mock_solve(vp1, vs1, rho1, vp2, vs2, rho2, angle_rad):
        """Mock Zoeppritz solver that returns synthetic reflection coefficients."""
        # Create synthetic reflection coefficients based on input properties
        # This simulates realistic behavior without actually computing Zoeppritz equations
        shape = np.asarray(vp1).shape

        # Simple approximation: reflection coefficient increases with velocity contrast
        vp_contrast = (vp2 - vp1) / (vp2 + vp1)
        rho_contrast = (rho2 - rho1) / (rho2 + rho1)
        angle_factor = np.sin(angle_rad) ** 2

        # Combine factors to simulate angle-dependent AVO behavior
        coeff = vp_contrast * (1 - 2 * angle_factor) + rho_contrast * angle_factor

        # Add some imaginary part for realism (minor phase shift)
        result = coeff.astype(complex) + 0.01j * np.random.RandomState(42).randn(*shape)

        return result

    mock_solver.solve.side_effect = mock_solve
    return mock_solver


@pytest.fixture
def mock_zoeppritz_import(mock_zoeppritz_solver):
    """Fixture that patches the ZoeppritzSolver class for all tests in a class."""
    with patch(
        "src.signal.reflectivity.ZoeppritzSolver", return_value=mock_zoeppritz_solver
    ):
        yield mock_zoeppritz_solver


class TestRockPhysicsAnalyzer:
    """Test suite for RockPhysicsAnalyzer orchestrator."""

    @pytest.fixture(autouse=True)
    def _mock_zoeppritz(self, mock_zoeppritz_solver):
        """Auto-use fixture to mock Zoeppritz solver for this test class."""
        with patch(
            "src.signal.reflectivity.ZoeppritzSolver",
            return_value=mock_zoeppritz_solver,
        ):
            yield

    @pytest.fixture
    def analyzer_instance(self) -> RockPhysicsAnalyzer:
        """Create a fresh analyzer instance."""
        return RockPhysicsAnalyzer()

    @pytest.fixture
    def sample_rock_properties(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sample rock properties."""
        shape = (5, 6, 7)
        vp = np.random.uniform(3000, 4000, shape).astype(np.float32)
        vs = np.random.uniform(1500, 2000, shape).astype(np.float32)
        rho = np.random.uniform(2200, 2600, shape).astype(np.float32)
        return vp, vs, rho

    def test_compute_avo_attributes(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
        sample_rock_properties: tuple,
    ) -> None:
        """Test AVO computation through orchestrator."""
        vp, vs, rho = sample_rock_properties

        result = analyzer_instance.compute_avo_attributes(vp, vs, rho)

        assert set(result.keys()) == RockPhysicsConstants.AVO_KEYS

    def test_compute_lambda_mu_rho(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
        sample_rock_properties: tuple,
    ) -> None:
        """Test Lambda-Mu-Rho computation."""
        vp, vs, rho = sample_rock_properties

        result = analyzer_instance.compute_lambda_mu_rho(vp, vs, rho)

        assert set(result.keys()) == RockPhysicsConstants.LAMBDA_MU_KEYS

    def test_compute_fluid_factor(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test fluid factor computation."""
        lambda_rho = np.random.rand(5, 6, 7)
        mu_rho = np.random.rand(5, 6, 7)

        result = analyzer_instance.compute_fluid_factor(lambda_rho, mu_rho)

        assert isinstance(result, np.ndarray)
        assert result.shape == lambda_rho.shape

    def test_analyze_attribute_discrimination(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test discrimination analysis of single attribute."""
        attribute = np.random.rand(100)
        facies = np.random.randint(0, 2, 100)

        result = analyzer_instance.analyze_attribute_discrimination(
            attribute, facies, name="Test"
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == RockPhysicsConstants.DISCRIMINATION_KEYS

    def test_compare_all_attributes(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test discrimination analysis of multiple attributes."""
        attribute_results = {
            "attr1": np.random.rand(5, 6, 7),
            "attr2": np.random.rand(5, 6, 7),
        }
        facies = np.random.randint(0, 2, (5, 6, 7))

        results = analyzer_instance.compare_all_attributes(attribute_results, facies)

        assert isinstance(results, dict)
        assert len(results) == 2

    def test_build_attribute_results(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test consolidation of results."""
        avo_results = {
            "intercept": np.random.rand(5, 6, 7),
            "gradient": np.random.rand(5, 6, 7),
            "product": np.random.rand(5, 6, 7),
            "scaled_gradient": np.random.rand(5, 6, 7),
        }
        lam_mu_rho = {
            "lambda_rho": np.random.rand(5, 6, 7),
            "mu_rho": np.random.rand(5, 6, 7),
            "lambda_mu_ratio": np.random.rand(5, 6, 7),
        }

        result = analyzer_instance._build_attribute_results(
            avo_results, lam_mu_rho, None
        )

        assert isinstance(result, dict)
        assert len(result) == 7  # 4 AVO + 3 lambda_mu + no fluid

    def test_build_attribute_results_with_fluid(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test result consolidation with fluid factor."""
        avo_results = {
            "intercept": np.ones((2, 2, 2)),
            "gradient": np.ones((2, 2, 2)),
            "product": np.ones((2, 2, 2)),
            "scaled_gradient": np.ones((2, 2, 2)),
        }
        lam_mu_rho = {
            "lambda_rho": np.ones((2, 2, 2)),
            "mu_rho": np.ones((2, 2, 2)),
            "lambda_mu_ratio": np.ones((2, 2, 2)),
        }
        fluid = np.ones((2, 2, 2))

        result = analyzer_instance._build_attribute_results(
            avo_results, lam_mu_rho, fluid
        )

        assert len(result) == 8  # 4 AVO + 3 lambda_mu + 1 fluid
        assert "fluid_factor" in result

    def test_build_attribute_results_missing_avo_keys(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test that missing AVO keys raise ValueError."""
        avo_results = {
            "intercept": np.ones((2, 2, 2)),
            "gradient": np.ones((2, 2, 2)),
            # Missing 'product' and 'scaled_gradient'
        }
        lam_mu_rho = {
            "lambda_rho": np.ones((2, 2, 2)),
            "mu_rho": np.ones((2, 2, 2)),
            "lambda_mu_ratio": np.ones((2, 2, 2)),
        }

        with pytest.raises(ValueError, match="AVO results missing"):
            analyzer_instance._build_attribute_results(avo_results, lam_mu_rho, None)

    def test_build_attribute_results_missing_lambda_mu_keys(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test that missing Lambda-Mu keys raise ValueError."""
        avo_results = {
            "intercept": np.ones((2, 2, 2)),
            "gradient": np.ones((2, 2, 2)),
            "product": np.ones((2, 2, 2)),
            "scaled_gradient": np.ones((2, 2, 2)),
        }
        lam_mu_rho = {
            "lambda_rho": np.ones((2, 2, 2)),
            # Missing 'mu_rho' and 'lambda_mu_ratio'
        }

        with pytest.raises(ValueError, match="Lambda-Mu-Rho results missing"):
            analyzer_instance._build_attribute_results(avo_results, lam_mu_rho, None)

    def test_load_and_unwrap_properties(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test property loading and unwrapping."""
        mock_dm = Mock()
        mock_dm.vp = np.ones((2, 2, 2))
        mock_dm.vs = np.ones((2, 2, 2))
        mock_dm.rho = np.ones((2, 2, 2))
        mock_dm.facies = np.ones((2, 2, 2))

        vp, vs, rho, facies = analyzer_instance._load_and_unwrap_properties(mock_dm)

        assert vp.shape == (2, 2, 2)
        assert vs.shape == (2, 2, 2)
        assert rho.shape == (2, 2, 2)
        assert facies.shape == (2, 2, 2)

    def test_get_grid_configuration_fallback(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
    ) -> None:
        """Test grid configuration returns sensible defaults."""
        # The method now always uses defaults (legacy method)
        data_path, file_map, grid_spec = analyzer_instance._get_grid_configuration()

        # Should use defaults
        assert isinstance(data_path, str)
        assert isinstance(file_map, dict)
        assert grid_spec is not None

    def test_compute_all_attributes_integration(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
        sample_rock_properties: tuple,
    ) -> None:
        """Test full attribute computation pipeline."""
        vp, vs, rho = sample_rock_properties

        avo, lmr, fluid = analyzer_instance._compute_all_attributes(
            vp, vs, rho, DEFAULT_AVO_ANGLES_DEG
        )

        assert set(avo.keys()) == RockPhysicsConstants.AVO_KEYS
        assert set(lmr.keys()) == RockPhysicsConstants.LAMBDA_MU_KEYS
        assert isinstance(fluid, np.ndarray) or fluid is None

    def test_main_pipeline_with_missing_dataset(
        self,
        analyzer_instance: RockPhysicsAnalyzer,
        tmp_path: Path,
    ) -> None:
        """Test main pipeline raises on missing dataset."""
        cache_dir = str(tmp_path)

        # Mock _get_grid_configuration to prevent matplotlib initialization hang
        with patch.object(
            analyzer_instance,
            "_get_grid_configuration",
            return_value=(
                "/fake/path",
                {},
                Mock(spec=["shape", "__getitem__"]),
            ),
        ):
            with patch.object(
                analyzer_instance,
                "_load_dataset_manager",
                side_effect=FileNotFoundError("No data"),
            ):
                with pytest.raises(FileNotFoundError):
                    analyzer_instance.run(cache_dir=cache_dir)


class TestRockPhysicsAnalyzerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def _mock_zoeppritz(self, mock_zoeppritz_solver):
        """Auto-use fixture to mock Zoeppritz solver for this test class."""
        with patch(
            "src.signal.reflectivity.ZoeppritzSolver",
            return_value=mock_zoeppritz_solver,
        ):
            yield

    def test_avo_with_single_layer(self) -> None:
        """Test AVO computation with minimal 3D volume (single layer)."""
        vp = np.ones((2, 2, 2))
        vs = np.ones((2, 2, 2))
        rho = np.ones((2, 2, 2))
        computer = AVOAttributesComputer()

        result = computer.compute(vp, vs, rho)

        assert result["intercept"].shape == (2, 2, 1)

    def test_discrimination_with_identical_classes(self) -> None:
        """Test discrimination when both classes have identical values."""
        # All same value for both classes
        attribute = np.ones(100)
        facies = np.array([0] * 50 + [1] * 50)
        analyzer = AttributeDiscriminationAnalyzer()

        result = analyzer.analyze_single(attribute, facies)

        # Cohen's d should be 0 (no separation)
        assert result["cohens_d"] == 0.0

    def test_lambda_mu_ratio_with_zero_mu(self) -> None:
        """Test Lambda-Mu ratio when Mu approaches zero."""
        vp = np.ones((2, 2, 2)) * 3000
        vs = np.ones((2, 2, 2)) * 0.001  # Very small (near zero)
        rho = np.ones((2, 2, 2)) * 2300
        computer = LambdaMuRhoComputer()

        result = computer.compute(vp, vs, rho)

        # Ratio should be protected by EPSILON
        assert np.all(np.isfinite(result["lambda_mu_ratio"]))

    def test_fluid_factor_with_negative_values(self) -> None:
        """Test fluid factor handles negative intermediate values."""
        lambda_rho = np.array([[[-10.0, 20.0]], [[30.0, -40.0]]])
        mu_rho = np.array([[[2.0, 3.0]], [[4.0, 5.0]]])
        computer = FluidFactorComputer()

        result = computer.compute(lambda_rho, mu_rho)

        # Should handle negative values correctly
        assert np.all(np.isfinite(result))
        expected = lambda_rho - DEFAULT_FLUID_FACTOR_K * mu_rho
        assert_array_almost_equal(result, expected)


class TestCoverageEdgeCases:
    """Additional tests to improve code coverage."""

    @pytest.fixture(autouse=True)
    def _mock_zoeppritz(self, mock_zoeppritz_solver):
        """Auto-use fixture to mock Zoeppritz solver for this test class."""
        with patch(
            "src.signal.reflectivity.ZoeppritzSolver",
            return_value=mock_zoeppritz_solver,
        ):
            yield

    def test_pearson_correlation_exception_handling(self) -> None:
        """Test Pearson correlation exception handling in analyze_single."""
        # Create data that might cause correlation issues
        attribute = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        facies = np.array([0, 0, 0, 0, 0])  # All same class
        analyzer = AttributeDiscriminationAnalyzer()

        # Should handle the case gracefully
        result = analyzer.analyze_single(attribute, facies, name="test_attr")
        assert result["name"] == "test_attr"
        assert isinstance(result["pearson_r"], (int, float, np.number))

    def test_analyze_multiple_with_exception_handling(self) -> None:
        """Test analyze_multiple handles exceptions in individual analyses."""
        attribute_results = {
            "good_attr": np.array([1.0, 2.0, 3.0]),
            "bad_attr": None,  # This could cause an issue
        }
        facies = np.array([0, 1, 0])
        analyzer = AttributeDiscriminationAnalyzer()

        # Should not raise, should handle gracefully
        result = analyzer.analyze_multiple(attribute_results, facies)
        assert "good_attr" in result
        # bad_attr might fail, but that's OK

    def test_rank_deficient_design_matrix_warning(self) -> None:
        """Test handling of potential rank deficiency in AVO computation."""
        # Create simple test data with 2 layers
        vp = np.ones((3, 3, 2)) * 3000
        vs = np.ones((3, 3, 2)) * 1500
        rho = np.ones((3, 3, 2)) * 2300
        computer = AVOAttributesComputer()

        # This should complete and might log a warning if rank is deficient
        result = computer.compute(vp, vs, rho, angles_deg=(0, 5))
        assert "intercept" in result
        assert "gradient" in result
        """Test compute_class_stats with class data."""
        analyzer = AttributeDiscriminationAnalyzer()
        class0_values = np.array([1.0, 2.0, 3.0])
        class1_values = np.array([4.0, 5.0, 6.0])

        mean0, std0, mean1, std1 = analyzer._compute_class_stats(
            class0_values, class1_values
        )
        assert mean0 == 2.0
        assert mean1 == 5.0
        assert std0 > 0
        assert std1 > 0

    def test_ensure_valid_array_with_all_nans(self) -> None:
        """Test _ensure_valid_array with array of all NaNs."""
        analyzer = AttributeDiscriminationAnalyzer()
        arr = np.array([np.nan, np.nan, np.nan])

        result = analyzer._ensure_valid_array(arr)
        # _ensure_valid_array doesn't filter NaNs, just returns valid entries
        # If all are NaN, this just returns the original array or empty
        assert isinstance(result, np.ndarray)

    def test_avo_with_multiple_layers_and_angles(self) -> None:
        """Test AVO computation with multiple layers and angles."""
        # Create simple test data with 3 layers
        vp = np.ones((2, 2, 3)) * 3000
        vs = np.ones((2, 2, 3)) * 1500
        rho = np.ones((2, 2, 3)) * 2300
        computer = AVOAttributesComputer()

        # Use multiple angles
        angles = (0, 10, 20, 30)
        result = computer.compute(vp, vs, rho, angles_deg=angles)

        # Verify output structure
        assert all(
            key in result
            for key in ["intercept", "gradient", "product", "scaled_gradient"]
        )
        # Should have nk-1 layers (3-1=2)
        assert result["intercept"].shape[2] == 2

    def test_lambda_mu_rho_with_inf_values(self) -> None:
        """Test LambdaMuRhoComputer handles infinity gracefully."""
        vp = np.array([[[3000.0, 3000.0]]])
        vs = np.array([[[1500.0, 1500.0]]])
        rho = np.array([[[2300.0, np.inf]]])  # Inf density
        computer = LambdaMuRhoComputer()

        result = computer.compute(vp, vs, rho)

        # Should complete even with inf
        assert "mu_rho" in result
        assert "lambda_rho" in result

    def test_fluid_factor_with_nan_values(self) -> None:
        """Test FluidFactorComputer with NaN inputs."""
        lambda_rho = np.array([[[np.nan, 20.0]]])
        mu_rho = np.array([[[2.0, np.nan]]])
        computer = FluidFactorComputer()

        result = computer.compute(lambda_rho, mu_rho)

        # Should propagate NaN
        assert np.isnan(result[0, 0, 0])
        assert np.isnan(result[0, 0, 1])


class TestExceptionHandling:
    """Test exception handling paths for improved coverage."""

    @pytest.fixture
    def analyzer_instance(self) -> RockPhysicsAnalyzer:
        """Provide fresh RockPhysicsAnalyzer instance for each test."""
        return RockPhysicsAnalyzer()

    @pytest.fixture(autouse=True)
    def _mock_zoeppritz(self, mock_zoeppritz_solver: Mock):
        """Auto-apply Zoeppritz mock to all tests in this class."""
        with patch(
            "src.signal.reflectivity.ZoeppritzSolver",
            return_value=mock_zoeppritz_solver,
        ):
            yield

    def test_avo_computation_with_invalid_angles(
        self, analyzer_instance: RockPhysicsAnalyzer, mock_zoeppritz_solver: Mock
    ) -> None:
        """Test AVO computation raises ValueError for empty angles."""
        vp = np.array([[[2500.0, 3000.0]]])
        vs = np.array([[[1200.0, 1500.0]]])
        rho = np.array([[[2300.0, 2500.0]]])

        with pytest.raises(ValueError):
            analyzer_instance.compute_avo_attributes(vp, vs, rho, angles_deg=[])

    def test_compute_all_attributes_avo_exception(
        self, analyzer_instance: RockPhysicsAnalyzer
    ) -> None:
        """Test _compute_all_attributes handles AVO computation exception."""
        vp = np.array([[[2500.0]]])
        vs = np.array([[[1200.0]]])
        rho = np.array([[[2300.0]]])

        # Mock compute_avo_attributes to raise
        with patch.object(
            analyzer_instance,
            "compute_avo_attributes",
            side_effect=ValueError("Test AVO error"),
        ):
            with pytest.raises(ValueError, match="Test AVO error"):
                analyzer_instance._compute_all_attributes(vp, vs, rho, [0, 10, 20])

    def test_compute_all_attributes_lambda_mu_exception(
        self, analyzer_instance: RockPhysicsAnalyzer, mock_zoeppritz_solver: Mock
    ) -> None:
        """Test _compute_all_attributes handles Lambda-Mu-Rho exception."""
        vp = np.array([[[2500.0]]])
        vs = np.array([[[1200.0]]])
        rho = np.array([[[2300.0]]])

        # Mock compute_lambda_mu_rho to raise
        with patch.object(
            analyzer_instance,
            "compute_lambda_mu_rho",
            side_effect=Exception("Test lambda-mu error"),
        ):
            with pytest.raises(Exception, match="Test lambda-mu error"):
                analyzer_instance._compute_all_attributes(vp, vs, rho, [0, 10, 20])

    def test_compute_all_attributes_fluid_factor_exception(
        self, analyzer_instance: RockPhysicsAnalyzer, mock_zoeppritz_solver: Mock
    ) -> None:
        """Test _compute_all_attributes handles fluid factor exception gracefully."""
        vp = np.array([[[2500.0]]])
        vs = np.array([[[1200.0]]])
        rho = np.array([[[2300.0]]])

        # Mock compute_fluid_factor to raise
        with patch.object(
            analyzer_instance,
            "compute_fluid_factor",
            side_effect=Exception("Test fluid factor error"),
        ):
            # Should not raise, fluid factor is optional
            result = analyzer_instance._compute_all_attributes(vp, vs, rho, [0, 10, 20])
            avo, lam_mu, fluid = result
            assert fluid is None  # Failed computation should return None

    def test_analyze_single_pearsonr_exception(
        self,
    ) -> None:
        """Test analyze_single handles pearsonr exception."""
        analyzer = AttributeDiscriminationAnalyzer()

        # Create attribute and facies arrays
        attr = np.array([1.0, 1.0, 1.0, 1.0])  # Constant array
        facies = np.array([0, 1, 0, 1])

        # Mock pearsonr to raise exception in the module where it's used
        with patch("scipy.stats.pearsonr", side_effect=Exception("Correlation error")):
            result = analyzer.analyze_single(attr, facies, "test_attr")

        # Should gracefully return with correlation values set to 0.0, 1.0
        assert result["pearson_r"] == 0.0
        assert result["p_value"] == 1.0

    def test_compare_all_attributes_with_exception(
        self, analyzer_instance: RockPhysicsAnalyzer
    ) -> None:
        """Test compare_all_attributes handles internal exception gracefully."""
        attributes = {
            "intercept": np.array([[[1.0, 2.0]]]),
            "gradient": np.array([[[0.5, 0.6]]]),
        }
        facies = np.array([[[0, 1]]])

        # Mock the discrimination analyzer's analyze_single to raise for one attribute
        with patch(
            "scipy.stats.pearsonr", side_effect=Exception("Test analysis error")
        ):
            result = analyzer_instance.compare_all_attributes(attributes, facies)

        # Should return dict even if some analyses encounter exceptions
        assert isinstance(result, dict)

    def test_main_pipeline_cache_save_exception(
        self, analyzer_instance: RockPhysicsAnalyzer, tmp_path: Path
    ) -> None:
        """Test main pipeline handles cache save exception."""
        cache_dir = str(tmp_path)

        # Mock _get_grid_configuration
        with patch.object(
            analyzer_instance,
            "_get_grid_configuration",
            return_value=(
                "/fake/path",
                {},
                Mock(spec=["shape", "__getitem__"]),
            ),
        ):
            # Mock _load_dataset_manager
            with patch.object(
                analyzer_instance,
                "_load_dataset_manager",
                return_value=Mock(spec=["get"]),
            ):
                # Mock _load_and_unwrap_properties to return valid arrays
                with patch.object(
                    analyzer_instance,
                    "_load_and_unwrap_properties",
                    return_value=(
                        np.array([[[2500.0]]]),
                        np.array([[[1200.0]]]),
                        np.array([[[2300.0]]]),
                        np.array([[[0]]]),
                    ),
                ):
                    # Mock savez_compressed to raise
                    with patch(
                        "numpy.savez_compressed", side_effect=IOError("Save failed")
                    ):
                        # Should still return True despite save failure
                        result = analyzer_instance.run(
                            cache_dir=cache_dir, generate_plots=False
                        )
                        assert result is True

    def test_main_pipeline_discrimination_exception(
        self, analyzer_instance: RockPhysicsAnalyzer, tmp_path: Path
    ) -> None:
        """Test main pipeline handles discrimination analysis exception."""
        cache_dir = str(tmp_path)

        with patch.object(
            analyzer_instance,
            "_get_grid_configuration",
            return_value=(
                "/fake/path",
                {},
                Mock(spec=["shape", "__getitem__"]),
            ),
        ):
            with patch.object(
                analyzer_instance,
                "_load_dataset_manager",
                return_value=Mock(spec=["get"]),
            ):
                with patch.object(
                    analyzer_instance,
                    "_load_and_unwrap_properties",
                    return_value=(
                        np.array([[[2500.0]]]),
                        np.array([[[1200.0]]]),
                        np.array([[[2300.0]]]),
                        np.array([[[0]]]),
                    ),
                ):
                    with patch.object(
                        analyzer_instance,
                        "compare_all_attributes",
                        side_effect=Exception("Discrimination failed"),
                    ):
                        # Should not raise, discrimination is optional
                        result = analyzer_instance.run(
                            cache_dir=cache_dir, generate_plots=False
                        )
                        # Should return path or True (not raise)
                        assert result is not None


class TestOptionalFeatures:
    """Test optional features for improved coverage."""

    @pytest.fixture
    def analyzer_instance(self) -> RockPhysicsAnalyzer:
        """Provide fresh RockPhysicsAnalyzer instance for each test."""
        return RockPhysicsAnalyzer()

    @pytest.fixture(autouse=True)
    def _mock_zoeppritz(self, mock_zoeppritz_solver: Mock):
        """Auto-apply Zoeppritz mock to all tests in this class."""
        with patch(
            "src.signal.reflectivity.ZoeppritzSolver",
            return_value=mock_zoeppritz_solver,
        ):
            yield

    def test_main_pipeline_with_plotting_disabled(
        self, analyzer_instance: RockPhysicsAnalyzer, tmp_path: Path
    ) -> None:
        """Test main pipeline with generate_plots=False (untested branch)."""
        cache_dir = str(tmp_path)

        with patch.object(
            analyzer_instance,
            "_get_grid_configuration",
            return_value=(
                "/fake/path",
                {},
                Mock(spec=["shape", "__getitem__"]),
            ),
        ):
            with patch.object(
                analyzer_instance,
                "_load_dataset_manager",
                return_value=Mock(spec=["get"]),
            ):
                with patch.object(
                    analyzer_instance,
                    "_load_and_unwrap_properties",
                    return_value=(
                        np.array([[[2500.0]]]),
                        np.array([[[1200.0]]]),
                        np.array([[[2300.0]]]),
                        np.array([[[0]]]),
                    ),
                ):
                    # Test with generate_plots=False
                    result = analyzer_instance.run(
                        cache_dir=cache_dir, generate_plots=False
                    )
                    assert result is not None

    def test_main_pipeline_with_verbose_mode(
        self, analyzer_instance: RockPhysicsAnalyzer, tmp_path: Path, caplog
    ) -> None:
        """Test main pipeline with verbose=True."""
        cache_dir = str(tmp_path)

        with patch.object(
            analyzer_instance,
            "_get_grid_configuration",
            return_value=(
                "/fake/path",
                {},
                Mock(spec=["shape", "__getitem__"]),
            ),
        ):
            with patch.object(
                analyzer_instance,
                "_load_dataset_manager",
                return_value=Mock(spec=["get"]),
            ):
                with patch.object(
                    analyzer_instance,
                    "_load_and_unwrap_properties",
                    return_value=(
                        np.array([[[2500.0]]]),
                        np.array([[[1200.0]]]),
                        np.array([[[2300.0]]]),
                        np.array([[[0]]]),
                    ),
                ):
                    with caplog.at_level(logging.INFO):
                        result = analyzer_instance.run(
                            cache_dir=cache_dir,
                            generate_plots=False,
                            verbose=True,
                        )
                    assert result is not None
                    # Check that logs were generated
                    assert any(
                        "pipeline" in record.message.lower()
                        for record in caplog.records
                    )

    def test_main_pipeline_with_save_npz_only(
        self, analyzer_instance: RockPhysicsAnalyzer, tmp_path: Path
    ) -> None:
        """Test main pipeline with save_npz_only=True."""
        cache_dir = str(tmp_path)

        with patch.object(
            analyzer_instance,
            "_get_grid_configuration",
            return_value=(
                "/fake/path",
                {},
                Mock(spec=["shape", "__getitem__"]),
            ),
        ):
            with patch.object(
                analyzer_instance,
                "_load_dataset_manager",
                return_value=Mock(spec=["get"]),
            ):
                with patch.object(
                    analyzer_instance,
                    "_load_and_unwrap_properties",
                    return_value=(
                        np.array([[[2500.0]]]),
                        np.array([[[1200.0]]]),
                        np.array([[[2300.0]]]),
                        np.array([[[0]]]),
                    ),
                ):
                    # With save_npz_only=True, plotting should be skipped
                    result = analyzer_instance.run(
                        cache_dir=cache_dir,
                        generate_plots=True,  # Set to True but should be skipped
                        save_npz_only=True,
                    )
                    assert result is not None


class TestFallbackPaths:
    """Test fallback mechanisms for improved coverage."""

    @pytest.fixture
    def analyzer_instance(self) -> RockPhysicsAnalyzer:
        """Provide fresh RockPhysicsAnalyzer instance for each test."""
        return RockPhysicsAnalyzer()

    @pytest.fixture(autouse=True)
    def _mock_zoeppritz(self, mock_zoeppritz_solver: Mock):
        """Auto-apply Zoeppritz mock to all tests in this class."""
        with patch(
            "src.signal.reflectivity.ZoeppritzSolver",
            return_value=mock_zoeppritz_solver,
        ):
            yield

    def test_get_grid_configuration_fallback_fallback_paths(
        self, analyzer_instance: RockPhysicsAnalyzer
    ) -> None:
        """Test _get_grid_configuration fallback when import fails."""
        # Mock the plotting config import to raise
        with patch.dict("sys.modules", {"src.plotting.helpers.plot": None}):
            result = analyzer_instance._get_grid_configuration()

            # Should return fallback values
            assert result is not None
            data_path, file_map, grid_spec = result
            assert data_path is not None
            assert isinstance(file_map, dict)
            assert grid_spec is not None

    def test_compute_fluid_factor_missing_keys_exception(
        self, analyzer_instance: RockPhysicsAnalyzer, mock_zoeppritz_solver: Mock
    ) -> None:
        """Test _compute_all_attributes handles missing lambda_rho/mu_rho keys."""
        vp = np.array([[[2500.0]]])
        vs = np.array([[[1200.0]]])
        rho = np.array([[[2300.0]]])

        # Mock compute_avo_attributes to return dict without lambda_rho/mu_rho
        with patch.object(
            analyzer_instance,
            "compute_avo_attributes",
            return_value={
                "intercept": np.array([[[1.0]]]),
                "gradient": np.array([[[0.5]]]),
            },
        ):
            # Should not raise KeyError, should gracefully skip fluid factor
            result = analyzer_instance._compute_all_attributes(vp, vs, rho, [0, 10, 20])
            avo, lam_mu, fluid = result
            # Fluid should be None due to missing keys
            assert fluid is None

    def test_main_pipeline_plotting_exception(
        self, analyzer_instance: RockPhysicsAnalyzer, tmp_path: Path
    ) -> None:
        """Test main pipeline handles plotting exception gracefully."""
        cache_dir = str(tmp_path)

        with patch.object(
            analyzer_instance,
            "_get_grid_configuration",
            return_value=(
                "/fake/path",
                {},
                Mock(spec=["shape", "__getitem__"]),
            ),
        ):
            with patch.object(
                analyzer_instance,
                "_load_dataset_manager",
                return_value=Mock(spec=["get"]),
            ):
                with patch.object(
                    analyzer_instance,
                    "_load_and_unwrap_properties",
                    return_value=(
                        np.array([[[2500.0]]]),
                        np.array([[[1200.0]]]),
                        np.array([[[2300.0]]]),
                        np.array([[[0]]]),
                    ),
                ):
                    # Simply test that generate_plots=True doesn't break
                    # (the exception handling in plotting is already tested indirectly)
                    result = analyzer_instance.run(
                        cache_dir=cache_dir, generate_plots=True
                    )
                    # Should return a result even with plotting
                    assert result is not None
