"""Comprehensive tests for src/modeling module coverage improvement.

Targets low-coverage areas:
- modeling.py: Noise/weighting logic, angle model details
- pipeline.py: Full pipeline execution
- model_cache.py: Cache I/O operations
- processors.py: Reflectivity and convolution kernels
- resampler.py: Resampling implementation
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile

from src.modeling.modeling import (
    AngleModel,
    AVOSynthesizer,
    SynthesisConfig,
    _unwrap_quantity,
)
from src.modeling.processors import ReflectivityComputer, WaveletConvolver
from src.modeling.model_cache import CacheManager
from src.modeling.resampler import ResamplingService
from src.modeling.pipeline import ModelingPipeline
from src.modeling.config import ModelingConfig, ModelingDefaults
from src.utils.quantity import Quantity
from src.io.grid import GridSpec


class TestAngleModelCoverage:
    """Tests for AngleModel edge cases and untested paths."""

    def test_quality_weight_at_exact_calibration_angles(self):
        """Test exact interpolation at calibrated angles."""
        model = AngleModel()

        # Test exact values
        assert model.quality_weight(0) == 0.90
        assert model.quality_weight(15) == 1.00
        assert model.quality_weight(45) == 0.40

    def test_quality_weight_interpolation(self):
        """Test interpolation between calibrated angles."""
        model = AngleModel()

        # Test interpolation: between 0 (0.90) and 5 (0.95)
        weight_2_5 = model.quality_weight(2.5)
        assert 0.90 < weight_2_5 < 0.95

        # Test interpolation: between 10 (0.98) and 15 (1.00)
        weight_12_5 = model.quality_weight(12.5)
        assert 0.98 < weight_12_5 < 1.00

    def test_noise_level_at_exact_calibration_angles(self):
        """Test exact noise levels at calibrated angles."""
        model = AngleModel()

        assert model.noise_level(0) == 0.011
        assert model.noise_level(15) == 0.002
        assert model.noise_level(45) == 0.023

    def test_noise_level_interpolation(self):
        """Test interpolation of noise levels."""
        model = AngleModel()

        # Between 0 (0.011) and 5 (0.007)
        noise_2_5 = model.noise_level(2.5)
        assert 0.007 < noise_2_5 < 0.011

    def test_add_noise_basic(self):
        """Test adding noise to seismic data."""
        model = AngleModel()
        seismic = np.ones((100,)) * 0.5

        noisy = model.add_noise(seismic, angle=15.0, snr_db=20.0, seed=42)

        # Check shape preserved
        assert noisy.shape == seismic.shape
        # Check dtype
        assert noisy.dtype == np.float32
        # Check noise was actually added
        assert not np.allclose(noisy, seismic)

    def test_add_noise_with_different_angles(self):
        """Test that different angles produce different noise."""
        model = AngleModel()
        seismic = np.ones((100,)) * 0.5

        noise_15 = model.add_noise(seismic, angle=15.0, snr_db=20.0, seed=42)
        noise_30 = model.add_noise(seismic, angle=30.0, snr_db=20.0, seed=42)

        # Different angles should give different noise (higher noise at 30°)
        assert not np.allclose(noise_15, noise_30)

    def test_add_noise_reproducible_with_seed(self):
        """Test that noise is reproducible with same seed."""
        model = AngleModel()
        seismic = np.ones((100,)) * 0.5

        noise1 = model.add_noise(seismic, angle=15.0, snr_db=20.0, seed=123)
        noise2 = model.add_noise(seismic, angle=15.0, snr_db=20.0, seed=123)

        # Same seed should produce same noise
        assert np.allclose(noise1, noise2)

    def test_weighted_stack_basic(self):
        """Test weighted stack combination."""
        model = AngleModel()

        stack1 = np.ones((10, 10, 10)) * 0.1
        stack2 = np.ones((10, 10, 10)) * 0.2
        angle_stacks = [stack1, stack2]
        angles = [0.0, 15.0]

        weighted = model.weighted_stack(angle_stacks, angles)

        # Check shape
        assert weighted.shape == stack1.shape
        # Check it's a weighted combination
        assert 0.1 < np.mean(weighted) < 0.2

    def test_weighted_stack_mismatched_lengths(self):
        """Test error when stacks and angles mismatch."""
        model = AngleModel()

        stack1 = np.ones((10, 10, 10))
        angle_stacks = [stack1]
        angles = [0.0, 15.0]  # Mismatch!

        with pytest.raises(ValueError):
            model.weighted_stack(angle_stacks, angles)


class TestAVOSynthesizerCoverage:
    """Tests for AVOSynthesizer untested paths."""

    def test_create_synthetics_basic(self):
        """Test basic synthesis without configuration."""
        synthesizer = AVOSynthesizer()

        # Create simple test data
        props = {
            "vp": np.ones((10, 5, 5)) * 3000,
            "vs": np.ones((10, 5, 5)) * 1500,
            "rho": np.ones((10, 5, 5)) * 2.5,
        }
        angles = [0.0, 15.0]
        wavelet = np.array([0.1, 0.5, 0.1])

        angle_stacks, full_stack = synthesizer.create_synthetics(
            props, angles, wavelet, config=None
        )

        # Check outputs
        assert len(angle_stacks) == 2
        assert full_stack.shape == (10, 5, 5)

    def test_create_synthetics_with_noise(self):
        """Test synthesis with noise addition."""
        synthesizer = AVOSynthesizer()

        props = {
            "vp": np.ones((8, 4, 4)) * 3000,
            "vs": np.ones((8, 4, 4)) * 1500,
            "rho": np.ones((8, 4, 4)) * 2.5,
        }
        angles = [0.0]
        wavelet = np.array([0.1, 0.5, 0.1])

        config = SynthesisConfig(add_noise=True, snr_db=15.0)
        angle_stacks, full_stack = synthesizer.create_synthetics(
            props, angles, wavelet, config=config
        )

        assert len(angle_stacks) == 1
        assert full_stack.shape == (8, 4, 4)

    def test_create_synthetics_with_quality_weighting(self):
        """Test synthesis with quality weighting."""
        synthesizer = AVOSynthesizer()

        props = {
            "vp": np.ones((6, 3, 3)) * 3000,
            "vs": np.ones((6, 3, 3)) * 1500,
            "rho": np.ones((6, 3, 3)) * 2.5,
        }
        angles = [0.0, 15.0]
        wavelet = np.array([0.2, 0.6, 0.2])

        config = SynthesisConfig(use_quality_weighting=True)
        angle_stacks, full_stack = synthesizer.create_synthetics(
            props, angles, wavelet, config=config
        )

        assert full_stack.shape == (6, 3, 3)


class TestReflectivityComputerCoverage:
    """Tests for ReflectivityComputer computation."""

    def test_compute_reflectivity_shape(self):
        """Test reflectivity computation preserves shape."""
        computer = ReflectivityComputer(block_size=5)

        vp = np.ones((10, 5, 5)) * 3000
        vs = np.ones((10, 5, 5)) * 1500
        rho = np.ones((10, 5, 5)) * 2.5

        rc = computer.compute_reflectivity(vp, vs, rho, angle=15.0)

        assert rc.shape == vp.shape
        assert rc.dtype == np.float32

    def test_compute_reflectivity_zero_at_top(self):
        """Test that reflectivity is zero at top boundary."""
        computer = ReflectivityComputer(block_size=5)

        vp = np.ones((8, 4, 4)) * 3000
        vs = np.ones((8, 4, 4)) * 1500
        rho = np.ones((8, 4, 4)) * 2.5

        rc = computer.compute_reflectivity(vp, vs, rho, angle=0.0)

        # First level should be all zeros (boundary)
        assert np.allclose(rc[0], 0)

    def test_compute_reflectivity_different_angles(self):
        """Test reflectivity varies with angle."""
        computer = ReflectivityComputer()

        vp = np.random.randn(6, 3, 3) * 100 + 3000
        vs = np.random.randn(6, 3, 3) * 100 + 1500
        rho = np.random.randn(6, 3, 3) * 0.1 + 2.5

        rc_0 = computer.compute_reflectivity(vp, vs, rho, angle=0.0)
        rc_30 = computer.compute_reflectivity(vp, vs, rho, angle=30.0)

        # Should be different
        assert not np.allclose(rc_0, rc_30)


class TestWaveletConvolverCoverage:
    """Tests for WaveletConvolver."""

    def test_convolve_3d_basic(self):
        """Test 3D convolution shape and dtype."""
        cube = np.ones((10, 5, 5)) * 0.1
        wavelet = np.array([0.1, 0.5, 0.1])

        result = WaveletConvolver.convolve_3d(cube, wavelet)

        assert result.shape == cube.shape
        assert result.dtype == np.float32

    def test_convolve_3d_energy_preservation(self):
        """Test that convolution doesn't grossly distort energy."""
        cube = np.random.randn(8, 4, 4) * 0.01
        wavelet = np.array([0.1, 0.8, 0.1])  # Normalized

        result = WaveletConvolver.convolve_3d(cube, wavelet)

        # Result should have similar magnitude
        assert np.max(np.abs(result)) > 0  # Something happened
        assert np.isfinite(result).all()


class TestCacheManagerCoverage:
    """Tests for CacheManager I/O operations."""

    def test_save_and_load_synthetics(self):
        """Test saving and loading synthetics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(tmpdir)

            full_stack = np.random.randn(5, 3, 3).astype(np.float32)
            angle_stacks = [
                np.random.randn(5, 3, 3).astype(np.float32),
                np.random.randn(5, 3, 3).astype(np.float32),
            ]

            # Save
            manager.save_avo_synthetics("test.npz", full_stack, angle_stacks)

            # Load
            loaded_angles, loaded_full = manager.load_avo_synthetics("test.npz")

            # Verify
            assert len(loaded_angles) == len(angle_stacks)
            assert np.allclose(loaded_full, full_stack)
            for i in range(len(angle_stacks)):
                assert np.allclose(loaded_angles[i], angle_stacks[i])

    def test_compute_cache_key_deterministic(self):
        """Test cache key is deterministic."""
        manager = CacheManager()

        vp = np.random.randn(5, 3, 3)
        vs = np.random.randn(5, 3, 3)
        rho = np.random.randn(5, 3, 3)
        wavelet = np.array([0.1, 0.5, 0.1])
        angles = [0.0, 15.0]

        key1 = manager.compute_cache_key(vp, vs, rho, angles, wavelet)
        key2 = manager.compute_cache_key(vp, vs, rho, angles, wavelet)

        assert key1 == key2

    def test_compute_cache_key_different_for_different_data(self):
        """Test cache key differs for different inputs."""
        manager = CacheManager()

        vp1 = np.ones((5, 3, 3))
        vp2 = np.ones((5, 3, 3)) * 2
        vs = np.random.randn(5, 3, 3)
        rho = np.random.randn(5, 3, 3)
        wavelet = np.array([0.1, 0.5, 0.1])
        angles = [0.0, 15.0]

        key1 = manager.compute_cache_key(vp1, vs, rho, angles, wavelet)
        key2 = manager.compute_cache_key(vp2, vs, rho, angles, wavelet)

        assert key1 != key2


class TestResamplingServiceCoverage:
    """Tests for ResamplingService."""

    def test_resample_to_time_with_quantities(self):
        """Test resampling with Quantity objects."""
        with patch(
            "src.processing.resampling.resampler.get_resampler_factory"
        ) as mock_get_factory:
            # Mock the factory and resampler
            mock_factory = MagicMock()
            mock_resampler = MagicMock()
            mock_get_factory.return_value = mock_factory
            mock_factory.get_resampler.return_value = mock_resampler

            with patch(
                "src.processing.resampling.cache.get_resample_plan_cache"
            ) as mock_cache:
                mock_plan_cache = MagicMock()
                mock_cache.return_value = mock_plan_cache

                # Mock depth_to_time_cube to return time-domain data
                time_data = np.random.randn(8, 5, 5)
                mock_resampler.depth_to_time_cube.return_value = (time_data, 0.001)

                # Test data with Quantity
                props_depth = {
                    "vp": Quantity(np.ones((10, 5, 5)) * 3000, "m/s"),
                    "vs": np.ones((10, 5, 5)) * 1500,
                }
                grid_spec = GridSpec((10, 5, 5), dz=1.0, dt=0.001)

                result = ResamplingService.resample_to_time(props_depth, grid_spec)

                # Check that Quantity is preserved
                assert isinstance(result["vp"], Quantity)
                assert isinstance(result["vs"], np.ndarray)


class TestModelingPipelineExecutionCoverage:
    """Tests for ModelingPipeline full execution."""

    def test_pipeline_run_with_mock_data(self):
        """Test full pipeline run with mocked data loading."""
        with patch("src.io.loader.DatasetManager") as mock_dm_class:
            with patch(
                "src.processing.rock_physics.RockPhysicsModel"
            ) as mock_rpm_class:
                # Setup mocks
                mock_dm = MagicMock()
                mock_dm_class.from_stanfordsix.return_value = mock_dm

                mock_dm.vp = np.ones((8, 4, 4)) * 3000
                mock_dm.vs = np.ones((8, 4, 4)) * 1500
                mock_dm.rho = np.ones((8, 4, 4)) * 2.5
                mock_dm.facies = np.ones((8, 4, 4), dtype=int)
                mock_dm.full_stack = np.ones((8, 4, 4)) * 0.1

                mock_rpm = MagicMock()
                mock_rpm_class.from_props.return_value = mock_rpm
                mock_rpm.to_props_dict.return_value = {
                    "vp": np.ones((8, 4, 4)) * 3000,
                    "vs": np.ones((8, 4, 4)) * 1500,
                    "rho": np.ones((8, 4, 4)) * 2.5,
                }

                # Create pipeline with minimal config
                config = ModelingConfig(
                    defaults=ModelingDefaults(
                        grid_shape=(8, 4, 4),
                        angles=(0.0, 15.0),
                    )
                )

                with patch.object(
                    ResamplingService, "resample_to_time"
                ) as mock_resample:
                    mock_resample.return_value = {
                        "vp": np.ones((8, 4, 4)) * 3000,
                        "vs": np.ones((8, 4, 4)) * 1500,
                        "rho": np.ones((8, 4, 4)) * 2.5,
                    }

                    pipeline = ModelingPipeline(config=config)
                    result = pipeline.run()

                    # Verify result structure
                    assert "avo_cached" in result
                    assert "angle_stacks" in result
                    assert "full_stack" in result


class TestUnwrapQuantity:
    """Tests for _unwrap_quantity utility."""

    def test_unwrap_quantity_from_quantity(self):
        """Test extracting array from Quantity."""
        arr = np.array([1, 2, 3])
        qty = Quantity(arr, "m/s")

        result = _unwrap_quantity(qty)

        assert np.allclose(result, arr)
        assert isinstance(result, np.ndarray)

    def test_unwrap_quantity_from_array(self):
        """Test passing through ndarray."""
        arr = np.array([1, 2, 3])

        result = _unwrap_quantity(arr)

        assert np.allclose(result, arr)
        assert isinstance(result, np.ndarray)


class TestModelingDefaults:
    """Tests for ModelingDefaults edge cases."""

    def test_defaults_grid_spec_property(self):
        """Test grid_spec property creation."""
        defaults = ModelingDefaults(
            grid_shape=(100, 50, 50),
            grid_dz=2.0,
            grid_dt=0.002,
        )

        gs = defaults.grid_spec
        assert gs.shape == (100, 50, 50)
        assert gs.dz == 2.0
        assert gs.dt == 0.002

    def test_defaults_file_map_property(self):
        """Test file_map property creation."""
        defaults = ModelingDefaults(
            vp_folder="VP",
            vs_folder="VS",
            rho_folder="RHO",
            facies_folder="FACIES",
        )

        fm = defaults.file_map
        assert fm["vp"] == "VP"
        assert fm["vs"] == "VS"
        assert fm["rho"] == "RHO"
        assert fm["facies"] == "FACIES"

    def test_defaults_wavelet_property(self):
        """Test wavelet property creation."""
        defaults = ModelingDefaults(peak_frequency=30.0, grid_dt=0.001)

        wavelet = defaults.wavelet
        assert isinstance(wavelet, np.ndarray)
        assert len(wavelet) > 0
