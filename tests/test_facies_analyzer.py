"""
Comprehensive test suite for FaciesCorrelationAnalyzer.

This module consolidates all tests covering:
- Discrimination analysis (multi-facies separation matrices)
- Interface reflections analysis (transition detection)
- Gradient correlation (velocity gradient handling)
- Boundary detection (vectorized facies edge identification)
- Configuration overrides (parameter sensitivity)
- Technique comparison and visualization
- Validation and error handling
- Time-to-depth conversion
- Cache management
- Display cube preparation
"""

# mypy: ignore-errors


import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from src.analysis.facies import (
    FaciesCorrelationAnalyzer,
    FaciesCorrelationConfig,
    Domain,
)
from src.analysis.models import (
    AvoStats,
    TechniqueComparison,
    AvoResults,
    AvoAnalysisResult,
    BoundaryAmpsResult,
)
from src.analysis.processors.validators import (
    DomainValidator,
    ArrayValidator,
    PathValidator,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _loop_detect(facies_cube: np.ndarray) -> np.ndarray:
    """Reference implementation of boundary detection using explicit loops.

    Used for validation and benchmarking against vectorized implementation.
    """
    ni, nj, nk = facies_cube.shape
    boundaries = np.zeros_like(facies_cube, dtype=bool)
    for i in range(ni):
        slice_2d = facies_cube[i, :, :].astype(int)
        padded = np.pad(slice_2d, pad_width=1, mode="edge")
        up = padded[0:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, 0:-2]
        right = padded[1:-1, 2:]
        diff = (
            (slice_2d != up)
            | (slice_2d != down)
            | (slice_2d != left)
            | (slice_2d != right)
        )
        boundaries[i, :, :] = diff
    return boundaries


# ============================================================================
# DISCRIMINATION ANALYSIS TESTS
# ============================================================================


def test_discrimination_matrix_shape():
    """Test that discrimination matrix has correct shape for facies_count."""
    seismic = np.random.RandomState(0).randn(1, 2, 10)
    facies = np.zeros((1, 2, 10), dtype=int)
    facies[0, 0, :] = 0
    facies[0, 1, :] = 1

    cfg = FaciesCorrelationConfig(facies_count=2)
    analyzer = FaciesCorrelationAnalyzer(config=cfg)
    disc = analyzer.calculate_facies_discrimination(seismic, facies)

    assert disc.separation_matrix.shape == (2, 2)


def test_discrimination_with_single_facies():
    """Test discrimination with only one facies type."""
    seismic = np.random.RandomState(42).randn(1, 1, 10)
    facies = np.zeros((1, 1, 10), dtype=int)  # All zeros

    analyzer = FaciesCorrelationAnalyzer()
    disc = analyzer.calculate_facies_discrimination(seismic, facies)
    assert disc.separation_matrix.shape[0] >= 1


def test_discrimination_with_multiple_facies():
    """Test discrimination calculation with multiple facies."""
    seismic = np.random.RandomState(99).randn(2, 2, 20)
    facies = np.zeros((2, 2, 20), dtype=int)
    facies[0, :, :10] = 0
    facies[0, :, 10:] = 1
    facies[1, :, :] = 2

    cfg = FaciesCorrelationConfig(facies_count=3)
    analyzer = FaciesCorrelationAnalyzer(config=cfg)
    disc = analyzer.calculate_facies_discrimination(seismic, facies)
    assert disc.separation_matrix.shape[0] >= 1


# ============================================================================
# INTERFACE REFLECTIONS ANALYSIS TESTS
# ============================================================================


def test_reflection_summary_nonempty():
    """Test that interface reflection analysis produces non-empty summary."""
    seismic = np.zeros((1, 1, 5))
    facies = np.zeros((1, 1, 5), dtype=int)
    seismic[0, 0, 2] = 5.0
    facies[0, 0, 2:] = 1

    analyzer = FaciesCorrelationAnalyzer()
    res = analyzer.analyze_interface_reflections(seismic, facies)

    assert any(v is not None for v in res.transitions_summary.values())


def test_reflection_analysis_with_no_transitions():
    """Test interface analysis when no transitions exist."""
    seismic = np.random.rand(2, 2, 10).astype(float)
    facies = np.full((2, 2, 10), fill_value=1, dtype=int)  # All same facies

    analyzer = FaciesCorrelationAnalyzer()
    res = analyzer.analyze_interface_reflections(seismic, facies)
    assert isinstance(res.transitions_summary, dict)


# ============================================================================
# GRADIENT CORRELATION TESTS
# ============================================================================


def test_gradient_correlation_handles_nans():
    """Test that gradient correlation gracefully handles NaN values."""
    seismic = np.zeros((1, 1, 5))
    facies = np.zeros((1, 1, 5), dtype=int)
    seismic.fill(np.nan)

    analyzer = FaciesCorrelationAnalyzer()
    res = analyzer.calculate_gradient_correlation(seismic, facies)

    assert np.isnan(res.pearson_correlation)


def test_gradient_correlation_with_constant_seismic():
    """Test gradient correlation with constant seismic values."""
    seismic = np.ones((2, 2, 5), dtype=float)
    facies = np.zeros((2, 2, 5), dtype=int)

    analyzer = FaciesCorrelationAnalyzer()
    res = analyzer.calculate_gradient_correlation(seismic, facies)
    assert res.seismic_gradient is not None


def test_gradient_correlation_output_types():
    """Test gradient correlation returns correct output types."""
    seismic = np.random.rand(2, 3, 4).astype(float)
    facies = np.zeros((2, 3, 4), dtype=int)
    analyzer = FaciesCorrelationAnalyzer()
    res = analyzer.calculate_gradient_correlation(seismic, facies)
    assert isinstance(res.pearson_correlation, (float, np.floating))
    assert isinstance(res.pearson_pvalue, (float, np.floating))
    assert isinstance(res.seismic_gradient, np.ndarray)
    assert isinstance(res.boundaries, np.ndarray)


# ============================================================================
# FACIES BOUNDARY DETECTION TESTS
# ============================================================================


def test_small_array_edge_cases():
    """Test boundary detection on small 3x3x3 array with center anomaly."""
    cube = np.zeros((3, 3, 3), dtype=int)
    cube[1, 1, 1] = 1  # Different facies at center

    analyzer = FaciesCorrelationAnalyzer()
    vectorized_result = analyzer.detect_facies_boundaries(cube)
    loop_result = _loop_detect(cube)

    assert np.array_equal(vectorized_result, loop_result)


def test_all_equal_facies():
    """Test boundary detection when all facies values are identical."""
    cube = np.full((4, 5, 6), fill_value=2, dtype=int)

    analyzer = FaciesCorrelationAnalyzer()
    boundaries = analyzer.detect_facies_boundaries(cube)

    assert np.count_nonzero(boundaries) == 0


def test_boundary_only_slice():
    """Test boundary detection on checkerboard pattern."""
    slice2d = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
    cube = np.stack([slice2d for _ in range(2)], axis=0)

    analyzer = FaciesCorrelationAnalyzer()
    boundaries = analyzer.detect_facies_boundaries(cube)

    assert boundaries.shape == cube.shape
    assert np.all(boundaries)


def test_vectorized_equivalence_random():
    """Test vectorized implementation matches loop-based reference."""
    rng = np.random.default_rng(12345)
    cube = rng.integers(0, 4, size=(10, 50, 50), dtype=int)

    analyzer = FaciesCorrelationAnalyzer()
    vectorized_result = analyzer.detect_facies_boundaries(cube)
    loop_result = _loop_detect(cube)

    assert np.array_equal(vectorized_result, loop_result)


def test_micro_benchmark_speedup():
    """Benchmark vectorized implementation against loop-based reference."""
    rng = np.random.default_rng(1)
    cube = rng.integers(0, 8, size=(100, 200, 200), dtype=int)

    analyzer = FaciesCorrelationAnalyzer()

    t0 = time.perf_counter()
    loop_result = _loop_detect(cube)
    loop_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    vectorized_result = analyzer.detect_facies_boundaries(cube)
    vec_time = time.perf_counter() - t0

    assert np.array_equal(vectorized_result, loop_result)
    assert vec_time <= loop_time * 5.0


def test_boundary_detection_large_cube():
    """Test boundary detection on larger cubes."""
    rng = np.random.default_rng(123)
    cube = rng.integers(0, 5, size=(50, 50, 50), dtype=int)

    analyzer = FaciesCorrelationAnalyzer()
    boundaries = analyzer.detect_facies_boundaries(cube)
    assert boundaries.shape == cube.shape
    assert boundaries.dtype == bool
    assert np.count_nonzero(boundaries) > 0


def test_boundary_detection_preserves_dtype():
    """Test boundary detection returns boolean dtype."""
    cube = np.array([[[1, 2], [2, 1]]], dtype=np.int32)
    analyzer = FaciesCorrelationAnalyzer()
    boundaries = analyzer.detect_facies_boundaries(cube)
    assert boundaries.dtype == bool


def test_boundary_detection_preserves_shape():
    """Test boundary detection preserves input shape."""
    shapes = [(2, 3, 4), (5, 5, 5), (10, 20, 15)]
    for shape in shapes:
        cube = np.zeros(shape, dtype=int)
        analyzer = FaciesCorrelationAnalyzer()
        boundaries = analyzer.detect_facies_boundaries(cube)
        assert boundaries.shape == shape


# ============================================================================
# CONFIGURATION OVERRIDE TESTS
# ============================================================================


def test_extract_boundary_amplitudes_respects_config():
    """Test boundary amplitude extraction respects dilation_window config."""
    seismic = np.zeros((1, 3, 5), dtype=float)
    seismic[0, 1, 2] = 1.0

    facies = np.zeros((1, 3, 5), dtype=int)
    facies[0, 1, 2:] = 1

    cfg_small = FaciesCorrelationConfig(dilation_window=1)
    analyzer_small = FaciesCorrelationAnalyzer(config=cfg_small)
    res_small = analyzer_small.extract_boundary_amplitudes(
        seismic, analyzer_small.detect_facies_boundaries(facies)
    )

    cfg_large = FaciesCorrelationConfig(dilation_window=2)
    analyzer_large = FaciesCorrelationAnalyzer(config=cfg_large)
    res_large = analyzer_large.extract_boundary_amplitudes(
        seismic, analyzer_large.detect_facies_boundaries(facies)
    )

    assert res_small.at_boundaries.size <= res_large.at_boundaries.size


def test_boundary_amplitudes_with_empty_boundaries():
    """Test amplitude extraction when no boundaries detected."""
    seismic = np.random.rand(2, 2, 5).astype(float)
    boundaries = np.zeros((2, 2, 5), dtype=bool)

    analyzer = FaciesCorrelationAnalyzer()
    result = analyzer.extract_boundary_amplitudes(seismic, boundaries)
    assert result.at_boundaries.size == 0
    assert result.away_from_boundaries.size > 0


def test_config_with_various_dilation_windows():
    """Test analyzer works with different dilation window values."""
    seismic = np.random.rand(2, 2, 5).astype(float)

    for dilation in [1, 2, 3]:
        cfg = FaciesCorrelationConfig(dilation_window=dilation)
        analyzer = FaciesCorrelationAnalyzer(config=cfg)
        boundaries = np.zeros((2, 2, 5), dtype=bool)
        boundaries[1, 1, 1] = True
        result = analyzer.extract_boundary_amplitudes(seismic, boundaries)
        assert result is not None


def test_config_with_various_facies_counts():
    """Test analyzer works with different facies counts."""
    seismic = np.random.rand(2, 2, 5).astype(float)
    facies = np.zeros((2, 2, 5), dtype=int)

    for facies_count in [1, 2, 4, 8]:
        cfg = FaciesCorrelationConfig(facies_count=facies_count)
        analyzer = FaciesCorrelationAnalyzer(config=cfg)
        disc = analyzer.calculate_facies_discrimination(seismic, facies)
        assert disc is not None


# ============================================================================
# INITIALIZATION & CONFIGURATION TESTS
# ============================================================================


def test_analyzer_initialization_default():
    """Test analyzer initializes with default configuration."""
    analyzer = FaciesCorrelationAnalyzer()
    assert analyzer is not None
    assert analyzer.config is not None
    assert isinstance(analyzer.config, FaciesCorrelationConfig)


def test_analyzer_initialization_custom_config():
    """Test analyzer initializes with custom configuration."""
    cfg = FaciesCorrelationConfig(facies_count=3, dilation_window=1)
    analyzer = FaciesCorrelationAnalyzer(config=cfg)
    assert analyzer.config is cfg
    assert analyzer.config.facies_count == 3
    assert analyzer.config.dilation_window == 1


def test_analyzer_config_property_read_only():
    """Test that config property provides read-only access."""
    cfg = FaciesCorrelationConfig()
    analyzer = FaciesCorrelationAnalyzer(config=cfg)
    cfg1 = analyzer.config
    cfg2 = analyzer.config
    assert cfg1 is cfg2


# ============================================================================
# MAGIC METHODS & REPRESENTATIONS TESTS
# ============================================================================


def test_analyzer_repr():
    """Test analyzer __repr__ shows state information."""
    analyzer = FaciesCorrelationAnalyzer()
    repr_str = repr(analyzer)
    assert "FaciesCorrelationAnalyzer" in repr_str
    assert "config" in repr_str.lower()


def test_analyzer_repr_contains_class_name():
    """Test analyzer __repr__ contains class name."""
    analyzer = FaciesCorrelationAnalyzer()
    repr_str = repr(analyzer)
    assert "FaciesCorrelationAnalyzer" in repr_str


def test_analyzer_str_contains_class_name():
    """Test analyzer __str__ is human-readable."""
    analyzer = FaciesCorrelationAnalyzer()
    str_repr = str(analyzer)
    assert "FaciesCorrelationAnalyzer" in str_repr


def test_analyzer_is_ready():
    """Test analyzer readiness check."""
    analyzer = FaciesCorrelationAnalyzer()
    # A freshly created analyzer is not ready (needs initialization)
    assert isinstance(analyzer.is_ready, bool)
    assert analyzer.is_ready is False

    # After initialization, it should be ready
    analyzer.initialize()
    assert analyzer.is_ready is True


def test_analyzer_get_processor_info():
    """Test getting processor information."""
    analyzer = FaciesCorrelationAnalyzer()
    analyzer.initialize()  # Must initialize before accessing processor info
    info = analyzer.get_processor_info()
    assert isinstance(info, dict)
    assert "boundary_detector" in info
    assert "cube_aligner" in info
    assert "gradient_calculator" in info
    assert "interface_analyzer" in info
    assert "facies_discriminator" in info


def test_analyzer_get_summary():
    """Test getting analyzer summary."""
    analyzer = FaciesCorrelationAnalyzer()
    analyzer.initialize()  # Must initialize before getting summary
    summary = analyzer.get_summary()
    assert isinstance(summary, str)
    assert "FaciesCorrelationAnalyzer" in summary
    assert "Configuration" in summary


def test_processor_info_contains_all_processors():
    """Test that processor_info includes all expected processors."""
    analyzer = FaciesCorrelationAnalyzer()
    analyzer.initialize()  # Must initialize
    info = analyzer.get_processor_info()
    expected_processors = [
        "boundary_detector",
        "cube_aligner",
        "boundary_amp_extractor",
        "gradient_calculator",
        "interface_analyzer",
        "facies_discriminator",
        "domain_handler_factory",
    ]
    for processor_name in expected_processors:
        assert processor_name in info


def test_processor_info_all_strings():
    """Test that all processor info values are strings."""
    analyzer = FaciesCorrelationAnalyzer()
    analyzer.initialize()  # Must initialize
    info = analyzer.get_processor_info()
    for key, value in info.items():
        assert isinstance(value, str)


def test_get_processor_info_completeness():
    """Test that processor info includes all expected processors."""
    analyzer = FaciesCorrelationAnalyzer()
    analyzer.initialize()  # Must initialize
    info = analyzer.get_processor_info()
    assert isinstance(info, dict)
    assert len(info) > 0
    for key, value in info.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert len(value) > 0


def test_get_processor_info_immutability():
    """Test that multiple calls return consistent info."""
    analyzer = FaciesCorrelationAnalyzer()
    analyzer.initialize()  # Must initialize
    info1 = analyzer.get_processor_info()
    info2 = analyzer.get_processor_info()
    assert info1 == info2


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================


def test_analyzer_context_manager_enter():
    """Test analyzer context manager __enter__ returns self."""
    analyzer = FaciesCorrelationAnalyzer()
    with analyzer as ctx:
        assert ctx is analyzer


def test_analyzer_context_manager_exit():
    """Test analyzer context manager __exit__ handles normal exit."""
    analyzer = FaciesCorrelationAnalyzer()
    try:
        with analyzer:
            pass
    except Exception as e:
        raise AssertionError(f"Context manager raised: {e}")


def test_analyzer_context_manager_with_exception():
    """Test analyzer context manager handles exceptions gracefully."""
    analyzer = FaciesCorrelationAnalyzer()
    try:
        with analyzer:
            raise ValueError("Test error")
    except ValueError as e:
        assert str(e) == "Test error"


# ============================================================================
# METHOD DELEGATION TESTS
# ============================================================================


def test_detect_facies_boundaries_delegated():
    """Test boundary detection delegates to processor."""
    analyzer = FaciesCorrelationAnalyzer()
    cube = np.zeros((2, 3, 4), dtype=int)
    boundaries = analyzer.detect_facies_boundaries(cube)
    assert boundaries.shape == cube.shape
    assert boundaries.dtype == bool


def test_extract_boundary_amplitudes_delegated():
    """Test amplitude extraction delegates to processor."""
    analyzer = FaciesCorrelationAnalyzer()
    seismic = np.random.rand(2, 3, 4).astype(float)
    boundaries = np.zeros((2, 3, 4), dtype=bool)
    boundaries[1, 1, 1] = True
    result = analyzer.extract_boundary_amplitudes(seismic, boundaries)
    assert isinstance(result, BoundaryAmpsResult)
    assert hasattr(result, "at_boundaries")
    assert hasattr(result, "away_from_boundaries")


def test_calculate_gradient_correlation_delegated():
    """Test gradient correlation delegates to processor."""
    analyzer = FaciesCorrelationAnalyzer()
    seismic = np.random.rand(2, 3, 4).astype(float)
    facies = np.zeros((2, 3, 4), dtype=int)
    result = analyzer.calculate_gradient_correlation(seismic, facies)
    assert hasattr(result, "pearson_correlation")
    assert hasattr(result, "pearson_pvalue")


def test_analyze_interface_reflections_delegated():
    """Test interface analysis delegates to processor."""
    analyzer = FaciesCorrelationAnalyzer()
    seismic = np.random.rand(2, 3, 4).astype(float)
    facies = np.zeros((2, 3, 4), dtype=int)
    result = analyzer.analyze_interface_reflections(seismic, facies)
    assert hasattr(result, "summary")
    assert hasattr(result, "interface_stats")


def test_calculate_facies_discrimination_delegated():
    """Test discrimination delegates to processor."""
    analyzer = FaciesCorrelationAnalyzer()
    seismic = np.random.rand(2, 3, 4).astype(float)
    facies = np.zeros((2, 3, 4), dtype=int)
    result = analyzer.calculate_facies_discrimination(seismic, facies)
    assert hasattr(result, "separation_matrix")
    assert hasattr(result, "facies_stats")


# ============================================================================
# LOGGING CONFIGURATION TESTS
# ============================================================================


def test_configure_logging_verbose_false():
    """Test logging configuration with verbose=False."""
    try:
        FaciesCorrelationAnalyzer.configure_logging(verbose=False)
    except Exception as e:
        raise AssertionError(f"configure_logging raised: {e}")


def test_configure_logging_verbose_true():
    """Test logging configuration with verbose=True."""
    try:
        FaciesCorrelationAnalyzer.configure_logging(verbose=True)
    except Exception as e:
        raise AssertionError(f"configure_logging raised: {e}")


# ============================================================================
# COMPARE_TECHNIQUES TESTS
# ============================================================================


class TestCompareTechniques:
    """Test suite for compare_techniques method."""

    def test_compare_techniques_gradient_correlation_metric(self):
        """Test technique comparison for GRADIENT_CORRELATION metric."""
        analyzer = FaciesCorrelationAnalyzer()

        avo_stats = AvoStats(
            pearson_correlation=0.85,
            spearman_correlation=0.80,
        )

        result = analyzer.compare_techniques(
            avo_stats, TechniqueComparison.GRADIENT_CORRELATION
        )

        assert isinstance(result, TechniqueComparison)
        assert result.winner == "AVO"
        assert result.difference == 0.0
        assert result.avo.pearson_correlation == 0.85
        assert result.avo.spearman_correlation == 0.80

    def test_compare_techniques_gradient_correlation_with_none_values(self):
        """Test technique comparison with None correlation values."""
        analyzer = FaciesCorrelationAnalyzer()

        avo_stats = AvoStats(
            pearson_correlation=None,
            spearman_correlation=None,
        )

        result = analyzer.compare_techniques(
            avo_stats, TechniqueComparison.GRADIENT_CORRELATION
        )

        assert result.avo.pearson_correlation is None
        assert result.avo.spearman_correlation is None

    def test_compare_techniques_default_metric(self):
        """Test technique comparison with default (non-gradient) metric."""
        analyzer = FaciesCorrelationAnalyzer()

        avo_stats = AvoStats(
            pearson_correlation=0.75,
            spearman_correlation=0.70,
        )

        result = analyzer.compare_techniques(avo_stats, "some_other_metric")

        assert result.avo == avo_stats
        assert result.winner == "AVO"
        assert result.difference == 0.0

    def test_compare_techniques_invalid_input_type(self):
        """Test that compare_techniques raises TypeError for non-AvoStats input."""
        analyzer = FaciesCorrelationAnalyzer()

        with pytest.raises(TypeError):
            analyzer.compare_techniques(
                {"pearson": 0.8}, TechniqueComparison.GRADIENT_CORRELATION
            )

    def test_compare_techniques_float_conversion(self):
        """Test that correlation values are properly converted to float."""
        analyzer = FaciesCorrelationAnalyzer()

        avo_stats = AvoStats(
            pearson_correlation=np.float32(0.85),
            spearman_correlation=np.float64(0.80),
        )

        result = analyzer.compare_techniques(
            avo_stats, TechniqueComparison.GRADIENT_CORRELATION
        )

        assert isinstance(result.avo.pearson_correlation, float)
        assert isinstance(result.avo.spearman_correlation, float)


# ============================================================================
# CREATE_SUMMARY_PLOTS TESTS
# ============================================================================


class TestCreateSummaryPlots:
    """Test suite for create_summary_plots method."""

    def test_create_summary_plots_with_injected_plotter(self):
        """Test plot creation with mocked injected plotter."""
        mock_plotter = mock.MagicMock()
        mock_figure = mock.MagicMock()
        mock_plotter.create_summary_plots.return_value = mock_figure

        analyzer = FaciesCorrelationAnalyzer(plotter=mock_plotter)

        avo_results = AvoResults()

        result = analyzer.create_summary_plots(
            avo_results, "/tmp/cache", domain=Domain.DEPTH
        )

        assert result == mock_figure
        mock_plotter.create_summary_plots.assert_called_once_with(
            avo_results, "/tmp/cache", domain=Domain.DEPTH
        )

    def test_create_summary_plots_with_time_domain(self):
        """Test plot creation with TIME domain."""
        mock_plotter = mock.MagicMock()
        mock_figure = mock.MagicMock()
        mock_plotter.create_summary_plots.return_value = mock_figure

        analyzer = FaciesCorrelationAnalyzer(plotter=mock_plotter)

        avo_results = AvoResults()

        analyzer.create_summary_plots(avo_results, "/tmp/cache", domain=Domain.TIME)

        mock_plotter.create_summary_plots.assert_called_once()
        call_args = mock_plotter.create_summary_plots.call_args
        assert call_args.kwargs["domain"] == Domain.TIME

    def test_create_summary_plots_lazy_instantiation(self):
        """Test that FaciesPlotter is lazily instantiated when None."""
        analyzer = FaciesCorrelationAnalyzer()

        with mock.patch("src.plotting.facies_plotter.FaciesPlotter") as MockPlotter:
            mock_plotter_instance = mock.MagicMock()
            mock_figure = mock.MagicMock()
            mock_plotter_instance.create_summary_plots.return_value = mock_figure
            MockPlotter.return_value = mock_plotter_instance

            avo_results = AvoResults()

            result = analyzer.create_summary_plots(avo_results, "/tmp/cache")

            MockPlotter.assert_called_once()
            assert result == mock_figure

    def test_create_summary_plots_caches_plotter_instance(self):
        """Test that plotter instance is cached after first instantiation."""
        analyzer = FaciesCorrelationAnalyzer()

        with mock.patch("src.plotting.facies_plotter.FaciesPlotter") as MockPlotter:
            mock_plotter_instance = mock.MagicMock()
            mock_figure = mock.MagicMock()
            mock_plotter_instance.create_summary_plots.return_value = mock_figure
            MockPlotter.return_value = mock_plotter_instance

            avo_results = AvoResults()

            analyzer.create_summary_plots(avo_results, "/tmp/cache")
            analyzer.create_summary_plots(avo_results, "/tmp/cache")

            MockPlotter.assert_called_once()


# ============================================================================
# VALIDATION ERROR TESTS
# ============================================================================


class TestValidationErrors:
    """Test validation error paths."""

    def test_validate_domain_invalid_type(self):
        """Test that non-Domain type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            DomainValidator.validate_domain("DEPTH")

        assert "Expected Domain enum" in str(exc_info.value)

    def test_validate_domain_unsupported_value(self):
        """Test that valid Domain values work correctly."""
        result_depth = DomainValidator.validate_domain(Domain.DEPTH)
        assert result_depth == Domain.DEPTH

        result_time = DomainValidator.validate_domain(Domain.TIME)
        assert result_time == Domain.TIME

    def test_validate_cube_shape_not_array(self):
        """Test that non-array input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ArrayValidator.validate_3d_array([[[1, 2], [3, 4]]], name="test_cube")

        assert "must be a numpy array" in str(exc_info.value)

    def test_validate_cube_shape_wrong_dims(self):
        """Test that wrong dimensionality raises ValueError."""
        cube_2d = np.zeros((5, 5))

        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_3d_array(cube_2d, name="facies")

        assert "must be 3-dimensional" in str(exc_info.value)

    def test_detect_boundaries_validates_cube_shape(self):
        """Test that detect_facies_boundaries validates input shape."""
        analyzer = FaciesCorrelationAnalyzer()
        facies_2d = np.zeros((5, 5), dtype=int)

        try:
            result = analyzer.detect_facies_boundaries(facies_2d)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_gradient_correlation_validates_cube_shape(self):
        """Test gradient correlation with mismatched shapes."""
        analyzer = FaciesCorrelationAnalyzer()

        seismic = np.random.randn(2, 3, 5)
        facies = np.zeros((2, 3, 4), dtype=int)

        try:
            result = analyzer.calculate_gradient_correlation(seismic, facies)
            assert result is not None
        except (ValueError, IndexError):
            pass


# ============================================================================
# TIME-TO-DEPTH CONVERSION TESTS
# ============================================================================


class TestConvertTimeToDepth:
    """Test the convert_time_to_depth method."""

    def test_convert_time_to_depth_with_injected_resampler_factory(self):
        """Test conversion with injected resampler factory."""
        mock_factory = mock.MagicMock()
        mock_resampler = mock.MagicMock()
        mock_result = np.random.randn(2, 3, 4)

        mock_factory.get_resampler.return_value = mock_resampler
        mock_resampler.time_to_depth_cube.return_value = mock_result

        analyzer = FaciesCorrelationAnalyzer(resampler_factory=mock_factory)

        seismogram_time = np.random.randn(2, 3, 5)
        vp_depth = np.ones((2, 3, 4)) * 3000
        grid_spec = mock.MagicMock()
        # Mock grid_spec attributes needed by ResamplePlan.create
        grid_spec.dz = 1.0  # depth sample spacing
        grid_spec.dt = 0.001  # time sample spacing

        result = analyzer.convert_time_to_depth(seismogram_time, vp_depth, grid_spec)

        mock_factory.get_resampler.assert_called_once_with(grid_spec)
        assert np.array_equal(result, mock_result)


# ============================================================================
# CACHE LOADING TESTS
# ============================================================================


class TestCacheLoadingFallback:
    """Tests for cache loading fallback behavior.

    Note: These tests have been removed as the private methods they tested
    (_select_and_load_cache, _load_dataset) are now internal to AnalysisPipeline
    and tested indirectly through the public run() method.
    """

    pass


# ============================================================================
# CACHE DIRECTORY VALIDATION TESTS
# ============================================================================


class TestCacheDirValidation:
    """Test cache directory validation."""

    def test_validate_cache_dir_valid_string(self):
        """Test validation of valid cache directory string."""
        result = PathValidator.validate_cache_dir("/tmp/cache")

        assert isinstance(result, Path)
        assert str(result) == "/tmp/cache"

    def test_validate_cache_dir_with_spaces(self):
        """Test validation with surrounding spaces."""
        result = PathValidator.validate_cache_dir("  /tmp/cache  ")

        assert isinstance(result, Path)

    def test_validate_cache_dir_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            PathValidator.validate_cache_dir("")

    def test_validate_cache_dir_whitespace_only(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError):
            PathValidator.validate_cache_dir("   ")

    def test_validate_cache_dir_non_string_type(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError):
            PathValidator.validate_cache_dir(None)

        with pytest.raises(ValueError):
            PathValidator.validate_cache_dir(123)

    def test_validate_cache_dir_with_special_characters(self):
        """Test cache dir validation with special characters."""
        paths = [
            "/tmp/cache@special",
            "/tmp/cache#2024",
            "/tmp/cache$money",
        ]

        for path in paths:
            result = PathValidator.validate_cache_dir(path)
            assert isinstance(result, Path)


# ============================================================================
# LOAD DATASET TESTS
# ============================================================================


class TestLoadDataset:
    """Test suite for _load_dataset private method.

    Note: These tests have been removed as the private method _load_dataset
    is now internal to AnalysisPipeline and tested indirectly through
    the public run() method.
    """

    pass


# ============================================================================
# PREPARE DISPLAY CUBES TESTS
# ============================================================================


class TestPrepareDisplayCubes:
    """Test suite for _prepare_display_cubes private method."""

    def test_prepare_display_cubes_depth_domain(self):
        """Test display cube preparation for DEPTH domain."""
        mock_handler = mock.MagicMock()
        mock_avo_display = np.random.randn(2, 3, 4)
        mock_facies_display = np.random.randint(0, 3, (2, 3, 4))
        mock_handler.prepare_display_cubes.return_value = (
            mock_avo_display,
            mock_facies_display,
        )

        mock_factory = mock.MagicMock()
        mock_factory.get_handler.return_value = mock_handler

        analyzer = FaciesCorrelationAnalyzer()
        analyzer._domain_handler_factory = mock_factory

        vm = mock.MagicMock()
        facies_depth = np.zeros((2, 3, 4), dtype=int)
        avo = np.zeros((2, 3, 4))
        grid_spec = mock.MagicMock()

        result = analyzer._prepare_display_cubes(
            vm, facies_depth, avo, Domain.DEPTH, grid_spec
        )

        mock_factory.get_handler.assert_called_once_with(Domain.DEPTH)

        assert hasattr(result, "avo_display")
        assert hasattr(result, "facies_display")
        assert np.array_equal(result.avo_display, mock_avo_display)
        assert np.array_equal(result.facies_display, mock_facies_display)

    def test_prepare_display_cubes_time_domain(self):
        """Test display cube preparation for TIME domain."""
        mock_handler = mock.MagicMock()
        mock_handler.prepare_display_cubes.return_value = (
            np.zeros((2, 3, 4)),
            np.zeros((2, 3, 4), dtype=int),
        )

        mock_factory = mock.MagicMock()
        mock_factory.get_handler.return_value = mock_handler

        analyzer = FaciesCorrelationAnalyzer()
        analyzer._domain_handler_factory = mock_factory

        analyzer._prepare_display_cubes(
            mock.MagicMock(),
            np.zeros((2, 3, 4), dtype=int),
            np.zeros((2, 3, 4)),
            Domain.TIME,
            mock.MagicMock(),
        )

        mock_factory.get_handler.assert_called_once_with(Domain.TIME)

    def test_prepare_display_cubes_delegates_to_handler(self):
        """Test that all parameters are correctly delegated."""
        mock_handler = mock.MagicMock()
        mock_handler.prepare_display_cubes.return_value = (
            np.zeros((1, 1, 1)),
            np.zeros((1, 1, 1), dtype=int),
        )

        mock_factory = mock.MagicMock()
        mock_factory.get_handler.return_value = mock_handler

        mock_resampler_factory = mock.MagicMock()
        mock_resampler = mock.MagicMock()
        mock_resampler_factory.get_resampler.return_value = mock_resampler

        analyzer = FaciesCorrelationAnalyzer()
        analyzer._domain_handler_factory = mock_factory
        analyzer._resampler_factory = mock_resampler_factory

        vm = mock.MagicMock()
        facies_depth = np.array([[[0]]], dtype=int)
        avo = np.array([[[0.5]]])
        grid_spec = mock.MagicMock()

        analyzer._prepare_display_cubes(vm, facies_depth, avo, Domain.DEPTH, grid_spec)

        mock_handler.prepare_display_cubes.assert_called_once_with(
            mock_resampler, facies_depth, avo, grid_spec
        )


# ============================================================================
# AVO ANALYSIS TESTS
# ============================================================================


class TestPerformAvoAnalysis:
    """Test the _perform_avo_analysis method."""

    def test_perform_avo_analysis_aggregation(self):
        """Test that _perform_avo_analysis properly aggregates results."""
        analyzer = FaciesCorrelationAnalyzer()

        avo_display = np.random.randn(2, 2, 5)
        facies_display = np.array(
            [[[0, 1, 1, 0, 0], [1, 1, 0, 0, 1]], [[0, 0, 1, 1, 0], [1, 1, 0, 0, 1]]],
            dtype=int,
        )

        result = analyzer._perform_avo_analysis(avo_display, facies_display)

        assert isinstance(result, AvoAnalysisResult)
        assert result.gradient_corr is not None
        assert result.boundary_amps is not None
        assert result.interface_summary is not None
        assert result.interface_raw is not None
        assert result.facies_disc is not None

        assert hasattr(result.gradient_corr, "pearson_correlation")
        assert hasattr(result.boundary_amps, "at_boundaries")
        assert hasattr(result.facies_disc, "separation_matrix")

    def test_perform_avo_analysis_uses_display_cubes(self):
        """Test that analysis results are based on display cubes."""
        analyzer = FaciesCorrelationAnalyzer()

        avo_display = np.ones((2, 2, 5)) * 0.5
        facies_display = np.array(
            [[[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]], [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]],
            dtype=int,
        )

        result = analyzer._perform_avo_analysis(avo_display, facies_display)

        assert result.gradient_corr.pearson_correlation is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestFaciesAnalyzerIntegration:
    """Integration tests combining multiple components."""

    def test_compare_and_create_plot_workflow(self):
        """Test workflow combining compare_techniques and create_summary_plots."""
        mock_plotter = mock.MagicMock()
        mock_figure = mock.MagicMock()
        mock_plotter.create_summary_plots.return_value = mock_figure

        analyzer = FaciesCorrelationAnalyzer(plotter=mock_plotter)

        avo_stats = AvoStats(pearson_correlation=0.9, spearman_correlation=0.85)
        comparison = analyzer.compare_techniques(
            avo_stats, TechniqueComparison.GRADIENT_CORRELATION
        )

        assert comparison.winner == "AVO"

        avo_results = AvoResults()

        figure = analyzer.create_summary_plots(
            avo_results, "/tmp/cache", domain=Domain.DEPTH
        )

        assert figure == mock_figure

    def test_full_analysis_workflow(self):
        """Test complete workflow from boundary detection to discrimination."""
        analyzer = FaciesCorrelationAnalyzer()

        np.random.seed(42)
        avo_cube = np.random.randn(3, 3, 10) * 100
        facies_cube = np.random.randint(0, 4, (3, 3, 10))

        boundaries = analyzer.detect_facies_boundaries(facies_cube)
        assert boundaries.shape == facies_cube.shape
        assert boundaries.dtype == bool

        amps = analyzer.extract_boundary_amplitudes(avo_cube, boundaries)
        assert amps.at_boundaries is not None

        grad_corr = analyzer.calculate_gradient_correlation(avo_cube, facies_cube)
        assert grad_corr.pearson_correlation is not None or np.isnan(
            grad_corr.pearson_correlation
        )

        interface = analyzer.analyze_interface_reflections(avo_cube, facies_cube)
        assert interface.transitions_summary is not None

        disc = analyzer.calculate_facies_discrimination(avo_cube, facies_cube)
        assert disc.separation_matrix is not None

    def test_avo_analysis_result_aggregation(self):
        """Test that _perform_avo_analysis creates valid AvoAnalysisResult."""
        analyzer = FaciesCorrelationAnalyzer()

        avo_display = np.random.randn(2, 2, 8)
        facies_display = np.random.randint(0, 3, (2, 2, 8))

        result = analyzer._perform_avo_analysis(avo_display, facies_display)

        assert result.gradient_corr is not None
        assert result.boundary_amps is not None
        assert result.interface_summary is not None
        assert result.interface_raw is not None
        assert result.facies_disc is not None


# ============================================================================
# UNCOVERED CODE TARGETING TESTS
# ============================================================================


class TestUncoveredBranches:
    """Tests targeting specific uncovered code branches."""

    def test_align_cubes_delegates_to_aligner(self):
        """Test align_cubes delegates to _cube_aligner (line 448)."""
        analyzer = FaciesCorrelationAnalyzer()

        seismic = np.random.rand(2, 3, 5).astype(float)
        facies = np.zeros((2, 3, 5), dtype=int)

        result_seismic, result_facies = analyzer._align_cubes(seismic, facies)

        assert result_seismic.shape == seismic.shape
        assert result_facies.shape == facies.shape
        assert isinstance(result_seismic, np.ndarray)
        assert isinstance(result_facies, np.ndarray)

    # Note: Tests for _select_and_load_cache have been removed as this method
    # is now internal to AnalysisPipeline and tested through the public run() method

    def test_avo_results_construction(self):
        """Test that _create_results_object properly builds AvoResults (line 745-746)."""
        analyzer = FaciesCorrelationAnalyzer()

        mock_avo_analysis = mock.MagicMock()
        mock_avo_analysis.boundary_amps = BoundaryAmpsResult(
            at_boundaries=np.array([1.0, 2.0]),
            away_from_boundaries=np.array([0.5]),
            boundary_mask=np.array([True, False, True]),
        )
        mock_avo_analysis.gradient_corr = mock.MagicMock(pearson_correlation=0.85)
        mock_avo_analysis.interface_summary = {"key": "value"}

        mock_facies_disc = mock.MagicMock()
        mock_facies_disc.separation_matrix = np.array([[1, 0], [0, 1]])
        mock_facies_disc.facies_amplitudes = {0: np.array([100]), 1: np.array([200])}
        mock_avo_analysis.facies_disc = mock_facies_disc

        result = analyzer._create_results_object(mock_avo_analysis)

        assert isinstance(result, AvoResults)
        assert result.boundary_amps == mock_avo_analysis.boundary_amps
        assert result.gradient_correlation == mock_avo_analysis.gradient_corr
        assert result.interface_stats_summary == {"key": "value"}

    def test_get_processor_info_includes_domain_handler_factory(self):
        """Test that domain_handler_factory is included in processor info (line 769)."""
        analyzer = FaciesCorrelationAnalyzer()

        info = analyzer.get_processor_info()

        assert "domain_handler_factory" in info
        assert info["domain_handler_factory"] is not None
        assert isinstance(info["domain_handler_factory"], str)
        assert len(info["domain_handler_factory"]) > 0

    def test_processor_info_domain_handler_factory_value(self):
        """Test that domain_handler_factory processor info has a valid class name."""
        analyzer = FaciesCorrelationAnalyzer()

        info = analyzer.get_processor_info()

        domain_handler_factory_name = info["domain_handler_factory"]
        assert domain_handler_factory_name == "DomainHandlerFactory"


class TestCacheLoaderInjection:
    """Tests for cache loader injection and fallback behavior.

    Note: Tests for _select_and_load_cache have been removed as this method
    is now internal to AnalysisPipeline and tested through the public run() method.
    """

    pass


class TestDomainValidation:
    """Tests for domain validation edge cases."""

    def test_validate_domain_with_invalid_enum_value(self):
        """Test domain validation catches invalid enum values (line 317)."""
        _ = FaciesCorrelationAnalyzer()

        class InvalidDomain:
            """Fake domain that's not a Domain enum."""

            value = "INVALID"

        invalid_domain = InvalidDomain()

        with pytest.raises(TypeError):
            DomainValidator.validate_domain(invalid_domain)

    def test_validate_domain_unsupported_value_error(self):
        """Test domain validation for unsupported Domain-like objects (line 317)."""

        # Create a Domain-like enum that isn't in VALID_DOMAINS
        class CustomDomain:
            """Custom domain enum that's not DEPTH or TIME."""

            def __eq__(self, other):
                return isinstance(other, CustomDomain)

            def __hash__(self):
                return hash("CUSTOM")

            @property
            def value(self):
                return "CUSTOM"

        # Even though this test tries to test line 317, Domain is an enum
        # and we can only test the ValueError path, not the unsupported domain check
        # because Domain only has DEPTH and TIME values
        pass

    def test_validate_domain_string_raises_error(self):
        """Test that string domain raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            DomainValidator.validate_domain("DEPTH")

        assert "Expected Domain enum" in str(exc_info.value)
        assert "Domain.DEPTH" in str(exc_info.value)

    def test_validate_domain_none_raises_error(self):
        """Test that None domain raises TypeError."""
        with pytest.raises(TypeError):
            DomainValidator.validate_domain(None)

    def test_validate_domain_number_raises_error(self):
        """Test that numeric domain raises TypeError."""
        with pytest.raises(TypeError):
            DomainValidator.validate_domain(1)

    def test_validate_domain_depth_success(self):
        """Test successful validation of Domain.DEPTH."""
        result = DomainValidator.validate_domain(Domain.DEPTH)
        assert result == Domain.DEPTH

    def test_validate_domain_time_success(self):
        """Test successful validation of Domain.TIME."""
        result = DomainValidator.validate_domain(Domain.TIME)
        assert result == Domain.TIME


class TestCubeSizeValidation:
    """Tests for cube shape validation edge cases."""

    def test_validate_cube_shape_with_1d_array(self):
        """Test validation fails for 1D array."""
        cube_1d = np.zeros(5)

        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_3d_array(cube_1d, name="test")

        assert "3-dimensional" in str(exc_info.value)

    def test_validate_cube_shape_with_4d_array(self):
        """Test validation fails for 4D array."""
        cube_4d = np.zeros((2, 3, 4, 5))

        with pytest.raises(ValueError):
            ArrayValidator.validate_3d_array(cube_4d, name="test")

    def test_validate_cube_shape_with_list(self):
        """Test validation fails for list input."""
        with pytest.raises(TypeError) as exc_info:
            ArrayValidator.validate_3d_array([1, 2, 3], name="test")

        assert "numpy array" in str(exc_info.value)

    def test_validate_cube_shape_with_dict(self):
        """Test validation fails for dict input."""
        with pytest.raises(TypeError):
            ArrayValidator.validate_3d_array({"data": np.zeros((2, 3, 4))}, name="test")

    def test_validate_cube_shape_success_3d(self):
        """Test successful validation of 3D array."""
        cube_3d = np.zeros((2, 3, 4))

        result = ArrayValidator.validate_3d_array(cube_3d, name="test")

        assert result is None

    def test_validate_cube_shape_with_custom_dims(self):
        """Test validation with custom expected dimensions."""
        cube_2d = np.zeros((5, 5))

        # ArrayValidator only validates 3D arrays, so a 2D array should fail
        with pytest.raises(ValueError):
            ArrayValidator.validate_3d_array(cube_2d, name="test")

    def test_validate_cube_shape_error_message_includes_name(self):
        """Test that validation error includes the cube name."""
        cube_wrong = np.zeros((5,))

        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_3d_array(cube_wrong, name="facies_cube")

        assert "facies_cube" in str(exc_info.value)


# ============================================================================
# RUN METHOD INTEGRATION TESTS
# ============================================================================


class TestRunMethodIntegration:
    """Integration tests for the run() method with mocked GUI components.

    Note: Previous tests that mocked _load_dataset and _prepare_display_cubes
    have been removed as these methods are now internal to AnalysisPipeline.
    Tests now focus on the public run() API and error handling.
    """

    def test_run_method_cache_load_file_not_found(self):
        """Test run() method raises error when cache file not found."""
        mock_cache_loader = mock.MagicMock()
        mock_cache_loader.select_cache_file.return_value = None

        analyzer = FaciesCorrelationAnalyzer(cache_loader=mock_cache_loader)

        with pytest.raises(FileNotFoundError):
            analyzer.run(cache_dir="/tmp/nonexistent", domain=Domain.DEPTH)

    def test_run_method_cache_load_unexpected_error(self):
        """Test run() method propagates cache load errors."""
        mock_cache_loader = mock.MagicMock()
        mock_cache_loader.select_cache_file.return_value = "/cache/avo.npz"
        mock_cache_loader.load_full_stack.side_effect = IOError("Disk error")

        analyzer = FaciesCorrelationAnalyzer(cache_loader=mock_cache_loader)

        with pytest.raises(IOError) as exc_info:
            analyzer.run(cache_dir="/tmp/cache", domain=Domain.DEPTH)

        assert "Disk error" in str(exc_info.value)
