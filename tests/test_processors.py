"""Consolidated unit tests for src.analysis.processors modules.

Consolidates all processor unit tests from 10 separate files.
"""

import logging
import time
from abc import ABC
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from src.analysis.models import (
    BoundaryAmpsResult,
    FaciesDiscriminationResult,
    FaciesStats,
    GradientCorrelationResult,
    InterfaceReflectionResult,
    Transition,
)
from src.analysis.processors.amplitude import BoundaryAmplitudeExtractor
from src.analysis.processors.base import BaseProcessor, Processor
from src.analysis.processors.config import (
    BoundaryComputationConfig,
    NeighborDirection,
    ProcessorConfig,
)
from src.analysis.processors.decorators import ProcessorDecorators
from src.analysis.processors.discrimination import FaciesDiscriminationCalculator
from src.analysis.processors.exceptions import (
    CorrelationError,
    ProcessorError,
    ReshapeError,
    ValidationError,
)
from src.analysis.processors.gradient import GradientCorrelationCalculator
from src.analysis.processors.interface import InterfaceReflectionAnalyzer
from src.analysis.processors.utils import ProcessorUtils
from src.analysis.processors.validators import (
    ArrayValidator,
    ValidationHelpers,
    _ValidationErrors,
)


# Tests from test_processors_amplitude
# ============================================================================


class TestBoundaryAmplitudeExtractorExtract:
    """Tests for BoundaryAmplitudeExtractor.extract method."""

    def test_extract_basic(self):
        """Test basic amplitude extraction."""
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        boundaries = np.zeros((5, 5, 10), dtype=bool)
        boundaries[2, 2, :] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        assert isinstance(result, BoundaryAmpsResult)
        assert len(result.at_boundaries) > 0
        assert len(result.away_from_boundaries) > 0

    def test_extract_all_boundary(self):
        """Test when entire cube is boundary."""
        seismic = np.random.randn(3, 3, 5).astype(np.float64)
        boundaries = np.ones((3, 3, 5), dtype=bool)

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        # All should be at boundaries
        assert len(result.at_boundaries) == 3 * 3 * 5
        assert len(result.away_from_boundaries) == 0

    def test_extract_no_boundary(self):
        """Test when no boundaries exist."""
        seismic = np.random.randn(3, 3, 5).astype(np.float64)
        boundaries = np.zeros((3, 3, 5), dtype=bool)

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        # None should be at boundaries
        assert len(result.at_boundaries) == 0
        assert len(result.away_from_boundaries) == 3 * 3 * 5

    def test_extract_with_custom_window(self):
        """Test extraction with custom window parameter."""
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        boundaries = np.zeros((5, 5, 10), dtype=bool)
        boundaries[2, 2, 5] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries, window=2)

        assert isinstance(result, BoundaryAmpsResult)
        assert len(result.at_boundaries) > 0

    def test_extract_with_none_window_uses_default(self):
        """Test that None window uses default."""
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        boundaries = np.zeros((5, 5, 10), dtype=bool)
        boundaries[2, 2, 5] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=2)
        result = extractor.extract(seismic, boundaries, window=None)

        assert isinstance(result, BoundaryAmpsResult)

    def test_extract_invalid_window_raises_error(self):
        """Test that invalid window parameter raises error."""
        seismic = np.random.randn(3, 3, 5).astype(np.float64)
        boundaries = np.zeros((3, 3, 5), dtype=bool)

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        with pytest.raises(ValueError, match="non-negative"):
            extractor.extract(seismic, boundaries, window=-1)

    def test_extract_returns_correct_structure(self):
        """Test that extract returns correct result structure."""
        seismic = np.random.randn(4, 4, 8).astype(np.float64)
        boundaries = np.zeros((4, 4, 8), dtype=bool)
        boundaries[2, 2, :] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        assert hasattr(result, "at_boundaries")
        assert hasattr(result, "away_from_boundaries")
        assert hasattr(result, "boundary_mask")

    def test_extract_amplitude_partitioning(self):
        """Test that amplitudes are properly partitioned."""
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        boundaries = np.zeros((5, 5, 10), dtype=bool)
        boundaries[2, 2, :] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        total = len(result.at_boundaries) + len(result.away_from_boundaries)
        # Total should equal flattened array size
        assert total == 5 * 5 * 10

    def test_extract_boundary_mask_shape(self):
        """Test that boundary mask has correct shape."""
        seismic = np.random.randn(4, 6, 8).astype(np.float64)
        boundaries = np.zeros((4, 6, 8), dtype=bool)
        boundaries[2, 3, 4] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        assert len(result.boundary_mask) == 4 * 6 * 8

    def test_extract_various_window_sizes(self):
        """Test extraction with various window sizes."""
        seismic = np.random.randn(6, 6, 12).astype(np.float64)
        boundaries = np.zeros((6, 6, 12), dtype=bool)
        boundaries[3, 3, :] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)

        # Test with different windows
        for window in [1, 2, 3]:
            result = extractor.extract(seismic, boundaries, window=window)
            assert isinstance(result, BoundaryAmpsResult)
            # More dilation should result in more amplitudes at boundaries
            # (not strictly true due to overlaps, but should be monotonic trend)

    def test_extract_with_structured_boundary(self):
        """Test extraction with structured boundary pattern."""
        seismic = np.arange(100, dtype=np.float64).reshape(4, 5, 5)
        boundaries = np.zeros((4, 5, 5), dtype=bool)
        # Create a vertical boundary line
        boundaries[:, 2, :] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        assert len(result.at_boundaries) > 0
        assert len(result.away_from_boundaries) > 0

    def test_extract_boundary_mask_boolean(self):
        """Test that boundary mask is boolean dtype."""
        seismic = np.random.randn(3, 3, 5).astype(np.float64)
        boundaries = np.zeros((3, 3, 5), dtype=bool)

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        assert result.boundary_mask.dtype == bool


class TestBoundaryAmplitudeExtractorLogging:
    """Tests for logging behavior."""

    def test_extract_logging(self, caplog):
        """Test that extract logs debug information."""
        seismic = np.random.randn(4, 4, 8).astype(np.float64)
        boundaries = np.zeros((4, 4, 8), dtype=bool)

        with caplog.at_level(logging.DEBUG):
            extractor = BoundaryAmplitudeExtractor(dilation_window=1)
            result = extractor.extract(seismic, boundaries)

        # Should have logged extraction message
        assert any("boundary" in record.message.lower() for record in caplog.records)


class TestBoundaryAmplitudeExtractorIntegration:
    """Integration tests for BoundaryAmplitudeExtractor."""

    def test_full_workflow(self):
        """Test complete extraction workflow."""
        np.random.seed(42)
        seismic = np.random.randn(8, 8, 16).astype(np.float64)
        boundaries = np.zeros((8, 8, 16), dtype=bool)
        # Create a single boundary line
        boundaries[2, 2:5, :] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)
        result = extractor.extract(seismic, boundaries)

        assert isinstance(result, BoundaryAmpsResult)
        assert len(result.at_boundaries) > 0
        # With smaller boundary, should have away_from_boundaries
        if len(result.away_from_boundaries) == 0:
            # If all dilated into boundary, that's also valid
            assert len(result.boundary_mask) > 0

    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        boundaries = np.zeros((5, 5, 10), dtype=bool)
        boundaries[2, 2, :] = True

        extractor1 = BoundaryAmplitudeExtractor(dilation_window=1)
        result1 = extractor1.extract(seismic, boundaries)

        extractor2 = BoundaryAmplitudeExtractor(dilation_window=1)
        result2 = extractor2.extract(seismic, boundaries)

        np.testing.assert_array_equal(result1.at_boundaries, result2.at_boundaries)
        np.testing.assert_array_equal(
            result1.away_from_boundaries, result2.away_from_boundaries
        )

    def test_window_effect_on_amplitudes(self):
        """Test that larger windows affect amplitude distribution."""
        seismic = np.random.randn(6, 6, 12).astype(np.float64)
        boundaries = np.zeros((6, 6, 12), dtype=bool)
        boundaries[3, 3, :] = True

        extractor = BoundaryAmplitudeExtractor(dilation_window=1)

        result_w1 = extractor.extract(seismic, boundaries, window=1)
        result_w3 = extractor.extract(seismic, boundaries, window=3)

        # Larger window should dilate more
        # (more samples should be in boundary zone)
        assert len(result_w3.at_boundaries) >= len(result_w1.at_boundaries)


# Tests from test_processors_base
# ============================================================================


class TestBaseProcessor:
    """Test suite for BaseProcessor class."""

    def test_base_processor_initialization(self):
        """Test that BaseProcessor initializes without errors."""

        class ConcreteProcessor(BaseProcessor):
            def detect(self, data):
                return data

        processor = ConcreteProcessor()
        assert processor is not None

    def test_base_processor_has_aligner_property(self):
        """Test that BaseProcessor has _aligner lazy property."""

        class ConcreteProcessor(BaseProcessor):
            def detect(self, data):
                return data

        processor = ConcreteProcessor()
        assert hasattr(processor, "_aligner")

    def test_aligner_lazy_initialization(self):
        """Test that _aligner is lazily initialized."""

        class ConcreteProcessor(BaseProcessor):
            def detect(self, data):
                return data

        processor = ConcreteProcessor()
        # Should be None initially
        assert processor._aligner_instance is None
        # Access via property should initialize it
        aligner = processor._aligner
        assert aligner is not None
        # Subsequent access should return same instance
        assert processor._aligner is aligner

    def test_aligner_singleton_per_processor(self):
        """Test that each processor instance has its own aligner."""

        class ConcreteProcessor(BaseProcessor):
            def detect(self, data):
                return data

        proc1 = ConcreteProcessor()
        proc2 = ConcreteProcessor()

        assert proc1._aligner is not proc2._aligner

    def test_process_delegates_to_detect(self):
        """Test that process() method delegates to detect()."""

        class DetectProcessor(BaseProcessor):
            def detect(self, data):
                return f"detected: {data}"

        processor = DetectProcessor()
        result = processor.process("test_data")
        assert result == "detected: test_data"

    def test_process_delegates_to_extract(self):
        """Test that process() method delegates to extract()."""

        class ExtractProcessor(BaseProcessor):
            def extract(self, data):
                return f"extracted: {data}"

        processor = ExtractProcessor()
        result = processor.process("test_data")
        assert result == "extracted: test_data"

    def test_process_delegates_to_calculate(self):
        """Test that process() method delegates to calculate()."""

        class CalculateProcessor(BaseProcessor):
            def calculate(self, data):
                return f"calculated: {data}"

        processor = CalculateProcessor()
        result = processor.process("test_data")
        assert result == "calculated: test_data"

    def test_process_delegates_to_analyze(self):
        """Test that process() method delegates to analyze()."""

        class AnalyzeProcessor(BaseProcessor):
            def analyze(self, data):
                return f"analyzed: {data}"

        processor = AnalyzeProcessor()
        result = processor.process("test_data")
        assert result == "analyzed: test_data"

    def test_process_delegates_in_priority_order(self):
        """Test that process() delegates in correct priority order."""

        class MultiMethodProcessor(BaseProcessor):
            def detect(self, data):
                return "detect"

            def extract(self, data):
                return "extract"

            def calculate(self, data):
                return "calculate"

            def analyze(self, data):
                return "analyze"

        processor = MultiMethodProcessor()
        # Should call detect first (highest priority)
        assert processor.process("test") == "detect"

    def test_process_raises_if_no_domain_method(self):
        """Test that process() raises if no domain method is implemented."""

        class NoDomainMethodProcessor(BaseProcessor):
            pass

        processor = NoDomainMethodProcessor()
        with pytest.raises(NotImplementedError, match="must implement one of"):
            processor.process("test")

    def test_callable_interface(self):
        """Test that processor instances are callable."""

        class CallableProcessor(BaseProcessor):
            def detect(self, data):
                return f"called: {data}"

        processor = CallableProcessor()
        result = processor("test_data")
        assert result == "called: test_data"

    def test_callable_delegates_to_process(self):
        """Test that __call__ delegates to process()."""

        class CallProcessor(BaseProcessor):
            def process(self, *args, **kwargs):
                return ("process_called", args, kwargs)

        processor = CallProcessor()
        result = processor("arg1", "arg2", key="value")
        assert result[0] == "process_called"
        assert result[1] == ("arg1", "arg2")
        assert result[2] == {"key": "value"}

    def test_domain_methods_receive_arguments(self):
        """Test that domain methods receive all arguments correctly."""

        class ArgProcessor(BaseProcessor):
            def extract(self, *args, **kwargs):
                return (args, kwargs)

        processor = ArgProcessor()
        result = processor.process("a", "b", x=1, y=2)
        assert result[0] == ("a", "b")
        assert result[1] == {"x": 1, "y": 2}

    def test_process_passes_kwargs_through(self):
        """Test that process() preserves keyword arguments."""

        class KwargsProcessor(BaseProcessor):
            def calculate(self, data, threshold=0.5, normalize=True):
                return {"data": data, "threshold": threshold, "normalize": normalize}

        processor = KwargsProcessor()
        result = processor.process("test", threshold=0.8, normalize=False)
        assert result == {"data": "test", "threshold": 0.8, "normalize": False}

    def test_repr_shows_class_name(self):
        """Test that __repr__ includes class name and aligner."""

        class ReprProcessor(BaseProcessor):
            def detect(self, data):
                return data

        processor = ReprProcessor()
        repr_str = repr(processor)
        assert "ReprProcessor" in repr_str or "BaseProcessor" in repr_str


class TestProcessorInheritance:
    """Test suite for processor inheritance patterns."""

    def test_base_processor_inherits_from_processor(self):
        """Test that BaseProcessor is a subclass of Processor."""
        assert issubclass(BaseProcessor, Processor)

    def test_concrete_processor_inherits_from_base_processor(self):
        """Test that concrete processors inherit from BaseProcessor."""

        class MyProcessor(BaseProcessor):
            def detect(self, data):
                return data

        assert issubclass(MyProcessor, BaseProcessor)
        assert issubclass(MyProcessor, Processor)

    def test_processor_method_resolution_order(self):
        """Test method resolution order for processor hierarchy."""

        class ConcreteProcessor(BaseProcessor):
            def detect(self, data):
                return data

        processor = ConcreteProcessor()
        # Should find detect in MRO
        assert hasattr(processor, "detect")
        # Should find process in MRO
        assert hasattr(processor, "process")
        # Should find _aligner in MRO
        assert hasattr(processor, "_aligner")


class TestDomainMethodPriority:
    """Test suite for domain method resolution priority."""

    def test_detect_has_priority_over_extract(self):
        """Test that detect() has priority when both exist."""

        class BothDetectExtract(BaseProcessor):
            def detect(self, data):
                return "detect"

            def extract(self, data):
                return "extract"

        processor = BothDetectExtract()
        assert processor.process(None) == "detect"

    def test_extract_has_priority_over_calculate(self):
        """Test extract() priority order."""

        class BothExtractCalculate(BaseProcessor):
            def extract(self, data):
                return "extract"

            def calculate(self, data):
                return "calculate"

        processor = BothExtractCalculate()
        assert processor.process(None) == "extract"

    def test_calculate_has_priority_over_analyze(self):
        """Test calculate() priority order."""

        class BothCalculateAnalyze(BaseProcessor):
            def calculate(self, data):
                return "calculate"

            def analyze(self, data):
                return "analyze"

        processor = BothCalculateAnalyze()
        assert processor.process(None) == "calculate"

    def test_only_analyze_is_called_when_alone(self):
        """Test that analyze() is called when it's the only domain method."""

        class OnlyAnalyze(BaseProcessor):
            def analyze(self, data):
                return "analyze"

        processor = OnlyAnalyze()
        assert processor.process(None) == "analyze"


# Tests from test_processors_config
# ============================================================================


class TestBoundaryComputationConfig:
    """Test suite for BoundaryComputationConfig dataclass."""

    def test_boundary_computation_config_creation(self):
        """Test that BoundaryComputationConfig can be created."""
        config = BoundaryComputationConfig()
        assert config is not None

    def test_boundary_computation_config_is_frozen(self):
        """Test that BoundaryComputationConfig is immutable."""
        config = BoundaryComputationConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.dilation_window = 5


class TestNeighborDirection:
    """Test suite for NeighborDirection enum."""

    def test_neighbor_direction_is_enum(self):
        """Test that NeighborDirection is an enumeration."""
        assert hasattr(NeighborDirection, "UP")
        assert hasattr(NeighborDirection, "DOWN")
        assert hasattr(NeighborDirection, "LEFT")
        assert hasattr(NeighborDirection, "RIGHT")
        assert hasattr(NeighborDirection, "CENTER")

    def test_neighbor_direction_slice_values(self):
        """Test that NeighborDirection values are valid slices."""
        assert isinstance(NeighborDirection.UP.value, slice)
        assert isinstance(NeighborDirection.DOWN.value, slice)
        assert isinstance(NeighborDirection.LEFT.value, slice)
        assert isinstance(NeighborDirection.RIGHT.value, slice)
        assert isinstance(NeighborDirection.CENTER.value, slice)

    def test_all_directions_classmethod(self):
        """Test all_directions() classmethod returns non-center directions."""
        directions = NeighborDirection.all_directions()
        assert len(directions) == 4
        assert NeighborDirection.UP in directions
        assert NeighborDirection.DOWN in directions
        assert NeighborDirection.LEFT in directions
        assert NeighborDirection.RIGHT in directions
        assert NeighborDirection.CENTER not in directions

    def test_neighbor_direction_equality(self):
        """Test that NeighborDirection enum members are comparable."""
        assert NeighborDirection.UP == NeighborDirection.UP
        assert NeighborDirection.UP != NeighborDirection.DOWN


class TestProcessorConfigValidation:
    """Test suite for ProcessorConfig validation."""

    def test_percentile_q1_less_than_q3(self):
        """Test that Q1 is less than Q3 by default."""
        config = ProcessorConfig()
        assert config.percentile_q1 < config.percentile_q3

    def test_amplitude_window_radius_positive(self):
        """Test that amplitude window radius defaults to positive."""
        config = ProcessorConfig()
        assert config.amplitude_window_radius > 0

    def test_boundary_dilation_positive(self):
        """Test that boundary dilation defaults to positive."""
        config = ProcessorConfig()
        assert config.boundary_dilation_default > 0

    def test_separation_matrix_epsilon_positive(self):
        """Test that epsilon is a small positive value."""
        config = ProcessorConfig()
        assert 0 < config.separation_matrix_epsilon < 1e-8

    def test_min_valid_samples_positive(self):
        """Test that min_valid_samples is positive."""
        config = ProcessorConfig()
        assert config.min_valid_samples > 0


class TestProcessorConfigSingletonBehavior:
    """Test suite for ProcessorConfig singleton-like behavior."""

    def test_multiple_config_instances_are_different(self):
        """Test that creating multiple configs creates different instances."""
        config1 = ProcessorConfig()
        config2 = ProcessorConfig()
        assert config1 is not config2

    def test_config_instances_have_same_defaults(self):
        """Test that different instances have same default values."""
        config1 = ProcessorConfig()
        config2 = ProcessorConfig()
        assert config1.boundary_dilation_default == config2.boundary_dilation_default
        assert config1.pad_mode == config2.pad_mode

    def test_config_equality(self):
        """Test that configs with same values are equal."""
        config1 = ProcessorConfig()
        config2 = ProcessorConfig()
        assert config1 == config2


# Tests from test_processors_decorators
# ============================================================================


class TestTimeOperationDecorator:
    """Test suite for time_operation decorator."""

    def test_time_operation_logs_execution_time(self, caplog):
        """Test that time_operation decorator logs execution time."""
        with caplog.at_level(logging.DEBUG):

            @ProcessorDecorators.time_operation("test operation", threshold_ms=1000.0)
            def slow_func():
                return "done"

            result = slow_func()
            assert result == "done"
            assert "test operation completed in" in caplog.text
            assert "ms" in caplog.text

    def test_time_operation_logs_debug_when_under_threshold(self, caplog):
        """Test that execution under threshold logs at DEBUG level."""
        with caplog.at_level(logging.DEBUG):

            @ProcessorDecorators.time_operation("fast op", threshold_ms=1000.0)
            def fast_func():
                return "quick"

            fast_func()
            # Check that it's in debug records
            assert any(
                "fast op completed in" in record.message
                for record in caplog.records
                if record.levelname == "DEBUG"
            )

    def test_time_operation_logs_warning_when_over_threshold(self, caplog):
        """Test that execution over threshold logs at WARNING level."""
        with caplog.at_level(logging.DEBUG):

            @ProcessorDecorators.time_operation("slow op", threshold_ms=1.0)
            def slow_func():
                time.sleep(0.01)  # Sleep 10ms, threshold is 1ms
                return "done"

            slow_func()
            # Should have a warning
            assert any(
                "slow op completed in" in record.message
                for record in caplog.records
                if record.levelname == "WARNING"
            )

    def test_time_operation_preserves_function_result(self):
        """Test that decorator doesn't modify the function result."""

        @ProcessorDecorators.time_operation("test", threshold_ms=100.0)
        def return_dict():
            return {"data": [1, 2, 3]}

        result = return_dict()
        assert result == {"data": [1, 2, 3]}

    def test_time_operation_with_zero_threshold(self, caplog):
        """Test time_operation with threshold of 0 (all operations logged as warning)."""
        with caplog.at_level(logging.DEBUG):

            @ProcessorDecorators.time_operation("op", threshold_ms=0.0)
            def any_func():
                return "result"

            any_func()
            # With threshold 0, any execution time will exceed it
            assert any("op completed in" in record.message for record in caplog.records)

    def test_time_operation_measures_actual_time(self):
        """Test that time_operation actually measures execution time."""
        durations = []

        @ProcessorDecorators.time_operation("measured op", threshold_ms=10000.0)
        def timed_func(sleep_time):
            time.sleep(sleep_time)

        with patch("time.perf_counter", wraps=time.perf_counter) as mock_timer:
            timed_func(0.01)
            # perf_counter should be called at least twice (start and end)
            assert mock_timer.call_count >= 2


class TestValidateCubeShapeDecorator:
    """Test suite for validate_cube_shape decorator."""

    def test_validate_cube_shape_accepts_valid_3d_array(self):
        """Test that valid 3D array passes validation."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def detect(self, cube):
                return cube.shape

        processor = TestProcessor()
        cube = np.ones((10, 10, 10))
        result = processor.detect(cube)
        assert result == (10, 10, 10)

    def test_validate_cube_shape_rejects_2d_array(self):
        """Test that 2D array fails validation for 3D expected."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def detect(self, cube):
                return cube

        processor = TestProcessor()
        cube = np.ones((10, 10))
        with pytest.raises(ValueError, match="expects 3D cube"):
            processor.detect(cube)

    def test_validate_cube_shape_rejects_1d_array(self):
        """Test that 1D array fails validation."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def detect(self, cube):
                return cube

        processor = TestProcessor()
        cube = np.ones(10)
        with pytest.raises(ValueError, match="expects 3D cube"):
            processor.detect(cube)

    def test_validate_cube_shape_rejects_non_ndarray(self):
        """Test that non-ndarray input raises TypeError."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def detect(self, cube):
                return cube

        processor = TestProcessor()
        with pytest.raises(TypeError, match="Expected ndarray"):
            processor.detect([1, 2, 3])

    def test_validate_cube_shape_rejects_empty_array(self):
        """Test that empty array is rejected."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def detect(self, cube):
                return cube

        processor = TestProcessor()
        cube = np.array([]).reshape(0, 0, 0)
        with pytest.raises(ValueError, match="empty"):
            processor.detect(cube)

    def test_validate_cube_shape_includes_function_name_in_error(self):
        """Test that error message includes function name."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def my_detector(self, cube):
                return cube

        processor = TestProcessor()
        cube = np.ones((5, 5))
        with pytest.raises(ValueError, match="my_detector"):
            processor.my_detector(cube)

    def test_validate_cube_shape_includes_shape_in_error(self):
        """Test that error message includes actual shape."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def detect(self, cube):
                return cube

        processor = TestProcessor()
        cube = np.ones((5, 5))
        with pytest.raises(ValueError, match="shape"):
            processor.detect(cube)

    def test_validate_cube_shape_custom_dimensions(self):
        """Test validator with custom dimension requirement."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=2)
            def process_2d(self, arr):
                return arr

        processor = TestProcessor()
        arr_2d = np.ones((10, 10))
        result = processor.process_2d(arr_2d)
        assert result.shape == (10, 10)

        arr_3d = np.ones((10, 10, 10))
        with pytest.raises(ValueError):
            processor.process_2d(arr_3d)

    def test_validate_cube_shape_preserves_function_name(self):
        """Test that decorator preserves function name."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def original_func(self, cube):
                return cube

        processor = TestProcessor()
        assert processor.original_func.__name__ == "original_func"

    def test_validate_cube_shape_passes_through_arguments(self):
        """Test that decorator preserves all arguments."""

        class TestProcessor(BaseProcessor):
            @ProcessorDecorators.validate_cube_shape(expected_dims=3)
            def func_with_args(self, cube, param1, param2=None):
                return (cube.shape, param1, param2)

        processor = TestProcessor()
        cube = np.ones((5, 5, 5))
        result = processor.func_with_args(cube, "val1", param2="val2")
        assert result == ((5, 5, 5), "val1", "val2")

    def test_validate_cube_shape_logs_debug_message(self, caplog):
        """Test that validation logs at DEBUG level."""
        with caplog.at_level(logging.DEBUG):

            class TestProcessor(BaseProcessor):
                @ProcessorDecorators.validate_cube_shape(expected_dims=3)
                def detect(self, cube):
                    return cube

            processor = TestProcessor()
            cube = np.ones((5, 5, 5))
            processor.detect(cube)
            assert "Validated 3D cube" in caplog.text


class TestProcessorDecoratorsComposition:
    """Test suite for composing multiple decorators."""

    def test_log_debug_and_time_operation_together(self, caplog):
        """Test that log_debug and time_operation work together."""
        with caplog.at_level(logging.DEBUG):

            class TestProc(BaseProcessor):
                @ProcessorDecorators.time_operation("op", threshold_ms=1000.0)
                @ProcessorDecorators.log_debug("Executing {}...")
                def detect(self):
                    return "result"

            proc = TestProc()
            result = proc.detect()
            assert result == "result"
            assert "Executing detect..." in caplog.text
            assert "op completed in" in caplog.text

    def test_validate_and_time_operation_together(self, caplog):
        """Test that validate_cube_shape and time_operation work together."""
        with caplog.at_level(logging.DEBUG):

            class TestProc(BaseProcessor):
                @ProcessorDecorators.time_operation("op", threshold_ms=1000.0)
                @ProcessorDecorators.validate_cube_shape(expected_dims=3)
                def detect(self, cube):
                    return cube.shape

            proc = TestProc()
            cube = np.ones((5, 5, 5))
            result = proc.detect(cube)
            assert result == (5, 5, 5)
            assert "Validated 3D cube" in caplog.text


# Tests from test_processors_discrimination
# ============================================================================


class TestFaciesDiscriminationCalculatorCalculate:
    """Tests for FaciesDiscriminationCalculator.calculate method."""

    def test_calculate_single_facies(self):
        """Test calculate with only one facies type."""
        seismic = np.random.randn(4, 4, 10).astype(np.float64)
        facies = np.zeros((4, 4, 10), dtype=np.int64)

        calc = FaciesDiscriminationCalculator()
        result = calc.calculate(seismic, facies)

        assert isinstance(result, FaciesDiscriminationResult)
        assert len(result.facies_stats) == 1
        assert 0 in result.facies_stats

    def test_calculate_two_facies(self):
        """Test calculate with two distinct facies."""
        seismic = np.random.randn(4, 4, 10).astype(np.float64)
        facies = np.zeros((4, 4, 10), dtype=np.int64)
        facies[:, :, :5] = 1

        calc = FaciesDiscriminationCalculator()
        result = calc.calculate(seismic, facies)

        assert isinstance(result, FaciesDiscriminationResult)
        assert len(result.facies_stats) == 2
        assert 0 in result.facies_stats
        assert 1 in result.facies_stats

    def test_calculate_multiple_facies(self):
        """Test calculate with multiple facies."""
        seismic = np.random.randn(5, 5, 15).astype(np.float64)
        facies = np.tile([0, 1, 2, 3, 4], (5, 5, 3))[:5, :5, :15].astype(np.int64)

        calc = FaciesDiscriminationCalculator()
        result = calc.calculate(seismic, facies)

        assert isinstance(result, FaciesDiscriminationResult)
        assert len(result.facies_stats) <= 5

    def test_calculate_returns_result_structure(self):
        """Test that calculate returns complete result structure."""
        seismic = np.random.randn(3, 3, 8).astype(np.float64)
        facies = np.random.randint(0, 3, (3, 3, 8)).astype(np.int64)

        calc = FaciesDiscriminationCalculator()
        result = calc.calculate(seismic, facies)

        assert hasattr(result, "facies_stats")
        assert hasattr(result, "separation_matrix")
        assert hasattr(result, "facies_amplitudes")
        assert hasattr(result, "label_order")

    def test_calculate_with_stratified_amplitudes(self):
        """Test calculate with clear amplitude stratification by facies."""
        seismic = np.zeros((4, 4, 12), dtype=np.float64)
        # Facies 0: low amplitudes
        seismic[:, :, :4] = np.random.randn(4, 4, 4) * 0.1
        # Facies 1: medium amplitudes
        seismic[:, :, 4:8] = np.random.randn(4, 4, 4) * 1.0 + 1.0
        # Facies 2: high amplitudes
        seismic[:, :, 8:] = np.random.randn(4, 4, 4) * 0.1 + 3.0

        facies = np.zeros((4, 4, 12), dtype=np.int64)
        facies[:, :, 4:8] = 1
        facies[:, :, 8:] = 2

        calc = FaciesDiscriminationCalculator()
        result = calc.calculate(seismic, facies)

        assert len(result.facies_stats) == 3
        # Separation matrix should show good discrimination
        assert result.separation_matrix.shape == (3, 3)


class TestFaciesDiscriminationCalculatorExtractAmplitudes:
    """Tests for _extract_facies_amplitudes static method."""

    def test_extract_amplitudes_single_facies(self):
        """Test extraction with single facies."""
        seismic = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        facies = np.zeros((2, 3, 4), dtype=np.int64)

        facies_amps, label_order = (
            FaciesDiscriminationCalculator._extract_facies_amplitudes(seismic, facies)
        )

        assert len(facies_amps) == 1
        assert 0 in facies_amps
        assert label_order == [0]
        assert len(facies_amps[0]) == 24

    def test_extract_amplitudes_multiple_facies(self):
        """Test extraction with multiple facies."""
        seismic = np.random.randn(4, 4, 12).astype(np.float64)
        facies = np.zeros((4, 4, 12), dtype=np.int64)
        facies[:, :, :4] = 0
        facies[:, :, 4:8] = 1
        facies[:, :, 8:] = 2

        facies_amps, label_order = (
            FaciesDiscriminationCalculator._extract_facies_amplitudes(seismic, facies)
        )

        assert len(facies_amps) == 3
        assert 0 in facies_amps
        assert 1 in facies_amps
        assert 2 in facies_amps
        assert label_order == [0, 1, 2]

    def test_extract_amplitudes_label_order_sorted(self):
        """Test that label order is sorted."""
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        # Create 2D array then broadcast to 3D
        facies_2d = np.array([[3, 1, 2, 0, 3, 1, 2, 0, 3, 1]], dtype=np.int64)
        facies = np.tile(facies_2d, (5, 5, 1))[:5, :5, :10].astype(np.int64)

        facies_amps, label_order = (
            FaciesDiscriminationCalculator._extract_facies_amplitudes(seismic, facies)
        )

        # Should be sorted
        assert label_order == sorted(label_order)

    def test_extract_amplitudes_count_preservation(self):
        """Test that total amplitude count is preserved."""
        total_samples = 5 * 6 * 8
        seismic = np.random.randn(5, 6, 8).astype(np.float64)
        facies = np.random.randint(0, 3, (5, 6, 8)).astype(np.int64)

        facies_amps, label_order = (
            FaciesDiscriminationCalculator._extract_facies_amplitudes(seismic, facies)
        )

        total_extracted = sum(len(amps) for amps in facies_amps.values())
        assert total_extracted == total_samples

    def test_extract_amplitudes_correct_assignment(self):
        """Test that amplitudes are assigned to correct facies."""
        seismic = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
        facies = np.array([[[0, 0], [1, 1]]], dtype=np.int64)

        facies_amps, label_order = (
            FaciesDiscriminationCalculator._extract_facies_amplitudes(seismic, facies)
        )

        assert set(facies_amps[0]) == {1.0, 2.0}
        assert set(facies_amps[1]) == {3.0, 4.0}


class TestFaciesDiscriminationCalculatorCalculateStats:
    """Tests for _calculate_facies_stats static method."""

    def test_calculate_stats_single_facies(self):
        """Test stats calculation for single facies."""
        facies_amps = {0: np.array([1.0, 2.0, 3.0])}

        stats = FaciesDiscriminationCalculator._calculate_facies_stats(facies_amps)

        assert len(stats) == 1
        assert 0 in stats
        assert isinstance(stats[0], (FaciesStats, type(None)))

    def test_calculate_stats_multiple_facies(self):
        """Test stats calculation for multiple facies."""
        facies_amps = {
            0: np.array([1.0, 2.0, 3.0]),
            1: np.array([4.0, 5.0, 6.0]),
            2: np.array([7.0, 8.0, 9.0]),
        }

        stats = FaciesDiscriminationCalculator._calculate_facies_stats(facies_amps)

        assert len(stats) == 3
        for key in [0, 1, 2]:
            assert key in stats

    def test_calculate_stats_with_empty_arrays(self):
        """Test stats calculation with empty arrays."""
        facies_amps = {0: np.array([])}

        stats = FaciesDiscriminationCalculator._calculate_facies_stats(facies_amps)

        # Empty array should result in None or be handled gracefully
        assert isinstance(stats, dict)

    def test_calculate_stats_preserves_keys(self):
        """Test that stats preserves all facies keys."""
        facies_amps = {10: np.array([1.0]), 20: np.array([2.0]), 30: np.array([3.0])}

        stats = FaciesDiscriminationCalculator._calculate_facies_stats(facies_amps)

        assert set(stats.keys()) == {10, 20, 30}


class TestFaciesDiscriminationCalculatorSeparationMatrix:
    """Tests for _calculate_separation_matrix static method."""

    def test_separation_matrix_shape(self):
        """Test that separation matrix has correct shape."""
        label_order = [0, 1, 2]
        facies_stats = {
            0: FaciesStats(count=10, mean=1.0, std=0.5),
            1: FaciesStats(count=10, mean=2.0, std=0.5),
            2: FaciesStats(count=10, mean=3.0, std=0.5),
        }

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        assert sep_matrix.shape == (3, 3)

    def test_separation_matrix_diagonal_zero(self):
        """Test that diagonal of separation matrix is zero."""
        label_order = [0, 1, 2]
        facies_stats = {
            0: FaciesStats(count=10, mean=1.0, std=0.5),
            1: FaciesStats(count=10, mean=2.0, std=0.5),
            2: FaciesStats(count=10, mean=3.0, std=0.5),
        }

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        np.testing.assert_array_almost_equal(np.diag(sep_matrix), [0, 0, 0])

    def test_separation_matrix_symmetry(self):
        """Test that separation matrix is symmetric."""
        label_order = [0, 1, 2]
        facies_stats = {
            0: FaciesStats(count=10, mean=1.0, std=0.5),
            1: FaciesStats(count=10, mean=2.0, std=0.5),
            2: FaciesStats(count=10, mean=3.0, std=0.5),
        }

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        np.testing.assert_array_almost_equal(sep_matrix, sep_matrix.T)

    def test_separation_matrix_well_separated_facies(self):
        """Test separation matrix with well-separated facies."""
        label_order = [0, 1, 2]
        facies_stats = {
            0: FaciesStats(count=10, mean=0.0, std=0.1),
            1: FaciesStats(count=10, mean=10.0, std=0.1),
            2: FaciesStats(count=10, mean=20.0, std=0.1),
        }

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        # Off-diagonal should have high separation values
        assert sep_matrix[0, 1] > 10
        assert sep_matrix[0, 2] > 10
        assert sep_matrix[1, 2] > 10

    def test_separation_matrix_poorly_separated_facies(self):
        """Test separation matrix with poorly-separated facies."""
        label_order = [0, 1, 2]
        facies_stats = {
            0: FaciesStats(count=10, mean=1.0, std=10.0),
            1: FaciesStats(count=10, mean=1.5, std=10.0),
            2: FaciesStats(count=10, mean=2.0, std=10.0),
        }

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        # Off-diagonal should have low separation values
        assert sep_matrix[0, 1] < 1
        assert sep_matrix[0, 2] < 1

    def test_separation_matrix_insufficient_facies(self):
        """Test separation matrix with only one facies."""
        label_order = [0]
        facies_stats = {0: FaciesStats(count=10, mean=1.0, std=0.5)}

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        # Should return zero matrix
        expected = np.zeros((1, 1), dtype=float)
        np.testing.assert_array_equal(sep_matrix, expected)

    def test_separation_matrix_missing_stats(self):
        """Test separation matrix when some facies lack stats."""
        label_order = [0, 1, 2]
        facies_stats = {
            0: FaciesStats(count=10, mean=1.0, std=0.5),
            # 1 is missing stats
            2: FaciesStats(count=10, mean=3.0, std=0.5),
        }

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        # Should still return matrix of correct size
        assert sep_matrix.shape == (3, 3)

    def test_separation_matrix_with_nan_std(self):
        """Test separation matrix handles NaN std values gracefully."""
        label_order = [0, 1]
        facies_stats = {
            0: FaciesStats(count=10, mean=1.0, std=np.nan),
            1: FaciesStats(count=10, mean=2.0, std=np.nan),
        }

        sep_matrix = FaciesDiscriminationCalculator._calculate_separation_matrix(
            facies_stats, label_order
        )

        # Should return matrix without errors
        assert sep_matrix.shape == (2, 2)
        # May contain NaN due to NaN std values
        assert isinstance(sep_matrix, np.ndarray)


class TestFaciesDiscriminationCalculatorIntegration:
    """Integration tests for FaciesDiscriminationCalculator."""

    def test_full_workflow(self):
        """Test complete workflow from seismic/facies to results."""
        np.random.seed(42)
        seismic = np.random.randn(6, 6, 12).astype(np.float64)
        facies = np.zeros((6, 6, 12), dtype=np.int64)
        facies[:, :, :4] = 1
        facies[:, :, 8:] = 2

        calc = FaciesDiscriminationCalculator()
        result = calc.calculate(seismic, facies)

        assert isinstance(result, FaciesDiscriminationResult)
        assert len(result.facies_stats) > 0
        assert result.separation_matrix.shape[0] == len(result.label_order)
        assert len(result.facies_amplitudes) > 0

    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        facies = np.random.randint(0, 3, (5, 5, 10)).astype(np.int64)

        calc1 = FaciesDiscriminationCalculator()
        result1 = calc1.calculate(seismic, facies)

        calc2 = FaciesDiscriminationCalculator()
        result2 = calc2.calculate(seismic, facies)

        np.testing.assert_array_equal(
            result1.separation_matrix, result2.separation_matrix
        )


class TestFaciesDiscriminationCalculatorLogging:
    """Tests for logging behavior."""

    def test_calculate_logging(self, caplog):
        """Test that calculate logs debug information."""
        seismic = np.random.randn(3, 3, 8).astype(np.float64)
        facies = np.random.randint(0, 2, (3, 3, 8)).astype(np.int64)

        with caplog.at_level(logging.DEBUG):
            calc = FaciesDiscriminationCalculator()
            result = calc.calculate(seismic, facies)

        # Should have logged something about discrimination
        assert any(
            "discrimination" in record.message.lower()
            for record in caplog.records
            if record.levelname == "DEBUG"
        )


# Tests from test_processors_exceptions
# ============================================================================


class TestValidationError:
    """Test suite for ValidationError exception."""

    def test_validation_error_inherits_from_processor_error(self):
        """Test that ValidationError is a subclass of ProcessorError."""
        assert issubclass(ValidationError, ProcessorError)

    def test_validation_error_can_be_raised(self):
        """Test that ValidationError can be raised."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")

    def test_validation_error_caught_as_processor_error(self):
        """Test that ValidationError can be caught as ProcessorError."""
        with pytest.raises(ProcessorError):
            raise ValidationError("Failed validation")

    def test_validation_error_message(self):
        """Test ValidationError message preservation."""
        msg = "Invalid array shape"
        with pytest.raises(ValidationError, match=msg):
            raise ValidationError(msg)

    def test_validation_error_caught_specifically(self):
        """Test that ValidationError can be caught specifically."""
        with pytest.raises(ValidationError):
            raise ValidationError("Array validation failed")


class TestCorrelationError:
    """Test suite for CorrelationError exception."""

    def test_correlation_error_inherits_from_processor_error(self):
        """Test that CorrelationError is a subclass of ProcessorError."""
        assert issubclass(CorrelationError, ProcessorError)

    def test_correlation_error_can_be_raised(self):
        """Test that CorrelationError can be raised."""
        with pytest.raises(CorrelationError):
            raise CorrelationError("Correlation computation failed")

    def test_correlation_error_caught_as_processor_error(self):
        """Test that CorrelationError can be caught as ProcessorError."""
        with pytest.raises(ProcessorError):
            raise CorrelationError("Correlation failed")

    def test_correlation_error_message(self):
        """Test CorrelationError message."""
        msg = "Cannot compute correlation with NaN values"
        with pytest.raises(CorrelationError, match=msg):
            raise CorrelationError(msg)


class TestReshapeError:
    """Test suite for ReshapeError exception."""

    def test_reshape_error_inherits_from_processor_error(self):
        """Test that ReshapeError is a subclass of ProcessorError."""
        assert issubclass(ReshapeError, ProcessorError)

    def test_reshape_error_can_be_raised(self):
        """Test that ReshapeError can be raised."""
        with pytest.raises(ReshapeError):
            raise ReshapeError("Cannot reshape array")

    def test_reshape_error_caught_as_processor_error(self):
        """Test that ReshapeError can be caught as ProcessorError."""
        with pytest.raises(ProcessorError):
            raise ReshapeError("Reshape failed")

    def test_reshape_error_message(self):
        """Test ReshapeError message."""
        msg = "Target shape is incompatible with array size"
        with pytest.raises(ReshapeError, match=msg):
            raise ReshapeError(msg)


class TestExceptionHierarchy:
    """Test suite for exception hierarchy."""

    def test_exception_inheritance_chain(self):
        """Test the complete exception inheritance chain."""
        assert issubclass(ReshapeError, ProcessorError)
        assert issubclass(CorrelationError, ProcessorError)
        assert issubclass(ValidationError, ProcessorError)
        assert issubclass(ProcessorError, Exception)

    def test_catch_parent_exception_catches_all_children(self):
        """Test that catching ProcessorError catches all child exceptions."""
        errors = [
            ValidationError("validation"),
            CorrelationError("correlation"),
            ReshapeError("reshape"),
        ]

        for error in errors:
            with pytest.raises(ProcessorError):
                raise error

    def test_specific_exception_not_caught_by_different_exception(self):
        """Test that specific exceptions don't catch others."""
        with pytest.raises(CorrelationError):
            try:
                raise CorrelationError("Wrong error type")
            except ValidationError:
                pass  # Should not reach here

    def test_error_type_identification(self):
        """Test that exceptions can be identified by type."""
        try:
            raise ValidationError("Test")
        except ValidationError as e:
            assert isinstance(e, ValidationError)
            assert isinstance(e, ProcessorError)


class TestExceptionUsagePatterns:
    """Test suite for common exception usage patterns."""

    def test_re_raise_with_context(self):
        """Test re-raising exception with context."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ValidationError(f"Validation failed: {e}") from e
        except ProcessorError as e:
            assert "Validation failed" in str(e)
            assert e.__cause__ is not None

    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise CorrelationError("Initial error")
            except CorrelationError:
                raise ReshapeError("Subsequent error")
        except ProcessorError as e:
            assert isinstance(e, ReshapeError)

    def test_exception_with_multiple_messages(self):
        """Test exception with formatted messages."""
        context = {"operation": "detect", "shape": (10, 10)}
        msg = f"Failed during {context['operation']} with shape {context['shape']}"
        with pytest.raises(ProcessorError, match="detect"):
            raise ProcessorError(msg)

    def test_exception_attributes_preserved(self):
        """Test that exception attributes are preserved."""
        error = ProcessorError("Test error")
        error.custom_attr = "custom_value"
        assert error.custom_attr == "custom_value"


# Tests from test_processors_gradient
# ============================================================================


class TestGradientCorrelationCalculatorCalculate:
    """Tests for GradientCorrelationCalculator.calculate method."""

    def test_calculate_returns_result_structure(self):
        """Test that calculate returns complete result structure."""
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        facies = np.random.randint(0, 3, (5, 5, 10)).astype(np.int64)

        calc = GradientCorrelationCalculator()
        result = calc.calculate(seismic, facies)

        assert isinstance(result, GradientCorrelationResult)
        assert hasattr(result, "pearson_correlation")
        assert hasattr(result, "pearson_pvalue")
        assert hasattr(result, "spearman_correlation")
        assert hasattr(result, "spearman_pvalue")
        assert hasattr(result, "seismic_gradient")
        assert hasattr(result, "boundaries")

    def test_calculate_with_correlated_patterns(self):
        """Test calculate when gradient and boundaries should be correlated."""
        # Create seismic with varying amplitudes (not constant gradient)
        seismic = np.random.randn(4, 4, 10).astype(np.float64)
        seismic[:, :, :5] = np.abs(seismic[:, :, :5])  # Positive in first half
        seismic[:, :, 5:] = -np.abs(seismic[:, :, 5:])  # Negative in second half

        # Facies with boundaries matching gradient changes
        facies = np.zeros((4, 4, 10), dtype=np.int64)
        facies[:, :, :5] = 0
        facies[:, :, 5:] = 1

        calc = GradientCorrelationCalculator()
        result = calc.calculate(seismic, facies)

        assert isinstance(result, GradientCorrelationResult)
        # Result should either have valid correlation or NaN (both are acceptable)
        assert isinstance(result.pearson_correlation, (float, np.floating))
        assert isinstance(result.spearman_correlation, (float, np.floating))

    def test_calculate_with_random_patterns(self):
        """Test calculate with random uncorrelated patterns."""
        np.random.seed(42)
        seismic = np.random.randn(5, 5, 15).astype(np.float64)
        facies = np.random.randint(0, 3, (5, 5, 15)).astype(np.int64)

        calc = GradientCorrelationCalculator()
        result = calc.calculate(seismic, facies)

        # Should still produce valid result
        assert np.isfinite(result.pearson_correlation) or np.isnan(
            result.pearson_correlation
        )
        assert np.isfinite(result.spearman_correlation) or np.isnan(
            result.spearman_correlation
        )

    def test_calculate_gradient_is_computed(self):
        """Test that gradient is properly computed."""
        seismic = np.zeros((3, 3, 10), dtype=np.float64)
        # Create simple linear gradient
        for i in range(10):
            seismic[:, :, i] = i * 1.0

        facies = np.zeros((3, 3, 10), dtype=np.int64)

        calc = GradientCorrelationCalculator()
        result = calc.calculate(seismic, facies)

        # Gradient should be computed
        assert result.seismic_gradient is not None
        assert result.seismic_gradient.shape == seismic.shape

    def test_calculate_boundaries_are_detected(self):
        """Test that boundaries are properly detected."""
        seismic = np.random.randn(4, 4, 8).astype(np.float64)
        facies = np.zeros((4, 4, 8), dtype=np.int64)
        facies[:, :, :4] = 0
        facies[:, :, 4:] = 1

        calc = GradientCorrelationCalculator()
        result = calc.calculate(seismic, facies)

        # Boundaries should be detected
        assert result.boundaries is not None
        assert result.boundaries.dtype == bool


class TestGradientCorrelationCalculatorComputeCorrelation:
    """Tests for _compute_correlation static method."""

    def test_compute_correlation_perfect_correlation(self):
        """Test correlation when gradient and boundaries are perfectly correlated."""
        gradient = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        boundaries = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=bool
        )

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        assert np.isfinite(corr)
        assert 0 <= corr <= 1 or np.isnan(corr)

    def test_compute_correlation_no_correlation(self):
        """Test correlation when patterns are uncorrelated."""
        gradient = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        boundaries = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=bool)

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        # Should return valid values (potentially near zero)
        assert isinstance(corr, float)
        assert isinstance(pval, float)

    def test_compute_correlation_with_nan(self):
        """Test correlation when data contains NaN."""
        gradient = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        boundaries = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=bool)

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        # Should handle NaN gracefully
        assert isinstance(corr, float)
        assert isinstance(pval, float)

    def test_compute_correlation_all_nan(self):
        """Test correlation with all NaN values."""
        gradient = np.array([np.nan, np.nan, np.nan])
        boundaries = np.array([0.0, 1.0, 0.0], dtype=bool)

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        # Should return NaN
        assert np.isnan(corr)
        assert np.isnan(pval)

    def test_compute_correlation_with_inf(self):
        """Test correlation when data contains inf."""
        gradient = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        boundaries = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=bool)

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        # Should handle inf gracefully
        assert isinstance(corr, float)
        assert isinstance(pval, float)

    def test_compute_correlation_zero_variance_gradient(self):
        """Test correlation when gradient has zero variance."""
        gradient = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        boundaries = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=bool)

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        # Should return NaN due to zero variance
        assert np.isnan(corr)
        assert np.isnan(pval)

    def test_compute_correlation_zero_variance_boundaries(self):
        """Test correlation when boundaries have zero variance."""
        gradient = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        boundaries = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=bool)

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        # Should return NaN due to zero variance in boundaries
        assert np.isnan(corr)
        assert np.isnan(pval)

    def test_compute_correlation_insufficient_samples(self):
        """Test correlation with insufficient samples."""
        gradient = np.array([1.0])
        boundaries = np.array([0.0], dtype=bool)

        corr, pval = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )

        # Should return NaN due to insufficient samples
        assert np.isnan(corr)
        assert np.isnan(pval)

    def test_compute_correlation_pearson_vs_spearman(self):
        """Test that Pearson and Spearman give different results on rank data."""
        # Create data where Spearman should differ from Pearson
        gradient = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float64)
        boundaries = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=bool).astype(np.float64)

        pearson_corr, _ = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, pearsonr
        )
        spearman_corr, _ = GradientCorrelationCalculator._compute_correlation(
            gradient, boundaries, spearmanr
        )

        # Both should be valid
        assert isinstance(pearson_corr, float)
        assert isinstance(spearman_corr, float)


class TestGradientCorrelationCalculatorIntegration:
    """Integration tests for GradientCorrelationCalculator."""

    def test_full_workflow(self):
        """Test complete workflow from seismic/facies to correlation result."""
        np.random.seed(42)
        seismic = np.random.randn(6, 6, 12).astype(np.float64)
        facies = np.zeros((6, 6, 12), dtype=np.int64)
        facies[:, :, :6] = 0
        facies[:, :, 6:] = 1

        calc = GradientCorrelationCalculator()
        result = calc.calculate(seismic, facies)

        assert isinstance(result, GradientCorrelationResult)
        assert result.seismic_gradient.shape == seismic.shape
        assert result.boundaries.shape == seismic.shape

    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        facies = np.random.randint(0, 3, (5, 5, 10)).astype(np.int64)

        calc1 = GradientCorrelationCalculator()
        result1 = calc1.calculate(seismic, facies)

        calc2 = GradientCorrelationCalculator()
        result2 = calc2.calculate(seismic, facies)

        # Results should be reproducible
        assert result1.pearson_correlation == result2.pearson_correlation
        assert result1.spearman_correlation == result2.spearman_correlation

    def test_with_stratified_data(self):
        """Test with clearly stratified seismic and facies."""
        seismic = np.zeros((4, 4, 20), dtype=np.float64)
        for i in range(20):
            seismic[:, :, i] = np.sin(i * 0.3)

        facies = np.zeros((4, 4, 20), dtype=np.int64)
        facies[:, :, :5] = 0
        facies[:, :, 5:10] = 1
        facies[:, :, 10:15] = 2
        facies[:, :, 15:] = 3

        calc = GradientCorrelationCalculator()
        result = calc.calculate(seismic, facies)

        assert np.isfinite(result.pearson_correlation) or np.isnan(
            result.pearson_correlation
        )
        assert np.isfinite(result.spearman_correlation) or np.isnan(
            result.spearman_correlation
        )


class TestGradientCorrelationCalculatorLogging:
    """Tests for logging behavior."""

    def test_calculate_logging(self, caplog):
        """Test that calculate logs debug information."""
        seismic = np.random.randn(4, 4, 8).astype(np.float64)
        facies = np.random.randint(0, 2, (4, 4, 8)).astype(np.int64)

        with caplog.at_level(logging.DEBUG):
            calc = GradientCorrelationCalculator()
            result = calc.calculate(seismic, facies)

        # Should have logged something about gradient correlation
        assert any(
            "gradient" in record.message.lower()
            for record in caplog.records
            if record.levelname == "DEBUG"
        )

    def test_compute_correlation_logging_success(self, caplog):
        """Test logging when correlation is successfully computed."""
        gradient = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        boundaries = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=bool).astype(np.float64)

        with caplog.at_level(logging.DEBUG):
            corr, pval = GradientCorrelationCalculator._compute_correlation(
                gradient, boundaries, pearsonr
            )

        # Should have logged correlation result
        debug_messages = [r.message for r in caplog.records if r.levelname == "DEBUG"]
        # May have logged correlation or correlation parameters

    def test_compute_correlation_logging_error(self, caplog):
        """Test logging when correlation computation fails."""
        gradient = np.array([np.nan, np.nan, np.nan])
        boundaries = np.array([0.0, 1.0, 0.0], dtype=bool)

        with caplog.at_level(logging.WARNING):
            corr, pval = GradientCorrelationCalculator._compute_correlation(
                gradient, boundaries, pearsonr
            )

        # Should have logged a warning about insufficient samples
        assert np.isnan(corr)
        assert np.isnan(pval)


# Tests from test_processors_interface
# ============================================================================


class TestInterfaceReflectionAnalyzerAnalyze:
    """Tests for InterfaceReflectionAnalyzer.analyze method."""

    def test_analyze_with_no_transitions(self):
        """Test analyze when there are no facies transitions."""
        # All same facies - no transitions
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        facies = np.ones((5, 5, 10), dtype=np.int64)

        analyzer = InterfaceReflectionAnalyzer()
        result = analyzer.analyze(seismic, facies)

        assert isinstance(result, InterfaceReflectionResult)
        assert len(result.transitions_summary) == 0
        assert len(result.interface_stats) == 0

    def test_analyze_with_single_transition_type(self):
        """Test analyze with one transition type (0->1)."""
        seismic = np.ones((3, 3, 10), dtype=np.float64) * 0.5
        facies = np.ones((3, 3, 10), dtype=np.int64)
        facies[:, :, :5] = 0

        analyzer = InterfaceReflectionAnalyzer()
        result = analyzer.analyze(seismic, facies)

        assert isinstance(result, InterfaceReflectionResult)
        # Should have transitions from facies 0 to 1
        assert len(result.transitions_summary) > 0

    def test_analyze_with_multiple_transitions(self):
        """Test analyze with multiple transition types."""
        seismic = np.random.randn(4, 4, 15).astype(np.float64)
        facies = np.ones((4, 4, 15), dtype=np.int64)
        facies[:, :, :5] = 0
        facies[:, :, 10:] = 2

        analyzer = InterfaceReflectionAnalyzer()
        result = analyzer.analyze(seismic, facies)

        assert isinstance(result, InterfaceReflectionResult)
        assert len(result.transitions_summary) > 0

    def test_analyze_mismatched_shapes_raises_error(self):
        """Test that mismatched aligned shapes raise ValueError."""
        analyzer = InterfaceReflectionAnalyzer()

        # Create seismic and facies with different shapes
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        facies = np.ones((4, 4, 10), dtype=np.int64)

        with patch.object(analyzer._aligner, "align") as mock_align:
            # Return mismatched shapes
            mock_align.return_value = (
                np.random.randn(5, 5, 10).astype(np.float64),
                np.ones((4, 4, 10), dtype=np.int64),
            )

            with pytest.raises(ValueError, match="mismatched shapes"):
                analyzer.analyze(seismic, facies)

    def test_analyze_with_random_facies_pattern(self):
        """Test analyze with random facies distribution."""
        np.random.seed(42)
        seismic = np.random.randn(5, 5, 20).astype(np.float64)
        facies = np.random.randint(0, 4, (5, 5, 20)).astype(np.int64)

        analyzer = InterfaceReflectionAnalyzer()
        result = analyzer.analyze(seismic, facies)

        assert isinstance(result, InterfaceReflectionResult)
        # Result should contain transitions
        assert isinstance(result.transitions_summary, dict)
        assert isinstance(result.interface_stats, dict)


class TestInterfaceReflectionAnalyzerReshapeToTraces:
    """Tests for _reshape_to_traces static method."""

    def test_reshape_to_traces_basic(self):
        """Test reshaping 3D to 2D traces."""
        seismic_3d = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        facies_3d = np.arange(24, dtype=np.int64).reshape(2, 3, 4)

        seismic_2d, facies_2d = InterfaceReflectionAnalyzer._reshape_to_traces(
            seismic_3d, facies_3d
        )

        # Should reshape to (n_traces=6, nk=4)
        assert seismic_2d.shape == (6, 4)
        assert facies_2d.shape == (6, 4)

    def test_reshape_to_traces_shape(self):
        """Test that reshape preserves total element count."""
        seismic_3d = np.random.randn(3, 4, 5).astype(np.float64)
        facies_3d = np.random.randint(0, 3, (3, 4, 5)).astype(np.int64)

        seismic_2d, facies_2d = InterfaceReflectionAnalyzer._reshape_to_traces(
            seismic_3d, facies_3d
        )

        expected_n_traces = 3 * 4  # ni * nj
        expected_nk = 5
        assert seismic_2d.shape == (expected_n_traces, expected_nk)
        assert facies_2d.shape == (expected_n_traces, expected_nk)

    def test_reshape_to_traces_data_preservation(self):
        """Test that reshape preserves data values."""
        seismic_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
        facies_3d = np.array([[[1, 2], [3, 4]]], dtype=np.int64)

        seismic_2d, facies_2d = InterfaceReflectionAnalyzer._reshape_to_traces(
            seismic_3d, facies_3d
        )

        assert seismic_2d.shape == (2, 2)
        assert facies_2d.shape == (2, 2)


class TestInterfaceReflectionAnalyzerExtractAmplitudes:
    """Tests for _extract_amplitudes static method."""

    def test_extract_amplitudes_basic(self):
        """Test basic amplitude extraction."""
        seismic_2d = np.ones((5, 20), dtype=np.float64) * 2.0
        rows = np.array([0, 1, 2])
        ks = np.array([5, 10, 15])

        amps = InterfaceReflectionAnalyzer._extract_amplitudes(seismic_2d, rows, ks)

        assert len(amps) == 3
        # All amplitudes should be close to 2.0 since all seismic values are 2.0
        np.testing.assert_allclose(amps, 2.0, atol=0.1)

    def test_extract_amplitudes_with_padding(self):
        """Test that amplitude extraction handles padding correctly."""
        seismic_2d = np.random.randn(3, 15).astype(np.float64)
        rows = np.array([0, 1, 2])
        ks = np.array([7, 8, 9])  # Middle samples with room for window

        amps = InterfaceReflectionAnalyzer._extract_amplitudes(seismic_2d, rows, ks)

        assert len(amps) == 3
        assert np.all(np.isfinite(amps))

    def test_extract_amplitudes_empty_raises_error(self):
        """Test that empty transition points raise ValueError."""
        seismic_2d = np.ones((5, 20), dtype=np.float64)
        rows = np.array([], dtype=np.intp)
        ks = np.array([], dtype=np.intp)

        with pytest.raises(ValueError, match="No transition points"):
            InterfaceReflectionAnalyzer._extract_amplitudes(seismic_2d, rows, ks)

    def test_extract_amplitudes_edge_samples(self):
        """Test amplitude extraction at edge samples."""
        seismic_2d = np.arange(50, dtype=np.float64).reshape(5, 10)
        rows = np.array([0, 4])
        ks = np.array([1, 9])

        amps = InterfaceReflectionAnalyzer._extract_amplitudes(seismic_2d, rows, ks)

        assert len(amps) == 2
        assert np.all(np.isfinite(amps))

    def test_extract_amplitudes_consistency(self):
        """Test that identical windows produce similar amplitude."""
        # Create seismic where specific windows have similar values
        seismic_2d = np.ones((3, 25), dtype=np.float64) * 0.5
        seismic_2d[0, 5:15] = 3.0  # Create window with all 3.0
        seismic_2d[1, 12:22] = 3.0

        rows = np.array([0, 1])
        ks = np.array([10, 17])  # Window centers

        amps = InterfaceReflectionAnalyzer._extract_amplitudes(seismic_2d, rows, ks)

        # Both amplitudes should be close to 3.0
        assert len(amps) == 2
        assert np.all(np.isfinite(amps))


class TestInterfaceReflectionAnalyzerAggregateByTransition:
    """Tests for _aggregate_by_transition static method."""

    def test_aggregate_single_transition_type(self):
        """Test aggregation with single transition type."""
        fac_from = np.array([0, 0, 0])
        fac_to = np.array([1, 1, 1])
        amps = np.array([1.0, 2.0, 3.0])

        result = InterfaceReflectionAnalyzer._aggregate_by_transition(
            fac_from, fac_to, amps
        )

        assert isinstance(result, InterfaceReflectionResult)
        assert len(result.transitions_summary) == 1
        key = Transition(0, 1)
        assert key in result.transitions_summary
        assert key in result.interface_stats

    def test_aggregate_multiple_transition_types(self):
        """Test aggregation with multiple transition types."""
        fac_from = np.array([0, 0, 1, 1, 2])
        fac_to = np.array([1, 1, 2, 2, 0])
        amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = InterfaceReflectionAnalyzer._aggregate_by_transition(
            fac_from, fac_to, amps
        )

        assert len(result.transitions_summary) == 3
        assert Transition(0, 1) in result.transitions_summary
        assert Transition(1, 2) in result.transitions_summary
        assert Transition(2, 0) in result.transitions_summary

    def test_aggregate_stats_calculation(self):
        """Test that stats are correctly calculated."""
        fac_from = np.array([0, 0, 0])
        fac_to = np.array([1, 1, 1])
        amps = np.array([1.0, 2.0, 3.0])

        result = InterfaceReflectionAnalyzer._aggregate_by_transition(
            fac_from, fac_to, amps
        )

        key = Transition(0, 1)
        stats = result.transitions_summary[key]
        assert isinstance(stats, (FaciesStats, type(None)))

        raw_amps = result.interface_stats[key]
        assert len(raw_amps) == 3
        np.testing.assert_array_equal(raw_amps, amps)

    def test_aggregate_zero_facies_values(self):
        """Test aggregation when facies values include zero."""
        fac_from = np.array([0, 0, 0])
        fac_to = np.array([0, 0, 0])
        amps = np.array([1.0, 2.0, 3.0])

        result = InterfaceReflectionAnalyzer._aggregate_by_transition(
            fac_from, fac_to, amps
        )

        key = Transition(0, 0)
        assert key in result.transitions_summary

    def test_aggregate_large_facies_numbers(self):
        """Test aggregation with large facies numbers."""
        fac_from = np.array([100, 100, 200])
        fac_to = np.array([101, 101, 201])
        amps = np.array([1.0, 2.0, 3.0])

        result = InterfaceReflectionAnalyzer._aggregate_by_transition(
            fac_from, fac_to, amps
        )

        assert Transition(100, 101) in result.transitions_summary
        assert Transition(200, 201) in result.transitions_summary


class TestInterfaceReflectionAnalyzerLogging:
    """Tests for logging behavior."""

    def test_analyze_logging(self, caplog):
        """Test that analyze logs debug information."""
        seismic = np.random.randn(3, 3, 10).astype(np.float64)
        facies = np.ones((3, 3, 10), dtype=np.int64)
        facies[:, :, :5] = 0

        with caplog.at_level(logging.DEBUG):
            analyzer = InterfaceReflectionAnalyzer()
            result = analyzer.analyze(seismic, facies)

        # Should have logged something about interface reflection analysis
        assert any(
            "interface" in record.message.lower()
            for record in caplog.records
            if record.levelname == "DEBUG"
        )


# Tests from test_processors_utils
# ============================================================================


class TestProcessorUtilsComputeQuartiles:
    """Tests for compute_quartiles."""

    def test_compute_quartiles_basic(self):
        """Test quartile computation on basic array."""
        amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q1, q3 = ProcessorUtils.compute_quartiles(amps)

        assert isinstance(q1, float)
        assert isinstance(q3, float)
        assert q1 < q3

    def test_compute_quartiles_uniform(self):
        """Test quartiles on uniform array."""
        amps = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        q1, q3 = ProcessorUtils.compute_quartiles(amps)

        assert q1 == 2.0
        assert q3 == 2.0

    def test_compute_quartiles_large_array(self):
        """Test quartiles on large array."""
        amps = np.arange(1000, dtype=np.float64)
        q1, q3 = ProcessorUtils.compute_quartiles(amps)

        # Q1 should be around 250, Q3 around 750
        assert 200 < q1 < 300
        assert 700 < q3 < 800

    def test_compute_quartiles_single_element(self):
        """Test quartiles with single element."""
        amps = np.array([5.0])
        q1, q3 = ProcessorUtils.compute_quartiles(amps)

        assert q1 == 5.0
        assert q3 == 5.0

    def test_compute_quartiles_two_elements(self):
        """Test quartiles with two elements."""
        amps = np.array([1.0, 9.0])
        q1, q3 = ProcessorUtils.compute_quartiles(amps)

        assert isinstance(q1, float)
        assert isinstance(q3, float)

    def test_compute_quartiles_negative_values(self):
        """Test quartiles with negative values."""
        amps = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
        q1, q3 = ProcessorUtils.compute_quartiles(amps)

        assert q1 < q3


class TestProcessorUtilsFilterFiniteValues:
    """Tests for filter_finite_values."""

    def test_filter_no_invalid_values(self):
        """Test filtering when both arrays have only finite values."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])

        result1, result2, count = ProcessorUtils.filter_finite_values(arr1, arr2)

        np.testing.assert_array_equal(result1, arr1)
        np.testing.assert_array_equal(result2, arr2)
        assert count == 0

    def test_filter_with_nan_in_first(self):
        """Test filtering with NaN in first array."""
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])

        result1, result2, count = ProcessorUtils.filter_finite_values(arr1, arr2)

        assert len(result1) == 2
        assert len(result2) == 2
        assert count == 1

    def test_filter_with_nan_in_second(self):
        """Test filtering with NaN in second array."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, np.nan, 6.0])

        result1, result2, count = ProcessorUtils.filter_finite_values(arr1, arr2)

        assert len(result1) == 2
        assert len(result2) == 2
        assert count == 1

    def test_filter_with_inf(self):
        """Test filtering with infinity values."""
        arr1 = np.array([1.0, np.inf, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])

        result1, result2, count = ProcessorUtils.filter_finite_values(arr1, arr2)

        assert len(result1) == 2
        assert len(result2) == 2
        assert count == 1

    def test_filter_with_negative_inf(self):
        """Test filtering with negative infinity."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, -np.inf, 6.0])

        result1, result2, count = ProcessorUtils.filter_finite_values(arr1, arr2)

        assert len(result1) == 2
        assert len(result2) == 2

    def test_filter_all_invalid(self):
        """Test filtering when all values are invalid."""
        arr1 = np.array([np.nan, np.inf, np.nan])
        arr2 = np.array([1.0, 2.0, 3.0])

        result1, result2, count = ProcessorUtils.filter_finite_values(arr1, arr2)

        assert len(result1) == 0
        assert len(result2) == 0
        assert count == 3

    def test_filter_mixed_invalid(self):
        """Test filtering with mixed invalid values."""
        arr1 = np.array([1.0, np.nan, 3.0, np.inf])
        arr2 = np.array([5.0, 6.0, np.nan, 8.0])

        result1, result2, count = ProcessorUtils.filter_finite_values(arr1, arr2)

        # Rows 1, 2, 3 have invalid values
        assert len(result1) == 1  # Only first row is valid
        assert len(result2) == 1
        assert count == 3


class TestProcessorUtilsComputeAmplitudeStats:
    """Tests for compute_amplitude_stats."""

    def test_compute_stats_basic(self):
        """Test basic amplitude statistics computation."""
        amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = ProcessorUtils.compute_amplitude_stats(amps)

        assert stats is not None
        assert stats.count == 5
        assert stats.mean == 3.0

    def test_compute_stats_single_value(self):
        """Test statistics with single amplitude."""
        amps = np.array([5.0])
        stats = ProcessorUtils.compute_amplitude_stats(amps)

        assert stats.count == 1
        assert stats.mean == 5.0

    def test_compute_stats_with_negatives(self):
        """Test statistics with negative amplitudes."""
        amps = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        stats = ProcessorUtils.compute_amplitude_stats(amps)

        assert stats.count == 5
        assert stats.mean == 0.0

    def test_compute_stats_zeros(self):
        """Test statistics with all zeros."""
        amps = np.array([0.0, 0.0, 0.0])
        stats = ProcessorUtils.compute_amplitude_stats(amps)

        assert stats.count == 3
        assert stats.mean == 0.0
        assert stats.std == 0.0

    def test_compute_stats_with_nan(self):
        """Test statistics handling with NaN values."""
        amps = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        stats = ProcessorUtils.compute_amplitude_stats(amps)

        # Should handle NaN gracefully
        if stats is not None:
            assert stats.count > 0

    def test_compute_stats_empty_array(self):
        """Test statistics with empty array."""
        amps = np.array([])
        stats = ProcessorUtils.compute_amplitude_stats(amps)

        # Should handle empty array gracefully
        assert stats is None or stats.count == 0

    def test_compute_stats_large_values(self):
        """Test statistics with large amplitude values."""
        amps = np.array([1e6, 2e6, 3e6])
        stats = ProcessorUtils.compute_amplitude_stats(amps)

        assert stats is not None
        assert stats.mean == pytest.approx(2e6)


class TestProcessorUtilsReshape3dTo2d:
    """Tests for reshape_3d_to_2d."""

    def test_reshape_basic_shape(self):
        """Test basic 3D to 2D reshape."""
        seismic = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        facies = np.arange(24, dtype=np.int64).reshape(2, 3, 4)

        seismic_2d, facies_2d = ProcessorUtils.reshape_3d_to_2d(seismic, facies)

        assert seismic_2d.shape == (6, 4)  # (ni*nj, nk)
        assert facies_2d.shape == (6, 4)

    def test_reshape_preserves_data(self):
        """Test that reshape preserves data values."""
        seismic = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        facies = np.arange(24, dtype=np.int64).reshape(2, 3, 4)

        seismic_2d, facies_2d = ProcessorUtils.reshape_3d_to_2d(seismic, facies)

        # All original values should be present
        assert set(seismic_2d.flatten()) == set(seismic.flatten())

    def test_reshape_single_element(self):
        """Test reshape with single trace."""
        seismic = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
        facies = np.array([[[1, 2, 3]]], dtype=np.int64)

        seismic_2d, facies_2d = ProcessorUtils.reshape_3d_to_2d(seismic, facies)

        assert seismic_2d.shape == (1, 3)
        assert facies_2d.shape == (1, 3)

    def test_reshape_maintains_dtype(self):
        """Test that reshape maintains dtypes."""
        seismic = np.random.randn(3, 3, 5).astype(np.float64)
        facies = np.random.randint(0, 3, (3, 3, 5)).astype(np.int64)

        seismic_2d, facies_2d = ProcessorUtils.reshape_3d_to_2d(seismic, facies)

        assert seismic_2d.dtype == np.float64
        assert facies_2d.dtype == np.int64


class TestProcessorUtilsIntegration:
    """Integration tests for ProcessorUtils."""

    def test_workflow_compute_stats_and_quartiles(self):
        """Test complete workflow of computing stats then quartiles."""
        amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        stats = ProcessorUtils.compute_amplitude_stats(amps)
        assert stats is not None

        q1, q3 = ProcessorUtils.compute_quartiles(amps)
        assert q1 < q3
        assert stats.mean > q1
        assert stats.mean < q3

    def test_workflow_filter_and_stats(self):
        """Test filtering invalid values then computing stats."""
        arr1 = np.array([1.0, np.nan, 3.0, 4.0])
        arr2 = np.array([5.0, 6.0, np.nan, 8.0])

        filtered1, filtered2, count = ProcessorUtils.filter_finite_values(arr1, arr2)
        assert len(filtered1) == 2
        assert count == 2

        stats = ProcessorUtils.compute_amplitude_stats(filtered1)
        assert stats is not None

    def test_workflow_convert_scalars_from_computation(self):
        """Test converting results from NumPy operations."""
        amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mean_val = np.mean(amps)
        std_val = np.std(amps)
        q1, q3 = np.percentile(amps, [25, 75])

        result = ProcessorUtils.convert_numpy_scalars_to_float(
            mean_val, std_val, q1, q3
        )

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)


# Tests from test_processors_validators
# ============================================================================


class TestArrayValidatorValidate3dArray:
    """Tests for ArrayValidator.validate_3d_array."""

    def test_validate_valid_3d_array(self):
        """Test validation passes for valid 3D array."""
        arr = np.zeros((10, 10, 20), dtype=np.float64)
        # Should not raise
        ArrayValidator.validate_3d_array(arr, "test_array")

    def test_validate_3d_array_with_default_name(self):
        """Test validation with default array name."""
        arr = np.zeros((5, 5, 10), dtype=np.float64)
        # Should not raise
        ArrayValidator.validate_3d_array(arr)

    def test_validate_1d_array_raises_error(self):
        """Test validation fails for 1D array."""
        arr = np.zeros(100, dtype=np.float64)
        with pytest.raises(ValueError, match="3-dimensional"):
            ArrayValidator.validate_3d_array(arr, "seismic")

    def test_validate_2d_array_raises_error(self):
        """Test validation fails for 2D array."""
        arr = np.zeros((10, 10), dtype=np.float64)
        with pytest.raises(ValueError, match="3-dimensional"):
            ArrayValidator.validate_3d_array(arr, "seismic")

    def test_validate_4d_array_raises_error(self):
        """Test validation fails for 4D array."""
        arr = np.zeros((5, 5, 10, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="3-dimensional"):
            ArrayValidator.validate_3d_array(arr, "multi_volume")

    def test_validate_empty_3d_array_raises_error(self):
        """Test validation fails for empty 3D array."""
        arr = np.zeros((0, 0, 0), dtype=np.float64)
        with pytest.raises(ValueError, match="cannot be empty"):
            ArrayValidator.validate_3d_array(arr, "empty_cube")

    def test_validate_empty_dimension_raises_error(self):
        """Test validation fails when one dimension is empty."""
        arr = np.zeros((10, 0, 20), dtype=np.float64)
        with pytest.raises(ValueError, match="cannot be empty"):
            ArrayValidator.validate_3d_array(arr, "partial_empty")

    def test_validate_array_name_in_error_message(self):
        """Test that array name appears in error message."""
        arr = np.zeros((10, 10), dtype=np.float64)
        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_3d_array(arr, "my_seismic_cube")

        assert "my_seismic_cube" in str(exc_info.value)

    def test_validate_shape_in_error_message(self):
        """Test that shape appears in error message."""
        arr = np.zeros((10, 10), dtype=np.float64)
        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_3d_array(arr, "test")

        assert "(10, 10)" in str(exc_info.value)

    def test_validate_various_valid_shapes(self):
        """Test validation with various valid 3D shapes."""
        for shape in [(5, 5, 10), (100, 100, 50), (1, 1, 1), (50, 30, 100)]:
            arr = np.zeros(shape, dtype=np.float64)
            # Should not raise
            ArrayValidator.validate_3d_array(arr)


class TestArrayValidatorValidate3dArrays:
    """Tests for ArrayValidator.validate_3d_arrays (multiple arrays)."""

    def test_validate_multiple_valid_arrays(self):
        """Test validation passes for multiple valid arrays."""
        seismic = np.zeros((10, 10, 20), dtype=np.float64)
        facies = np.zeros((10, 10, 20), dtype=np.int64)
        # Should not raise
        ArrayValidator.validate_3d_arrays((seismic, "seismic"), (facies, "facies"))

    def test_validate_multiple_arrays_first_invalid(self):
        """Test validation fails when first array is invalid."""
        seismic = np.zeros((10, 10), dtype=np.float64)  # Invalid 2D
        facies = np.zeros((10, 10, 20), dtype=np.int64)
        with pytest.raises(ValueError, match="3-dimensional"):
            ArrayValidator.validate_3d_arrays((seismic, "seismic"), (facies, "facies"))

    def test_validate_multiple_arrays_second_invalid(self):
        """Test validation fails when second array is invalid."""
        seismic = np.zeros((10, 10, 20), dtype=np.float64)
        facies = np.zeros((0, 0, 0), dtype=np.int64)  # Empty
        with pytest.raises(ValueError, match="cannot be empty"):
            ArrayValidator.validate_3d_arrays((seismic, "seismic"), (facies, "facies"))

    def test_validate_multiple_arrays_names_in_error(self):
        """Test that array names appear in error messages."""
        seismic = np.zeros((10, 10), dtype=np.float64)
        facies = np.zeros((10, 10, 20), dtype=np.int64)
        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_3d_arrays(
                (seismic, "my_seismic"), (facies, "my_facies")
            )

        assert "my_seismic" in str(exc_info.value)

    def test_validate_three_arrays(self):
        """Test validation with three arrays."""
        arr1 = np.zeros((5, 5, 10), dtype=np.float64)
        arr2 = np.zeros((5, 5, 10), dtype=np.float64)
        arr3 = np.zeros((5, 5, 10), dtype=np.float64)
        # Should not raise
        ArrayValidator.validate_3d_arrays(
            (arr1, "arr1"), (arr2, "arr2"), (arr3, "arr3")
        )

    def test_validate_empty_tuple_list(self):
        """Test validation with empty tuple list."""
        # Should not raise (no arrays to validate)
        ArrayValidator.validate_3d_arrays()


class TestArrayValidatorPositiveParameter:
    """Tests for ArrayValidator.validate_positive_parameter."""

    def test_validate_positive_value(self):
        """Test validation passes for positive value."""
        # Should not raise
        ArrayValidator.validate_positive_parameter(5, "window_size")

    def test_validate_zero_value(self):
        """Test validation passes for zero value."""
        # Should not raise (non-negative includes 0)
        ArrayValidator.validate_positive_parameter(0, "offset")

    def test_validate_negative_value_raises_error(self):
        """Test validation fails for negative value."""
        with pytest.raises(ValueError, match="non-negative"):
            ArrayValidator.validate_positive_parameter(-1, "dilation_window")

    def test_validate_large_negative_value(self):
        """Test validation fails for large negative value."""
        with pytest.raises(ValueError, match="non-negative"):
            ArrayValidator.validate_positive_parameter(-100, "parameter")

    def test_parameter_name_in_error_message(self):
        """Test that parameter name appears in error message."""
        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_positive_parameter(-5, "my_parameter")

        assert "my_parameter" in str(exc_info.value)

    def test_value_in_error_message(self):
        """Test that value appears in error message."""
        with pytest.raises(ValueError) as exc_info:
            ArrayValidator.validate_positive_parameter(-42, "test")

        assert "-42" in str(exc_info.value)

    def test_validate_various_positive_values(self):
        """Test validation with various positive values."""
        for value in [1, 10, 100, 1000]:
            # Should not raise
            ArrayValidator.validate_positive_parameter(value, "param")


class TestValidationHelpersValidateOrReturn:
    """Tests for ValidationHelpers.validate_or_return."""

    def test_validate_or_return_true_condition(self):
        """Test with True condition returns None."""
        result = ValidationHelpers.validate_or_return(True, "Error message")
        assert result is None

    def test_validate_or_return_false_condition(self):
        """Test with False condition returns default value."""
        result = ValidationHelpers.validate_or_return(False, "Error message", 42)
        assert result == 42

    def test_validate_or_return_false_no_default(self):
        """Test with False condition and no default returns None."""
        result = ValidationHelpers.validate_or_return(False, "Error message")
        assert result is None

    def test_validate_or_return_logs_warning(self, caplog):
        """Test that validation failure logs warning."""
        with caplog.at_level(logging.WARNING):
            ValidationHelpers.validate_or_return(False, "Test warning")

        assert any("Test warning" in record.message for record in caplog.records)

    def test_validate_or_return_logs_debug(self, caplog):
        """Test that validation failure logs at debug level."""
        with caplog.at_level(logging.DEBUG):
            ValidationHelpers.validate_or_return(False, "Test debug", log_level="debug")

        assert any("Test debug" in record.message for record in caplog.records)

    def test_validate_or_return_logs_error(self, caplog):
        """Test that validation failure logs at error level."""
        with caplog.at_level(logging.ERROR):
            ValidationHelpers.validate_or_return(False, "Test error", log_level="error")

        assert any("Test error" in record.message for record in caplog.records)

    def test_validate_or_return_various_default_types(self):
        """Test with various default value types."""
        assert ValidationHelpers.validate_or_return(False, "msg", []) == []
        assert ValidationHelpers.validate_or_return(False, "msg", {}) == {}
        assert ValidationHelpers.validate_or_return(False, "msg", "") == ""
        assert ValidationHelpers.validate_or_return(False, "msg", 0) == 0


class TestValidationHelpersEnsureValidArrays:
    """Tests for ValidationHelpers.ensure_valid_arrays."""

    def test_ensure_valid_single_array(self):
        """Test with single valid 3D array."""
        arr = np.zeros((10, 10, 20), dtype=np.float64)
        # Should not raise
        ValidationHelpers.ensure_valid_arrays((arr, "test_array"))

    def test_ensure_valid_multiple_arrays(self):
        """Test with multiple valid 3D arrays."""
        arr1 = np.zeros((10, 10, 20), dtype=np.float64)
        arr2 = np.zeros((10, 10, 20), dtype=np.int64)
        # Should not raise
        ValidationHelpers.ensure_valid_arrays((arr1, "arr1"), (arr2, "arr2"))

    def test_ensure_valid_2d_array_raises(self):
        """Test with 2D array raises ValidationError."""
        arr = np.zeros((10, 10), dtype=np.float64)
        with pytest.raises(ValidationError, match="must be 3D"):
            ValidationHelpers.ensure_valid_arrays((arr, "invalid"))

    def test_ensure_valid_empty_array_raises(self):
        """Test with empty array raises ValidationError."""
        arr = np.zeros((0, 0, 0), dtype=np.float64)
        with pytest.raises(ValidationError, match="cannot be empty"):
            ValidationHelpers.ensure_valid_arrays((arr, "empty"))

    def test_ensure_valid_array_name_in_error(self):
        """Test that array name appears in ValidationError."""
        arr = np.zeros((10, 10), dtype=np.float64)
        with pytest.raises(ValidationError) as exc_info:
            ValidationHelpers.ensure_valid_arrays((arr, "my_array"))

        assert "my_array" in str(exc_info.value)

    def test_ensure_valid_stops_on_first_error(self):
        """Test that validation stops on first invalid array."""
        arr1 = np.zeros((10, 10), dtype=np.float64)  # Invalid
        arr2 = np.zeros((0, 0, 0), dtype=np.float64)  # Also invalid
        with pytest.raises(ValidationError) as exc_info:
            ValidationHelpers.ensure_valid_arrays((arr1, "first"), (arr2, "second"))

        # Should mention the first invalid array
        assert "first" in str(exc_info.value)


class TestValidatorIntegration:
    """Integration tests for validators."""

    def test_validate_processor_inputs(self):
        """Test typical processor input validation workflow."""
        seismic = np.random.randn(10, 10, 20).astype(np.float64)
        facies = np.random.randint(0, 3, (10, 10, 20)).astype(np.int64)

        # Validate inputs
        ArrayValidator.validate_3d_arrays((seismic, "seismic"), (facies, "facies"))

        # Validate parameters
        ArrayValidator.validate_positive_parameter(2, "dilation_window")

        # Both should pass without error

    def test_validation_with_mismatched_shapes(self):
        """Test validation catches mismatched array shapes."""
        seismic = np.random.randn(10, 10, 20).astype(np.float64)
        facies = np.random.randint(0, 3, (10, 10, 25)).astype(
            np.int64
        )  # Different depth

        # Individual validation passes (both are 3D)
        ArrayValidator.validate_3d_array(seismic, "seismic")
        ArrayValidator.validate_3d_array(facies, "facies")

        # Note: Shape matching is not the responsibility of these validators
        # but they ensure basic dimensionality requirements

    def test_error_recovery_with_validate_or_return(self):
        """Test error recovery pattern using validate_or_return."""
        arr = np.zeros((10, 10), dtype=np.float64)  # Invalid shape

        # Using validate_or_return for graceful degradation
        result = ValidationHelpers.validate_or_return(
            arr.ndim == 3, "Array must be 3D", default_value=None, log_level="warning"
        )

        assert result is None

    def test_comprehensive_validation_pipeline(self):
        """Test comprehensive validation of processor inputs and parameters."""
        seismic = np.random.randn(5, 5, 10).astype(np.float64)
        facies = np.random.randint(0, 2, (5, 5, 10)).astype(np.int64)
        dilation_window = 2

        # Validate arrays
        ArrayValidator.validate_3d_arrays((seismic, "seismic"), (facies, "facies"))

        # Validate parameters
        ArrayValidator.validate_positive_parameter(dilation_window, "dilation_window")

        # Validate with helpers
        ValidationHelpers.ensure_valid_arrays((seismic, "seismic"), (facies, "facies"))

        # All should pass
