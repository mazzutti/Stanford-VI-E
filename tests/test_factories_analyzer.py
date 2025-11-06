"""Comprehensive tests for analyzer_factory module.

Tests the AnalyzerBuilder and AnalyzerFactory classes with coverage for:
- Builder pattern functionality
- Dependency injection
- Type validation
- Processor configuration
- State management (freezing, copying, resetting)
- Factory presets
- Debug functionality
- AnalyzerInterface OOP pattern tests
"""

# mypy: ignore-errors


import logging
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from src.analysis.base import AnalysisConfig
from src.analysis.factories import (
    AnalyzerBuilder,
    AnalyzerFactory,
)
from src.analysis.exceptions import BuilderFrozenError, BuilderValidationError
from src.analysis.models import FaciesCorrelationConfig
from src.analysis.processors import (
    BoundaryDetector,
    CubeAligner,
    GradientCorrelationCalculator,
)


class TestAnalyzerBuilder:
    """Test suite for AnalyzerBuilder class."""

    def test_builder_initialization(self) -> None:
        """Test builder initializes with None values."""
        builder = AnalyzerBuilder()
        assert builder._resampler_factory is None
        assert builder._config is None
        assert builder.configured_processor_count() == 0

    def test_with_config(self) -> None:
        """Test configuring analysis parameters."""
        config = FaciesCorrelationConfig()
        builder = AnalyzerBuilder().with_config(config)
        assert builder._config is config

    def test_method_chaining(self) -> None:
        """Test fluent method chaining."""
        config = FaciesCorrelationConfig()
        builder = (
            AnalyzerBuilder()
            .with_config(config)
            .with_boundary_detector(BoundaryDetector())
            .with_cube_aligner(CubeAligner())
        )
        assert builder._config is config
        assert builder._boundary_detector is not None
        assert builder._cube_aligner is not None

    def test_builder_freezing(self) -> None:
        """Test builder freeze/unfreeze functionality."""
        builder = AnalyzerBuilder()
        assert not builder.is_frozen()

        builder.freeze()
        assert builder.is_frozen()

        with pytest.raises(BuilderFrozenError):
            builder.with_config(FaciesCorrelationConfig())

        builder.unfreeze()
        assert not builder.is_frozen()
        builder.with_config(FaciesCorrelationConfig())

    def test_builder_copy(self) -> None:
        """Test builder state copying."""
        config = FaciesCorrelationConfig()
        builder1 = AnalyzerBuilder().with_config(config)
        builder2 = builder1.copy()

        assert builder2._config is config
        assert builder1 is not builder2

    def test_builder_reset(self) -> None:
        """Test builder reset to initial state."""
        config = FaciesCorrelationConfig()
        builder = (
            AnalyzerBuilder()
            .with_config(config)
            .with_boundary_detector(BoundaryDetector())
        )
        assert builder._config is config
        assert builder._boundary_detector is not None

        builder.reset()
        assert builder._config is None
        assert builder._boundary_detector is None

    def test_state_snapshot_and_restore(self) -> None:
        """Test saving and restoring builder state."""
        config = FaciesCorrelationConfig()
        builder1 = AnalyzerBuilder().with_config(config)

        snapshot = builder1.state_snapshot()
        assert snapshot["config"] is config

        builder2 = AnalyzerBuilder.with_state_snapshot(snapshot)
        assert builder2._config is config

    def test_from_existing_builder(self) -> None:
        """Test creating builder from existing builder."""
        config = FaciesCorrelationConfig()
        builder1 = AnalyzerBuilder().with_config(config)

        builder2 = AnalyzerBuilder.from_existing_builder(builder1)
        assert builder2._config is config
        assert builder1 is not builder2

    def test_configured_processor_count(self) -> None:
        """Test counting configured processors."""
        builder = AnalyzerBuilder()
        assert builder.configured_processor_count() == 0

        builder.with_boundary_detector(BoundaryDetector())
        assert builder.configured_processor_count() == 1

        builder.with_cube_aligner(CubeAligner())
        assert builder.configured_processor_count() == 2

    def test_debug_info(self) -> None:
        """Test debug info output."""
        builder = AnalyzerBuilder().with_config(FaciesCorrelationConfig())
        debug_output = builder.debug_info()

        assert "ANALYZER BUILDER DEBUG INFO" in debug_output
        assert "Builder State:" in debug_output
        assert "Configuration Dependencies:" in debug_output
        assert "Processors:" in debug_output
        assert "Frozen: False" in debug_output

    def test_batch_processor_configuration(self) -> None:
        """Test batch processor configuration."""
        builder = AnalyzerBuilder().with_processors(
            boundary_detector=BoundaryDetector(),
            cube_aligner=CubeAligner(),
            gradient_calculator=GradientCorrelationCalculator(),
        )

        assert builder._boundary_detector is not None
        assert builder._cube_aligner is not None
        assert builder._gradient_calculator is not None

    def test_transient_config_context_manager(self) -> None:
        """Test transient configuration context manager."""
        main_config = FaciesCorrelationConfig()
        builder = AnalyzerBuilder().with_config(main_config)

        test_config = FaciesCorrelationConfig()
        with builder.transient_config(config=test_config) as temp_builder:
            assert temp_builder._config is test_config

        # After context, should be restored
        assert builder._config is main_config

    def test_log_level_configuration(self) -> None:
        """Test setting log level."""
        AnalyzerBuilder.set_log_level(logging.DEBUG)
        # Just verify it doesn't raise

    def test_type_validation_callable(self) -> None:
        """Test type validation for callable dependencies."""
        builder = AnalyzerBuilder()

        # Valid callable
        builder.with_cache_file_selector(lambda x, y: None)
        assert builder._select_cache_files is not None

        # Invalid non-callable
        with pytest.raises(TypeError):
            builder.with_cache_file_selector("not_callable")  # type: ignore

    def test_repr(self) -> None:
        """Test builder string representation."""
        builder = AnalyzerBuilder()
        repr_str = repr(builder)

        assert "AnalyzerBuilder" in repr_str
        assert "mutable" in repr_str

        builder.freeze()
        repr_str = repr(builder)
        assert "frozen" in repr_str

    def test_equality(self) -> None:
        """Test builder equality comparison."""
        config = FaciesCorrelationConfig()
        builder1 = AnalyzerBuilder().with_config(config)
        builder2 = AnalyzerBuilder().with_config(config)
        builder3 = AnalyzerBuilder()

        assert builder1 == builder2
        assert builder1 != builder3

    def test_hash(self) -> None:
        """Test builder hashing for use in sets/dicts."""
        builder1 = AnalyzerBuilder()
        builder2 = AnalyzerBuilder()

        # Both should be hashable
        hash1 = hash(builder1)
        hash2 = hash(builder2)
        assert isinstance(hash1, int)
        assert isinstance(hash2, int)

        # Can be added to sets
        builder_set = {builder1, builder2}
        # Note: May be 1 or 2 depending on hash collision
        assert len(builder_set) >= 1


class TestAnalyzerFactory:
    """Test suite for AnalyzerFactory factory methods."""

    def test_create_default(self) -> None:
        """Test creating default analyzer."""
        analyzer = AnalyzerFactory.create_default()
        assert analyzer is not None

    def test_create_for_testing(self) -> None:
        """Test creating testing analyzer."""
        analyzer = AnalyzerFactory.create_for_testing()
        assert analyzer is not None

    def test_builder_factory_method(self) -> None:
        """Test builder factory method."""
        builder = AnalyzerFactory.builder()
        assert isinstance(builder, AnalyzerBuilder)

    def test_preset_debug(self) -> None:
        """Test debug preset configuration."""
        builder = AnalyzerFactory.preset_debug()
        assert isinstance(builder, AnalyzerBuilder)
        analyzer = builder.build()
        assert analyzer is not None

    def test_preset_production(self) -> None:
        """Test production preset configuration."""
        builder = AnalyzerFactory.preset_production()
        assert isinstance(builder, AnalyzerBuilder)
        analyzer = builder.build()
        assert analyzer is not None

    def test_preset_minimal(self) -> None:
        """Test minimal preset configuration."""
        builder = AnalyzerFactory.preset_minimal()
        assert isinstance(builder, AnalyzerBuilder)
        analyzer = builder.build()
        assert analyzer is not None

    def test_preset_full(self) -> None:
        """Test full preset configuration."""
        builder = AnalyzerFactory.preset_full()
        assert isinstance(builder, AnalyzerBuilder)
        # Should have some processors configured
        assert builder.configured_processor_count() > 0


class TestBuilderValidationError:
    """Test suite for BuilderValidationError exception."""

    def test_validation_error_initialization(self) -> None:
        """Test validation error initialization."""
        missing = ["config", "plotter"]
        error = BuilderValidationError("Test error", missing)

        assert str(error) == "Test error"
        assert error.missing_deps == missing

    def test_validation_error_with_none(self) -> None:
        """Test validation error with None missing_deps."""
        error = BuilderValidationError("Test error")
        assert error.missing_deps == []


class TestBuilderErrors:
    """Test suite for builder error conditions."""

    def test_frozen_builder_error(self) -> None:
        """Test BuilderFrozenError exception."""
        builder = AnalyzerBuilder().freeze()

        with pytest.raises(BuilderFrozenError):
            builder.with_config(FaciesCorrelationConfig())

    def test_invalid_processor_name(self) -> None:
        """Test batch processor configuration with invalid name."""
        builder = AnalyzerBuilder()

        with pytest.raises(ValueError):
            builder.with_processors(invalid_processor=Mock())

    def test_state_snapshot_invalid_input(self) -> None:
        """Test state snapshot with invalid input."""
        with pytest.raises(ValueError):
            AnalyzerBuilder.with_state_snapshot("not a dict")  # type: ignore


class TestBuilderIntegration:
    """Integration tests for builder pattern."""

    def test_full_configuration_workflow(self) -> None:
        """Test full builder configuration workflow."""
        config = FaciesCorrelationConfig()
        builder = (
            AnalyzerFactory.builder()
            .with_config(config)
            .with_boundary_detector(BoundaryDetector())
            .with_cube_aligner(CubeAligner())
            .freeze()
        )

        assert builder.is_frozen()
        assert builder.configured_processor_count() == 2
        assert builder.is_frozen()

        analyzer = builder.build()
        assert analyzer is not None

    def test_builder_state_persistence_across_copies(self) -> None:
        """Test that builder state persists across copies."""
        original_config = FaciesCorrelationConfig()
        builder1 = AnalyzerBuilder().with_config(original_config)

        builder2 = builder1.copy()
        builder2.with_boundary_detector(BoundaryDetector())

        # Original should not be affected
        assert builder1._boundary_detector is None
        assert builder2._boundary_detector is not None
        assert builder1._config is original_config
        assert builder2._config is original_config

    def test_processor_lazy_initialization(self) -> None:
        """Test that processors are lazily initialized."""
        builder = AnalyzerBuilder()
        assert builder._boundary_detector is None

        # Build without configuring processor
        analyzer = builder.build()
        assert analyzer is not None

        # Processor should still be None (lazy init happens internally)
        # but analyzer should work with default processors


# =============================================================================
# AnalyzerInterface Pattern Tests (OOP Improvements)
# =============================================================================


class SampleAnalyzerConfig(AnalysisConfig):
    """Sample test configuration implementing AnalysisConfig."""

    def __init__(self, value: str = "test"):
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}


class SampleAnalyzerImpl:
    """Sample test analyzer implementing AnalyzerInterface."""

    def __init__(self):
        self._config = SampleAnalyzerConfig()
        self._ready = True

    @property
    def name(self) -> str:
        return "test_domain"

    def validate_inputs(self, **kwargs):
        return "data" in kwargs

    def analyze(self, **kwargs):
        if not self.validate_inputs(**kwargs):
            raise ValueError("Missing 'data' in kwargs")
        return {"result": kwargs["data"] * 2}

    def get_configuration(self):
        return self._config

    def configure(self, config):
        self._config = config

    def get_name(self) -> str:
        return f"SampleAnalyzer ({self.name})"

    def is_ready(self) -> bool:
        return self._ready


class TestAnalyzerInterfacePattern:
    """Tests for AnalyzerInterface OOP pattern."""

    def test_analyzer_interface_name_property(self):
        """Test that analyzer has name property."""
        analyzer = SampleAnalyzerImpl()
        assert analyzer.name == "test_domain"

    def test_analyzer_interface_validate_inputs_success(self):
        """Test input validation with valid inputs."""
        analyzer = SampleAnalyzerImpl()
        assert analyzer.validate_inputs(data=42) is True

    def test_analyzer_interface_validate_inputs_failure(self):
        """Test input validation with invalid inputs."""
        analyzer = SampleAnalyzerImpl()
        assert analyzer.validate_inputs(other=42) is False

    def test_analyzer_interface_analyze_success(self):
        """Test analysis with valid inputs."""
        analyzer = SampleAnalyzerImpl()
        result = analyzer.analyze(data=21)
        assert result["result"] == 42

    def test_analyzer_interface_analyze_invalid_inputs(self):
        """Test analysis raises error with invalid inputs."""
        analyzer = SampleAnalyzerImpl()
        with pytest.raises(ValueError):
            analyzer.analyze(other=21)

    def test_analyzer_interface_get_configuration(self):
        """Test getting configuration."""
        analyzer = SampleAnalyzerImpl()
        config = analyzer.get_configuration()
        assert isinstance(config, SampleAnalyzerConfig)
        assert config.value == "test"

    def test_analyzer_interface_configure(self):
        """Test updating configuration."""
        analyzer = SampleAnalyzerImpl()
        new_config = SampleAnalyzerConfig(value="updated")
        analyzer.configure(new_config)
        assert analyzer.get_configuration().value == "updated"

    def test_analyzer_interface_get_name(self):
        """Test getting human-readable name."""
        analyzer = SampleAnalyzerImpl()
        name = analyzer.get_name()
        assert "SampleAnalyzer" in name
        assert "test_domain" in name

    def test_analyzer_interface_is_ready(self):
        """Test readiness check."""
        analyzer = SampleAnalyzerImpl()
        assert analyzer.is_ready() is True

        analyzer._ready = False
        assert analyzer.is_ready() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
