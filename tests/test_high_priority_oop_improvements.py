"""Tests for High Priority OOP Improvements.

Tests for:
1. AnalyzerInterface
2. ProcessorRegistry
3. Pipeline Orchestrator
"""

import pytest
from typing import Dict, Any

from src.analysis.base import (
    AnalyzerInterface,
    AnalysisConfig,
)
from src.analysis.processors.registry import (
    ProcessorRegistry,
    ProcessorMetadata,
)
from src.analysis.pipelines.orchestrator import (
    Pipeline,
    PipelineStage,
    StageResult,
    ConditionalStage,
)


# =============================================================================
# Fixtures and Test Helpers
# =============================================================================


class TestConfig(AnalysisConfig):
    """Test configuration."""

    def __init__(self, value: str = "test"):
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}


class TestAnalyzer(AnalyzerInterface[TestConfig, Dict]):
    """Test analyzer implementing AnalyzerInterface."""

    def __init__(self):
        self._config = TestConfig()
        self._ready = True

    @property
    def name(self) -> str:
        return "test_domain"

    def validate_inputs(self, **kwargs: Any) -> bool:
        return "data" in kwargs

    def analyze(self, **kwargs: Any) -> Dict:
        if not self.validate_inputs(**kwargs):
            raise ValueError("Missing 'data' in kwargs")
        return {"result": kwargs["data"] * 2}

    def get_configuration(self) -> TestConfig:
        return self._config

    def configure(self, config: TestConfig) -> None:
        self._config = config

    def get_name(self) -> str:
        return f"TestAnalyzer ({self.name})"

    def is_ready(self) -> bool:
        return self._ready


class DummyStage(PipelineStage[int, int]):
    """Dummy stage for testing."""

    def __init__(self, name: str = "dummy", multiplier: int = 1):
        self._name = name
        self._multiplier = multiplier

    @property
    def name(self) -> str:
        return self._name

    def can_execute(self, input_data: int) -> bool:
        return input_data > 0

    def execute(self, input_data: int) -> int:
        return input_data * self._multiplier


class FailingStage(PipelineStage[int, int]):
    """Stage that always fails for testing error handling."""

    @property
    def name(self) -> str:
        return "failing_stage"

    def can_execute(self, input_data: int) -> bool:
        return True

    def execute(self, input_data: int) -> int:
        raise RuntimeError("Intentional failure for testing")


# =============================================================================
# AnalyzerInterface Tests
# =============================================================================


class TestAnalyzerInterface:
    """Tests for AnalyzerInterface."""

    def test_analyzer_has_name_property(self):
        """Test that analyzer has name property."""
        analyzer = TestAnalyzer()
        assert analyzer.name == "test_domain"

    def test_analyzer_validate_inputs_success(self):
        """Test input validation with valid inputs."""
        analyzer = TestAnalyzer()
        assert analyzer.validate_inputs(data=42) is True

    def test_analyzer_validate_inputs_failure(self):
        """Test input validation with invalid inputs."""
        analyzer = TestAnalyzer()
        assert analyzer.validate_inputs(other=42) is False

    def test_analyzer_analyze_success(self):
        """Test analysis with valid inputs."""
        analyzer = TestAnalyzer()
        result = analyzer.analyze(data=21)
        assert result["result"] == 42

    def test_analyzer_analyze_invalid_inputs(self):
        """Test analysis raises error with invalid inputs."""
        analyzer = TestAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze(other=21)

    def test_analyzer_get_configuration(self):
        """Test getting configuration."""
        analyzer = TestAnalyzer()
        config = analyzer.get_configuration()
        assert isinstance(config, TestConfig)
        assert config.value == "test"

    def test_analyzer_configure(self):
        """Test updating configuration."""
        analyzer = TestAnalyzer()
        new_config = TestConfig(value="updated")
        analyzer.configure(new_config)
        assert analyzer.get_configuration().value == "updated"

    def test_analyzer_get_name(self):
        """Test getting human-readable name."""
        analyzer = TestAnalyzer()
        name = analyzer.get_name()
        assert "TestAnalyzer" in name
        assert "test_domain" in name

    def test_analyzer_is_ready(self):
        """Test readiness check."""
        analyzer = TestAnalyzer()
        assert analyzer.is_ready() is True

        analyzer._ready = False
        assert analyzer.is_ready() is False

    def test_polymorphic_usage(self):
        """Test that analyzers can be used polymorphically."""
        analyzers: list[AnalyzerInterface] = [
            TestAnalyzer(),
            TestAnalyzer(),
        ]

        for analyzer in analyzers:
            assert analyzer.is_ready()
            assert analyzer.validate_inputs(data=10)
            result = analyzer.analyze(data=10)
            assert result["result"] == 20


# =============================================================================
# ProcessorRegistry Tests
# =============================================================================


class DummyProcessor:
    """Dummy processor for testing."""

    def __init__(self):
        self.name = "dummy"


class TestProcessorRegistry:
    """Tests for ProcessorRegistry."""

    def test_register_processor(self):
        """Test registering a processor."""
        registry = ProcessorRegistry()
        registry.register("test_proc", lambda: DummyProcessor())
        assert registry.has("test_proc")

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate processor raises error."""
        registry = ProcessorRegistry()
        registry.register("test_proc", lambda: DummyProcessor())
        with pytest.raises(ValueError):
            registry.register("test_proc", lambda: DummyProcessor())

    def test_register_non_callable_raises_error(self):
        """Test that non-callable factory raises error."""
        registry = ProcessorRegistry()
        with pytest.raises(TypeError):
            registry.register("test_proc", "not_callable")

    def test_create_processor(self):
        """Test creating a processor instance."""
        registry = ProcessorRegistry()
        registry.register("test_proc", DummyProcessor)
        proc = registry.create("test_proc")
        assert isinstance(proc, DummyProcessor)

    def test_create_unknown_processor_raises_error(self):
        """Test that creating unknown processor raises error."""
        registry = ProcessorRegistry()
        with pytest.raises(ValueError):
            registry.create("unknown_proc")

    def test_create_all_processors(self):
        """Test creating multiple processors."""
        registry = ProcessorRegistry()
        registry.register("proc1", DummyProcessor)
        registry.register("proc2", DummyProcessor)
        procs = registry.create_all(["proc1", "proc2"])
        assert len(procs) == 2
        assert "proc1" in procs
        assert "proc2" in procs

    def test_list_processors_all(self):
        """Test listing all processors."""
        registry = ProcessorRegistry()
        registry.register("proc1", DummyProcessor)
        registry.register("proc2", DummyProcessor)
        procs = registry.list_processors()
        assert len(procs) == 2
        assert "proc1" in procs
        assert "proc2" in procs

    def test_list_processors_by_domain(self):
        """Test listing processors by domain."""
        registry = ProcessorRegistry()
        registry.register("facies_proc", DummyProcessor, domain="facies")
        registry.register("physics_proc", DummyProcessor, domain="physics")

        facies = registry.list_processors(domain="facies")
        assert "facies_proc" in facies
        assert "physics_proc" not in facies

    def test_list_processors_by_tags(self):
        """Test listing processors by tags."""
        registry = ProcessorRegistry()
        registry.register("proc1", DummyProcessor, tags=["detection", "boundary"])
        registry.register("proc2", DummyProcessor, tags=["detection", "advanced"])
        registry.register("proc3", DummyProcessor, tags=["analysis"])

        # Find processors with 'detection' tag
        detection = registry.list_processors(tags=["detection"])
        assert "proc1" in detection
        assert "proc2" in detection
        assert "proc3" not in detection

        # Find processors with 'boundary' tag
        boundary = registry.list_processors(tags=["boundary"])
        assert "proc1" in boundary
        assert "proc2" not in boundary

    def test_list_processors_by_version(self):
        """Test listing processors by version."""
        registry = ProcessorRegistry()
        registry.register("proc_v1", DummyProcessor, version="1.0")
        registry.register("proc_v2", DummyProcessor, version="2.0")

        v1 = registry.list_processors(version="1.0")
        assert "proc_v1" in v1
        assert "proc_v2" not in v1

    def test_get_metadata(self):
        """Test getting processor metadata."""
        registry = ProcessorRegistry()
        registry.register(
            "test_proc",
            DummyProcessor,
            domain="facies",
            version="1.5",
            tags=["test", "demo"],
            description="Test processor",
        )

        meta = registry.get_metadata("test_proc")
        assert meta.name == "test_proc"
        assert meta.domain == "facies"
        assert meta.version == "1.5"
        assert "test" in meta.tags
        assert meta.description == "Test processor"

    def test_get_metadata_unknown_raises_error(self):
        """Test getting metadata for unknown processor raises error."""
        registry = ProcessorRegistry()
        with pytest.raises(ValueError):
            registry.get_metadata("unknown")

    def test_unregister_processor(self):
        """Test unregistering a processor."""
        registry = ProcessorRegistry()
        registry.register("test_proc", DummyProcessor)
        assert registry.has("test_proc")

        result = registry.unregister("test_proc")
        assert result is True
        assert not registry.has("test_proc")

    def test_unregister_unknown_returns_false(self):
        """Test unregistering unknown processor returns False."""
        registry = ProcessorRegistry()
        result = registry.unregister("unknown")
        assert result is False


# =============================================================================
# Pipeline Orchestrator Tests
# =============================================================================


class TestPipeline:
    """Tests for Pipeline."""

    def test_pipeline_add_stage(self):
        """Test adding stages to pipeline."""
        pipeline = Pipeline("test")
        stage = DummyStage()
        result = pipeline.add_stage(stage)

        # Should return self for fluent API
        assert result is pipeline

    def test_pipeline_fluent_api(self):
        """Test fluent API for adding stages."""
        pipeline = (
            Pipeline("test")
            .add_stage(DummyStage("stage1"))
            .add_stage(DummyStage("stage2"))
            .add_stage(DummyStage("stage3"))
        )
        assert len(pipeline._stages) == 3

    def test_pipeline_execute_success(self):
        """Test pipeline executes stages successfully."""
        pipeline = Pipeline("test")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))
        pipeline.add_stage(DummyStage("stage2", multiplier=3))

        result = pipeline.execute(5)
        # 5 * 2 * 3 = 30
        assert result == 30

    def test_pipeline_skips_stage_if_cannot_execute(self):
        """Test pipeline skips stages that cannot execute."""
        pipeline = Pipeline("test")
        pipeline.add_stage(
            DummyStage("stage1", multiplier=2)
        )  # Can't execute if input <= 0
        pipeline.add_stage(DummyStage("stage2", multiplier=3))

        # With input -1, first stage can't execute (can_execute returns False)
        # So it's skipped, and pipeline passes -1 to stage2
        # stage2 also can't execute with -1, so it's skipped too
        # Pipeline completes with output -1
        result = pipeline.execute(-1)
        assert result == -1  # No stages executed

    def test_pipeline_handles_stage_failure(self):
        """Test pipeline handles stage failures."""
        pipeline = Pipeline("test")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))
        pipeline.add_stage(FailingStage())
        pipeline.add_stage(DummyStage("stage3", multiplier=3))

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.execute(5)

        assert "failing_stage" in str(exc_info.value)

    def test_pipeline_tracks_stage_results(self):
        """Test pipeline tracks results from each stage."""
        pipeline = Pipeline("test")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))
        pipeline.add_stage(DummyStage("stage2", multiplier=3))

        result = pipeline.execute(5)

        results = pipeline.get_all_results()
        assert "stage1" in results
        assert "stage2" in results
        assert results["stage1"].success is True
        assert results["stage2"].success is True

    def test_pipeline_get_stage_result(self):
        """Test retrieving individual stage result."""
        pipeline = Pipeline("test")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))
        pipeline.add_stage(DummyStage("stage2", multiplier=3))

        pipeline.execute(5)

        stage1_result = pipeline.get_stage_result("stage1")
        assert stage1_result is not None
        assert stage1_result.success is True
        assert stage1_result.output == 10

    def test_pipeline_get_stage_output(self):
        """Test retrieving stage output directly."""
        pipeline = Pipeline("test")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))
        pipeline.add_stage(DummyStage("stage2", multiplier=3))

        pipeline.execute(5)

        output = pipeline.get_stage_output("stage1")
        assert output == 10

    def test_pipeline_get_execution_summary(self):
        """Test getting human-readable execution summary."""
        pipeline = Pipeline("test")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))
        pipeline.add_stage(DummyStage("stage2", multiplier=3))

        pipeline.execute(5)

        summary = pipeline.get_execution_summary()
        assert "test" in summary
        assert "stage1" in summary
        assert "stage2" in summary

    def test_conditional_stage_executes_when_condition_true(self):
        """Test conditional stage executes when condition is true."""
        inner_stage = DummyStage("inner", multiplier=5)
        condition_stage = ConditionalStage(
            inner_stage=inner_stage,
            condition=lambda x: x > 10,
        )

        # Condition is true (15 > 10)
        result = condition_stage.execute(15)
        assert result == 75  # 15 * 5

    def test_conditional_stage_skips_when_condition_false(self):
        """Test conditional stage skips when condition is false."""
        pipeline = Pipeline("test")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))

        inner_stage = DummyStage("inner", multiplier=5)
        condition_stage = ConditionalStage(
            inner_stage=inner_stage,
            condition=lambda x: x > 100,
        )
        pipeline.add_stage(condition_stage)

        # Condition is false (10 < 100), so conditional stage should be skipped
        # But stage1 will multiply by 2: 5 * 2 = 10
        # Then conditional stage won't run because condition is false
        # Pipeline completes with output 10
        result = pipeline.execute(5)
        assert result == 10  # Only stage1 executed

    def test_stage_result_properties(self):
        """Test StageResult has correct properties."""
        stage = DummyStage()
        output = stage.execute(10)

        result = StageResult(
            stage_name="test_stage",
            success=True,
            output=output,
            duration_ms=123.45,
            metadata={"items": 5},
        )

        assert result.stage_name == "test_stage"
        assert result.success is True
        assert result.output == 10
        assert result.duration_ms == 123.45
        assert result.metadata["items"] == 5

    def test_stage_result_string_representation(self):
        """Test StageResult string representation."""
        result = StageResult(
            stage_name="test",
            success=True,
            output=42,
            duration_ms=99.9,
        )

        str_repr = str(result)
        assert "test" in str_repr
        assert "99.9" in str_repr


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining all three improvements."""

    def test_analyzer_with_registry_and_pipeline(self):
        """Test using analyzer with registry and pipeline."""
        # Create registry and register processor
        registry = ProcessorRegistry()
        registry.register(
            "my_processor", DummyProcessor, domain="test", tags=["processing"]
        )

        # Create analyzer
        analyzer = TestAnalyzer()

        # Verify analyzer is ready
        assert analyzer.is_ready()

        # Verify processor is available
        assert registry.has("my_processor")
        proc = registry.create("my_processor")
        assert proc.name == "dummy"

        # Create pipeline
        pipeline = Pipeline("test_pipeline")
        pipeline.add_stage(DummyStage("stage1", multiplier=2))
        pipeline.add_stage(DummyStage("stage2", multiplier=3))

        # Use all three together
        pipeline_result = pipeline.execute(5)
        if analyzer.validate_inputs(data=pipeline_result):
            analysis_result = analyzer.analyze(data=pipeline_result)
            assert analysis_result["result"] == 60  # 30 * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
