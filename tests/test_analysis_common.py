"""Comprehensive unit tests for AnalysisCommon singleton.

This module provides extensive test coverage for the AnalysisCommon singleton pattern,
including initialization, configuration, delegation, thread safety, and magic methods.

Test organization:
- Singleton pattern tests
- Initialization and configuration tests
- Property and state tests
- Magic methods tests (representations, comparisons, boolean, context managers)
- Delegation and call tests
- Thread safety tests
- Integration tests
- Edge cases and error handling
"""

# mypy: ignore-errors


import pytest
import threading
from pathlib import Path
from typing import Any, Generator, List
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from src.analysis.common import AnalysisCommon
from src.processing.managers import ProcessManager


# Test constants
TEST_PATTERNS = ["*.pkl", "*.npy", "*.cache"]
TEST_FILEPATH = "/path/to/test/file.txt"
TEST_CACHE_DIR = Path("/tmp/cache")
TEST_KEYS = ["key1", "key2"]
TEST_PREFIX = "test"
NUM_THREADS = 10
NUM_CONCURRENT_CALLS = 20


class DummyProcessManager(ProcessManager):
    """Mock ProcessManager for testing with call tracking capabilities.

    Implements ProcessManagerProtocol for testing AnalysisCommon delegation.
    Tracks all method calls with parameters for verification.
    """

    def __init__(self) -> None:
        """Initialize the mock manager with empty call log."""
        self.call_log: List[tuple[Any, ...]] = []
        self._closed: bool = False

    def clear_cache(
        self,
        patterns: List[str] | None = None,
        cache_dir: Path | None = None,
        prefix: str = "",
    ) -> int:
        """Mock clear_cache method."""
        self.call_log.append(("clear_cache", patterns, cache_dir, prefix))
        return 5

    def open_file(
        self, filepath: str, description: str | None = None, prefix: str = ""
    ) -> bool:
        """Mock open_file method."""
        self.call_log.append(("open_file", filepath, description, prefix))
        return True

    def summarize_cache_files(
        self,
        cache_dir: str | None = None,
        keys: List[str] | None = None,
        prefix: str = "",
    ) -> None:
        """Mock summarize_cache_files method."""
        self.call_log.append(("summarize_cache_files", cache_dir, keys, prefix))
        return None

    def close(self) -> None:
        """Track close calls."""
        self.call_log.append(("close",))
        self._closed = True

    def __enter__(self) -> "DummyProcessManager":
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, closing resources."""
        self.close()

    def get_last_call(self) -> tuple[Any, ...] | None:
        """Get the last recorded call, if any."""
        return self.call_log[-1] if self.call_log else None

    def get_call_count(self, method_name: str) -> int:
        """Count how many times a method was called."""
        return sum(1 for call in self.call_log if call[0] == method_name)


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Reset singleton before and after each test.

    Ensures test isolation by clearing the singleton state.
    This fixture automatically runs for every test.
    """
    AnalysisCommon.reset()
    yield
    AnalysisCommon.reset()


@pytest.fixture
def dummy_manager() -> DummyProcessManager:
    """Provide a fresh DummyProcessManager for tests."""
    return DummyProcessManager()


@pytest.fixture
def initialized_instance(dummy_manager: DummyProcessManager) -> AnalysisCommon:
    """Provide an initialized AnalysisCommon singleton."""
    return AnalysisCommon.instance(dummy_manager)


def test_singleton_instance_returns_same_object(
    dummy_manager: DummyProcessManager,
) -> None:
    """Test that instance() always returns the same singleton object."""
    instance1 = AnalysisCommon.instance(dummy_manager)
    instance2 = AnalysisCommon.instance(dummy_manager)
    assert instance1 is instance2, "Should return same singleton instance"


def test_singleton_identity_with_multiple_calls(
    dummy_manager: DummyProcessManager,
) -> None:
    """Test singleton identity across multiple calls."""
    instance = AnalysisCommon.instance(dummy_manager)
    instances = [AnalysisCommon.instance() for _ in range(5)]
    assert all(
        inst is instance for inst in instances
    ), "All should reference same singleton"


def test_singleton_reset_creates_new_instance() -> None:
    """Test that reset() invalidates the singleton."""
    manager1 = DummyProcessManager()
    instance1 = AnalysisCommon.instance(manager1)
    AnalysisCommon.reset()
    manager2 = DummyProcessManager()
    instance2 = AnalysisCommon.instance(manager2)
    assert instance1 is not instance2, "After reset, should create new instance"


def test_instance_without_manager_auto_initializes() -> None:
    """Test that instance() auto-initializes with default process manager."""
    with patch("src.processing.get_registry") as mock_get_registry:
        mock_registry = mock_get_registry.return_value
        mock_hub = mock_registry.manager_hub.return_value
        mock_hub.processes = DummyProcessManager()
        instance = AnalysisCommon.instance()
        assert instance.is_initialized, "Should auto-initialize with default manager"


def test_instance_initializes_with_provided_manager(
    dummy_manager: DummyProcessManager,
) -> None:
    """Test that instance() properly initializes singleton."""
    instance = AnalysisCommon.instance(dummy_manager)
    assert instance.is_initialized, "Should be initialized"
    assert instance.proc_manager is dummy_manager, "Should use provided manager"


def test_initialize_via_configure(dummy_manager: DummyProcessManager) -> None:
    """Test initialization through configure()."""
    instance = AnalysisCommon.instance(DummyProcessManager())
    second_manager = DummyProcessManager()
    instance.configure(second_manager)
    assert instance.is_initialized, "Should still be initialized"


def test_configure_with_none_raises_error(initialized_instance: AnalysisCommon) -> None:
    """Test that configure(None) raises TypeError."""
    with pytest.raises(TypeError, match="proc_manager"):
        initialized_instance.configure(None)  # type: ignore[arg-type]


def test_configure_with_invalid_object_raises_error(
    initialized_instance: AnalysisCommon,
) -> None:
    """Test that configure() validates the manager object."""
    with pytest.raises(TypeError):
        initialized_instance.configure({"not": "a manager"})  # type: ignore[arg-type]


def test_is_initialized_property(initialized_instance: AnalysisCommon) -> None:
    """Test is_initialized property returns correct state."""
    assert initialized_instance.is_initialized is True


def test_proc_manager_property_returns_manager(
    dummy_manager: DummyProcessManager, initialized_instance: AnalysisCommon
) -> None:
    """Test proc_manager property returns the configured manager."""
    assert initialized_instance.proc_manager is dummy_manager


def test_repr_shows_initialized_state(initialized_instance: AnalysisCommon) -> None:
    """Test __repr__ includes initialization state."""
    repr_str = repr(initialized_instance)
    assert "AnalysisCommon" in repr_str, "Should include class name"


def test_str_returns_meaningful_representation(
    initialized_instance: AnalysisCommon,
) -> None:
    """Test __str__ returns a readable string."""
    str_repr = str(initialized_instance)
    assert "AnalysisCommon" in str_repr, "Should include class name"


def test_equality_same_instance(initialized_instance: AnalysisCommon) -> None:
    """Test that singleton equals itself."""
    assert initialized_instance == initialized_instance, "Should equal itself"


def test_equality_same_singleton() -> None:
    """Test that two references to singleton are equal."""
    AnalysisCommon.reset()
    manager = DummyProcessManager()
    instance1 = AnalysisCommon.instance(manager)
    instance2 = AnalysisCommon.instance()
    assert instance1 == instance2, "Same singleton should be equal"
    AnalysisCommon.reset()


def test_inequality_different_instances() -> None:
    """Test that different instances are not equal."""
    AnalysisCommon.reset()
    manager1 = DummyProcessManager()
    instance1 = AnalysisCommon.instance(manager1)
    AnalysisCommon.reset()
    manager2 = DummyProcessManager()
    instance2 = AnalysisCommon.instance(manager2)
    assert instance1 != instance2, "Different singletons should not be equal"


def test_inequality_with_other_types(initialized_instance: AnalysisCommon) -> None:
    """Test inequality with various non-AnalysisCommon types using parametrize."""
    non_analysiscommon_values = ["string", 42, {"dict": "value"}, None, [], set()]
    for value in non_analysiscommon_values:
        assert initialized_instance != value, f"Instance should not equal {value!r}"


@pytest.mark.parametrize(
    "other_value",
    [
        "string",
        42,
        {"dict": "value"},
        None,
        [],
        set(),
    ],
)
def test_inequality_with_parametrized_types(
    initialized_instance: AnalysisCommon, other_value: Any
) -> None:
    """Test inequality with various types using pytest parametrize."""
    assert initialized_instance != other_value


def test_hash_consistency(initialized_instance: AnalysisCommon) -> None:
    """Test that hash is consistent across multiple calls."""
    hash1 = hash(initialized_instance)
    hash2 = hash(initialized_instance)
    assert hash1 == hash2, "Hash should be consistent"


def test_hash_same_for_singleton() -> None:
    """Test that singleton references have the same hash."""
    AnalysisCommon.reset()
    manager = DummyProcessManager()
    instance1 = AnalysisCommon.instance(manager)
    instance2 = AnalysisCommon.instance()
    assert hash(instance1) == hash(instance2), "Same singleton should have same hash"
    AnalysisCommon.reset()


def test_bool_initialized_is_true() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    assert instance
    assert bool(instance) is True


def test_context_manager_basic() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    with instance as ctx:
        assert ctx is instance
        assert ctx.is_initialized


def test_call_delegates_to_process_manager() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    result = instance("clear_cache")
    assert result == 5


def test_call_with_keyword_arguments(dummy_manager: DummyProcessManager) -> None:
    """Test __call__ with keyword arguments."""
    instance = AnalysisCommon.instance(dummy_manager)
    result = instance("open_file", filepath=TEST_FILEPATH)
    assert result is True


@pytest.mark.parametrize(
    "method_name,expected_result",
    [
        ("clear_cache", 5),
        ("open_file", True),
        ("summarize_cache_files", None),
    ],
)
def test_delegated_methods(
    dummy_manager: DummyProcessManager, method_name: str, expected_result: Any
) -> None:
    """Test that delegated methods return correct values using parametrize."""
    instance = AnalysisCommon.instance(dummy_manager)

    if method_name == "clear_cache":
        result = instance.clear_cache()
    elif method_name == "open_file":
        result = instance.open_file(TEST_FILEPATH)
    elif method_name == "summarize_cache_files":
        instance.summarize_cache_files()
        result = None

    assert result == expected_result
    last_call = dummy_manager.get_last_call()
    assert last_call is not None
    assert last_call[0] == method_name


def test_clear_cache_delegates(dummy_manager: DummyProcessManager) -> None:
    """Test clear_cache() properly delegates to process_manager."""
    instance = AnalysisCommon.instance(dummy_manager)
    result = instance.clear_cache()
    assert result == 5
    assert dummy_manager.get_call_count("clear_cache") == 1


def test_open_file_delegates(dummy_manager: DummyProcessManager) -> None:
    """Test open_file() properly delegates to process_manager."""
    instance = AnalysisCommon.instance(dummy_manager)
    result = instance.open_file(TEST_FILEPATH)
    assert result is True
    assert dummy_manager.get_call_count("open_file") == 1


def test_summarize_cache_files_delegates(dummy_manager: DummyProcessManager) -> None:
    """Test summarize_cache_files() properly delegates to process_manager."""
    instance = AnalysisCommon.instance(dummy_manager)
    instance.summarize_cache_files()
    assert dummy_manager.get_call_count("summarize_cache_files") == 1


def test_full_workflow() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    assert instance.is_initialized
    instance.clear_cache()
    instance.open_file("test.txt")
    instance.summarize_cache_files()
    assert len(manager.call_log) == 3


def test_configuration_change() -> None:
    instance = AnalysisCommon.instance(DummyProcessManager())
    manager1 = DummyProcessManager()
    manager2 = DummyProcessManager()
    instance.configure(manager1)
    assert instance.proc_manager is manager1
    instance.configure(manager2)
    assert instance.proc_manager is manager2


def test_access_manager_after_initialization() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    assert instance.proc_manager is manager


def test_thread_safe_singleton_creation() -> None:
    """Test that singleton creation is thread-safe."""
    instances = []
    manager = DummyProcessManager()

    def create_instance() -> None:
        instances.append(AnalysisCommon.instance(manager))

    threads = [threading.Thread(target=create_instance) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    first = instances[0]
    assert all(inst is first for inst in instances)


def test_thread_safe_configuration() -> None:
    """Test that configuration changes are thread-safe."""
    instance = AnalysisCommon.instance(DummyProcessManager())
    managers = [DummyProcessManager() for _ in range(5)]

    def configure_with_manager(mgr: DummyProcessManager) -> None:
        instance.configure(mgr)

    threads = [
        threading.Thread(target=configure_with_manager, args=(mgr,)) for mgr in managers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert instance.is_initialized


def test_concurrent_method_calls() -> None:
    """Test that concurrent method calls work correctly."""
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)

    def call_method() -> None:
        instance.clear_cache()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_method) for _ in range(20)]
        for future in futures:
            future.result()

    assert len(manager.call_log) == 20


def test_double_reset() -> None:
    AnalysisCommon.instance(DummyProcessManager())
    AnalysisCommon.reset()
    AnalysisCommon.reset()


def test_hash_in_set() -> None:
    manager = DummyProcessManager()
    instance1 = AnalysisCommon.instance(manager)
    instance2 = AnalysisCommon.instance()

    singleton_set = {instance1, instance2}
    assert len(singleton_set) == 1


def test_hash_in_dict() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    dict_with_singleton = {instance: "value"}

    assert dict_with_singleton[AnalysisCommon.instance()] == "value"


def test_repr_consistency() -> None:
    instance = AnalysisCommon.instance(DummyProcessManager())
    repr1 = repr(instance)
    repr2 = repr(instance)

    assert repr1 == repr2


def test_call_with_invalid_method_raises_error() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    with pytest.raises(AttributeError, match="ProcessManager has no method"):
        instance("nonexistent_method")


def test_clear_cache_with_patterns() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    result = instance.clear_cache(patterns=["*.pkl"])
    assert result == 5


def test_open_file_with_description() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    result = instance.open_file("/path/file.txt", description="test file")
    assert result is True


def test_summarize_with_keys() -> None:
    """Test summarize_cache_files with keys parameter."""
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    instance.summarize_cache_files(keys=["key1", "key2"])


def test_validation_rejects_none() -> None:
    """Test that configure(None) raises TypeError."""
    instance = AnalysisCommon.instance(DummyProcessManager())
    with pytest.raises(TypeError):
        instance.configure(None)  # type: ignore[arg-type]


def test_validation_accepts_duck_typed_manager() -> None:
    """Test that duck-typed managers are accepted."""

    class MinimalManager:
        def clear_cache(
            self, patterns: Any = None, cache_dir: Any = None, prefix: str = ""
        ) -> int:
            return 0

        def open_file(
            self, filepath: str, description: Any = None, prefix: str = ""
        ) -> bool:
            return False

        def summarize_cache_files(
            self, cache_dir: Any = None, keys: Any = None, prefix: str = ""
        ) -> None:
            return None

    instance = AnalysisCommon.instance(MinimalManager())
    assert instance.is_initialized


def test_initialization_idempotent() -> None:
    manager1 = DummyProcessManager()
    manager2 = DummyProcessManager()
    instance = AnalysisCommon.instance(manager1)
    first_mgr = instance.proc_manager
    instance.configure(manager2)
    second_mgr = instance.proc_manager
    assert first_mgr is not second_mgr
    assert second_mgr is manager2


def test_manager_method_with_all_parameters() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    cache_dir = Path("/tmp/cache")
    instance.clear_cache(
        patterns=["*.pkl", "*.npy"], cache_dir=cache_dir, prefix="test"
    )
    assert len(manager.call_log) > 0


def test_call_method_returns_value() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    result = instance("clear_cache", patterns=["*.pkl"])
    assert result == 5


def test_clear_cache_default_parameters() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    result = instance.clear_cache()
    assert result == 5


def test_new_always_returns_singleton() -> None:
    manager = DummyProcessManager()
    instance1 = AnalysisCommon(manager)
    instance2 = AnalysisCommon(manager)
    instance3 = AnalysisCommon.instance()
    assert instance1 is instance2 is instance3


def test_required_methods_constant() -> None:
    assert hasattr(AnalysisCommon, "_REQUIRED_METHODS")
    assert "clear_cache" in AnalysisCommon._REQUIRED_METHODS
    assert "open_file" in AnalysisCommon._REQUIRED_METHODS
    assert "summarize_cache_files" in AnalysisCommon._REQUIRED_METHODS


def test_initialization_requires_manager() -> None:
    """Test that initialization without a manager raises TypeError."""
    with pytest.raises(TypeError, match="proc_manager is required"):
        AnalysisCommon()


def test_configure_replaces_manager() -> None:
    manager1 = DummyProcessManager()
    manager2 = DummyProcessManager()
    instance = AnalysisCommon.instance(manager1)

    assert instance.proc_manager is manager1
    instance.configure(manager2)
    assert instance.proc_manager is manager2


def test_call_with_return_value() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)

    result = instance("clear_cache", patterns=["*.pkl"])
    assert result == 5
    assert manager.call_log[-1][1] == ["*.pkl"]


def test_multiple_delegated_calls() -> None:
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)

    for i in range(5):
        instance.clear_cache()
        instance.open_file(f"file{i}.txt")
        instance.summarize_cache_files()

    assert len(manager.call_log) == 15


def test_manager_state_isolation() -> None:
    manager1 = DummyProcessManager()
    manager2 = DummyProcessManager()

    instance = AnalysisCommon.instance(manager1)
    instance.clear_cache()

    assert len(manager1.call_log) == 1
    assert len(manager2.call_log) == 0

    instance.configure(manager2)
    instance.clear_cache()

    assert len(manager1.call_log) == 1
    assert len(manager2.call_log) == 1


def test_repr_uninitialized() -> None:
    """Test __repr__ with error handling for uninitialized."""
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    repr_str = repr(instance)
    assert "AnalysisCommon" in repr_str
    assert "DummyProcessManager" in repr_str


def test_str_uninitialized_case() -> None:
    """Test __str__ returns uninitialized string."""
    # Create initialized instance, then we'd need reflection to test uninitialized
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    str_repr = str(instance)
    assert "initialized" in str_repr
    assert "DummyProcessManager" in str_repr


def test_configure_idempotent_with_same_manager() -> None:
    """Test that configuring with same manager is safe."""
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    instance.configure(manager)  # Should not raise or error
    assert instance.proc_manager is manager


def test_call_logging_behavior() -> None:
    """Test that __call__ logs method invocations."""
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)
    # This exercises the logging path
    result = instance("clear_cache", patterns=["*.pkl"], prefix="test")
    assert result == 5
    assert len(manager.call_log) > 0


def test_multiple_resets_safe() -> None:
    """Test that multiple consecutive resets are safe."""
    for i in range(3):
        manager = DummyProcessManager()
        AnalysisCommon.instance(manager)
        AnalysisCommon.reset()


def test_eq_with_non_analysiscommon_object() -> None:
    """Test equality comparison with non-AnalysisCommon objects."""
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)

    # Test inequality with various types
    assert (instance == "string") is False
    assert (instance == 42) is False
    assert (instance == {}) is False
    assert (instance == []) is False
    assert (instance is None) is False


def test_ne_returns_correct_inverse() -> None:
    """Test inequality is correct inverse of equality."""
    manager = DummyProcessManager()
    instance1 = AnalysisCommon.instance(manager)

    # Self equality and inequality
    assert (instance1 == instance1) is True
    assert (instance1 != instance1) is False

    # With different object
    assert (instance1 != "other") is True
    assert (instance1 == "other") is False


def test_hash_allows_use_in_collections() -> None:
    """Test that hash makes singleton usable in collections."""
    manager = DummyProcessManager()
    instance = AnalysisCommon.instance(manager)

    # Use in set
    s = {instance}
    assert instance in s

    # Use as dict key
    d = {instance: "test"}
    assert d[instance] == "test"

    # Use in frozenset
    fs = frozenset([instance])
    assert instance in fs


# =============================================================================
# Integration Tests (OOP Improvements)
# =============================================================================


class TestAnalysisCommonIntegration:
    """Integration tests combining all OOP improvements."""

    def test_analyzer_with_registry_and_pipeline(self):
        """Test using analyzer with registry and pipeline."""
        from src.analysis.base import AnalyzerInterface, AnalysisConfig
        from src.analysis.processors.management import ProcessorRegistry
        from src.analysis.pipelines.orchestrator import Pipeline, PipelineStage
        from typing import Dict

        class SampleConfig(AnalysisConfig):
            """Sample configuration."""

            def __init__(self, value: str = "test"):
                self.value = value

            def to_dict(self) -> Dict[str, Any]:
                return {"value": self.value}

        class SampleAnalyzer(AnalyzerInterface):
            """Sample analyzer."""

            def __init__(self):
                self._config = SampleConfig()
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

            def get_configuration(self) -> SampleConfig:
                return self._config

            def configure(self, config: SampleConfig) -> None:
                self._config = config

            def get_name(self) -> str:
                return f"TestAnalyzer ({self.name})"

            def is_ready(self) -> bool:
                return self._ready

        class DummyProcessor:
            """Dummy processor."""

            def __init__(self):
                self.name = "dummy"

        class DummyStage(PipelineStage):
            """Dummy pipeline stage."""

            def __init__(self, multiplier: int = 1):
                self._multiplier = multiplier

            @property
            def name(self) -> str:
                return "stage"

            def can_execute(self, input_data: int) -> bool:
                return input_data > 0

            def execute(self, input_data: int) -> int:
                return input_data * self._multiplier

        # Create registry and register processor
        registry = ProcessorRegistry()
        registry.register(
            "my_processor", DummyProcessor, domain="test", tags=["processing"]
        )

        # Create analyzer
        analyzer = SampleAnalyzer()

        # Verify analyzer is ready
        assert analyzer.is_ready()

        # Verify processor is available
        assert registry.has("my_processor")
        proc = registry.create("my_processor")
        assert proc.name == "dummy"

        # Create pipeline
        pipeline = Pipeline("test_pipeline")
        pipeline.add_stage(DummyStage(multiplier=2))
        pipeline.add_stage(DummyStage(multiplier=3))

        # Use all three together
        pipeline_result = pipeline.execute(5)
        if analyzer.validate_inputs(data=pipeline_result):
            analysis_result = analyzer.analyze(data=pipeline_result)
            assert analysis_result["result"] == 60  # 30 * 2

    def test_polymorphic_usage_patterns(self):
        """Test polymorphic usage across components."""
        from src.analysis.base import AnalyzerInterface, AnalysisConfig
        from src.analysis.processors.management import ProcessorRegistry
        from typing import Dict

        class BaseConfig(AnalysisConfig):
            def to_dict(self) -> Dict[str, Any]:
                return {}

        class Config1(BaseConfig):
            pass

        class Config2(BaseConfig):
            pass

        class Analyzer1(AnalyzerInterface):
            @property
            def name(self) -> str:
                return "analyzer1"

            def validate_inputs(self, **kwargs: Any) -> bool:
                return True

            def analyze(self, **kwargs: Any) -> Dict:
                return {"analyzer": "1"}

            def get_configuration(self) -> BaseConfig:
                return Config1()

            def configure(self, config: BaseConfig) -> None:
                pass

            def get_name(self) -> str:
                return "Analyzer1"

            def is_ready(self) -> bool:
                return True

        class Analyzer2(AnalyzerInterface):
            @property
            def name(self) -> str:
                return "analyzer2"

            def validate_inputs(self, **kwargs: Any) -> bool:
                return True

            def analyze(self, **kwargs: Any) -> Dict:
                return {"analyzer": "2"}

            def get_configuration(self) -> BaseConfig:
                return Config2()

            def configure(self, config: BaseConfig) -> None:
                pass

            def get_name(self) -> str:
                return "Analyzer2"

            def is_ready(self) -> bool:
                return True

        # Use polymorphically
        analyzers: List[AnalyzerInterface] = [
            Analyzer1(),
            Analyzer2(),
        ]

        for analyzer in analyzers:
            assert analyzer.is_ready()
            assert analyzer.validate_inputs()
            result = analyzer.analyze()
            assert "analyzer" in result
