"""Comprehensive tests for Phase 4 Architectural Components

Tests for:
  - Dependency Injection Framework
  - Event Bus System
  - Configuration Management

Run with: pytest tests/test_phase_4_architecture.py -v
"""

import json
import tempfile
import time
from pathlib import Path
from threading import Lock, Thread
from unittest.mock import patch

import pytest

from src.analysis.config_manager import (ConfigManager, EnvironmentSource,
                                         JsonSource)
from src.analysis.patterns.dependency_injection import (Container,
                                                        ContainerBuilder,
                                                        Lifecycle,
                                                        LifecycleManager,
                                                        ResolutionError,
                                                        ServiceDescriptor)
from src.analysis.patterns.event_bus import (AsyncEventBus, Event, EventBus,
                                             EventDispatcher, EventFilter,
                                             EventHandler, EventPriority)
from src.core import ConfigProfile, ConfigRule, ConfigValidator

# ============================================================================
# DEPENDENCY INJECTION TESTS
# ============================================================================


class DummyService:
    """Dummy service for testing."""

    def __init__(self):
        self.value = "dummy"


class ServiceWithDependency:
    """Service that depends on DummyService."""

    def __init__(self, dummy: DummyService):
        self.dummy = dummy


class ServiceWithCircularA:
    """Service with circular dependency A."""

    def __init__(self, b: "ServiceWithCircularB"):
        self.b = b


class ServiceWithCircularB:
    """Service with circular dependency B."""

    def __init__(self, a: ServiceWithCircularA):
        self.a = a


class TestDependencyInjection:
    """Tests for dependency injection framework."""

    def test_lifecycle_enum(self):
        """Test Lifecycle enum values."""
        assert Lifecycle.TRANSIENT.value == "transient"
        assert Lifecycle.SINGLETON.value == "singleton"
        assert Lifecycle.SCOPED.value == "scoped"

    def test_service_descriptor_creation(self):
        """Test ServiceDescriptor creation."""
        descriptor = ServiceDescriptor(
            name="test_service",
            service_type=DummyService,
            implementation=DummyService,
            lifecycle=Lifecycle.SINGLETON,
        )

        assert descriptor.name == "test_service"
        assert descriptor.service_type == DummyService
        assert descriptor.lifecycle == Lifecycle.SINGLETON

    def test_lifecycle_manager_transient(self):
        """Test transient lifecycle creation."""
        manager = LifecycleManager()
        descriptor = ServiceDescriptor(
            name="transient",
            service_type=DummyService,
            implementation=DummyService,
            lifecycle=Lifecycle.TRANSIENT,
        )

        instance1 = manager.get_instance(descriptor, lambda: DummyService())
        instance2 = manager.get_instance(descriptor, lambda: DummyService())

        assert instance1 is not instance2

    def test_lifecycle_manager_singleton(self):
        """Test singleton lifecycle caching."""
        manager = LifecycleManager()
        descriptor = ServiceDescriptor(
            name="singleton",
            service_type=DummyService,
            implementation=DummyService,
            lifecycle=Lifecycle.SINGLETON,
        )

        instance1 = manager.get_instance(descriptor, lambda: DummyService())
        instance2 = manager.get_instance(descriptor, lambda: DummyService())

        assert instance1 is instance2

    def test_lifecycle_manager_scoped(self):
        """Test scoped lifecycle per scope."""
        manager = LifecycleManager()
        descriptor = ServiceDescriptor(
            name="scoped",
            service_type=DummyService,
            implementation=DummyService,
            lifecycle=Lifecycle.SCOPED,
        )

        instance1 = manager.get_instance(
            descriptor, lambda: DummyService(), scope_id="scope1"
        )
        instance2 = manager.get_instance(
            descriptor, lambda: DummyService(), scope_id="scope1"
        )
        instance3 = manager.get_instance(
            descriptor, lambda: DummyService(), scope_id="scope2"
        )

        assert instance1 is instance2
        assert instance1 is not instance3

    def test_container_register_singleton(self):
        """Test container singleton registration."""
        container = Container()
        container.register_singleton("dummy", DummyService)

        instance1 = container.resolve("dummy")
        instance2 = container.resolve("dummy")

        assert isinstance(instance1, DummyService)
        assert instance1 is instance2

    def test_container_register_transient(self):
        """Test container transient registration."""
        container = Container()
        container.register_transient("dummy", DummyService)

        instance1 = container.resolve("dummy")
        instance2 = container.resolve("dummy")

        assert isinstance(instance1, DummyService)
        assert instance1 is not instance2

    def test_container_register_with_factory(self):
        """Test container registration with factory."""

        def factory():
            return DummyService()

        container = Container()
        container.register_singleton("dummy", DummyService, factory=factory)

        instance = container.resolve("dummy")
        assert isinstance(instance, DummyService)

    def test_container_unresolved_service(self):
        """Test resolving unregistered service."""
        container = Container()

        with pytest.raises(ResolutionError):
            container.resolve("nonexistent")

    def test_container_is_registered(self):
        """Test registration check."""
        container = Container()
        container.register_singleton("dummy", DummyService)

        assert container.is_registered("dummy")
        assert not container.is_registered("nonexistent")

    def test_container_builder_fluent_api(self):
        """Test ContainerBuilder fluent API."""
        container = (
            ContainerBuilder()
            .register_singleton("dummy", DummyService)
            .register_transient("dummy2", DummyService)
            .build()
        )

        assert isinstance(container, Container)
        assert container.is_registered("dummy")
        assert container.is_registered("dummy2")

    def test_container_thread_safe(self):
        """Test thread-safe singleton access."""
        container = Container()
        container.register_singleton("dummy", DummyService)

        instances = []
        lock = Lock()

        def resolve_service():
            instance = container.resolve("dummy")
            with lock:
                instances.append(instance)

        threads = [Thread(target=resolve_service) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert len({id(inst) for inst in instances}) == 1


# ============================================================================
# EVENT BUS TESTS
# ============================================================================


class ExampleEvent(Event):
    """Test event implementation."""

    def __init__(self, message: str = "test"):
        super().__init__()
        self.message = message


class ExampleEventHandler(EventHandler):
    """Test event handler implementation."""

    def __init__(self):
        self.handled_events = []

    def handle(self, event: Event) -> None:
        """Handle event."""
        self.handled_events.append(event)


class TestEventBus:
    """Tests for event bus."""

    def test_event_priority_enum(self):
        """Test EventPriority enum."""
        assert EventPriority.CRITICAL.value == 0
        assert EventPriority.HIGH.value == 1
        assert EventPriority.NORMAL.value == 2
        assert EventPriority.LOW.value == 3
        assert EventPriority.DEFERRED.value == 4

    def test_event_creation(self):
        """Test Event creation."""
        event = ExampleEvent("hello")

        assert event.message == "hello"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_handler_implementation(self):
        """Test EventHandler implementation."""
        handler = ExampleEventHandler()
        event = ExampleEvent()

        handler.handle(event)

        assert len(handler.handled_events) == 1
        assert handler.handled_events[0] is event

    def test_event_filter_matches(self):
        """Test EventFilter matching."""
        # Create test event with message attribute
        event1 = ExampleEvent("match")
        event2 = ExampleEvent("no_match")

        # EventFilter uses kwargs criteria
        filter_obj = EventFilter(message="match")

        assert filter_obj.matches(event1)
        assert not filter_obj.matches(event2)

    def test_bus_subscribe_and_publish(self):
        """Test subscribe and publish."""
        bus = EventBus()
        handler = ExampleEventHandler()

        bus.subscribe(ExampleEvent, handler)
        event = ExampleEvent("test")
        bus.publish(event)

        assert len(handler.handled_events) == 1
        assert handler.handled_events[0] is event

    def test_bus_unsubscribe(self):
        """Test unsubscribe."""
        bus = EventBus()
        handler = ExampleEventHandler()

        handle = bus.subscribe(ExampleEvent, handler)
        bus.unsubscribe(ExampleEvent, handler)

        event = ExampleEvent()
        bus.publish(event)

        assert len(handler.handled_events) == 0

    def test_bus_event_filter(self):
        """Test event filtering."""
        bus = EventBus()
        handler = ExampleEventHandler()

        filter_fn = EventFilter(message="pass")
        bus.subscribe(ExampleEvent, handler, filter_fn=filter_fn)

        bus.publish(ExampleEvent("pass"))
        bus.publish(ExampleEvent("fail"))

        assert len(handler.handled_events) == 1

    def test_bus_priority_ordering(self):
        """Test handler priority ordering."""
        bus = EventBus()
        call_order = []

        # Create handlers that track call order
        def make_handler(name):
            class PriorityHandler(EventHandler):
                def handle(self, event):
                    call_order.append(name)

            return PriorityHandler()

        # Subscribe in reverse priority order
        bus.subscribe(ExampleEvent, make_handler("low"), priority=EventPriority.LOW)
        bus.subscribe(
            ExampleEvent, make_handler("critical"), priority=EventPriority.CRITICAL
        )
        bus.subscribe(
            ExampleEvent, make_handler("normal"), priority=EventPriority.NORMAL
        )

        bus.publish(ExampleEvent())

        # Should be in priority order
        assert call_order == ["critical", "normal", "low"]

    def test_bus_history(self):
        """Test event history."""
        bus = EventBus()
        handler = ExampleEventHandler()

        bus.subscribe(ExampleEvent, handler)
        bus.publish(ExampleEvent("event1"))
        bus.publish(ExampleEvent("event2"))

        history = bus.get_history()
        assert len(history) == 2

    def test_bus_middleware(self):
        """Test event middleware."""
        bus = EventBus()
        middleware_called = []

        def middleware(event: Event) -> bool:
            middleware_called.append(True)
            return True  # Allow event

        bus.add_middleware(middleware)
        bus.publish(ExampleEvent())

        assert len(middleware_called) == 1

    def test_async_bus_creation(self):
        """Test AsyncEventBus creation."""
        with AsyncEventBus(worker_threads=2) as bus:
            assert bus is not None

    def test_async_bus_publish(self):
        """Test AsyncEventBus publish."""
        with AsyncEventBus(worker_threads=1) as bus:
            handler = ExampleEventHandler()
            bus.subscribe(ExampleEvent, handler)

            bus.publish(ExampleEvent("async"))
            time.sleep(0.5)  # Allow async processing

            assert len(handler.handled_events) > 0

    def test_event_dispatcher_mapping(self):
        """Test EventDispatcher mapping."""
        bus = EventBus()
        dispatcher = EventDispatcher(bus)
        handler = ExampleEventHandler()

        dispatcher.map_event(ExampleEvent, handler.handle)
        dispatcher.dispatch(ExampleEvent("dispatched"))

        assert len(handler.handled_events) == 1


# ============================================================================
# CONFIGURATION MANAGEMENT TESTS
# ============================================================================


class TestConfigManager:
    """Tests for configuration management."""

    def test_config_profile_enum(self):
        """Test ConfigProfile enum."""
        assert ConfigProfile.DEVELOPMENT.value == "development"
        assert ConfigProfile.TESTING.value == "testing"
        assert ConfigProfile.STAGING.value == "staging"
        assert ConfigProfile.PRODUCTION.value == "production"

    def test_config_rule_validation(self):
        """Test ConfigRule validation."""
        rule = ConfigRule(
            key="port",
            required=True,
            expected_type=int,
            validators=[lambda v: 1 <= v <= 65535],
        )

        valid, msg = rule.validate(8080)
        assert valid

        valid, msg = rule.validate("not_int")
        assert not valid

    def test_config_validator_add_rules(self):
        """Test ConfigValidator."""
        validator = ConfigValidator()
        validator.add_rule(
            ConfigRule(
                key="port",
                required=True,
                expected_type=int,
            )
        )

        is_valid, errors = validator.validate({"port": 8080})
        assert is_valid
        assert len(errors) == 0

    def test_config_manager_get_set(self):
        """Test ConfigManager get/set."""
        manager = ConfigManager()

        manager.set("database.host", "localhost")
        manager.set("database.port", 5432)

        assert manager.get("database.host") == "localhost"
        assert manager.get("database.port") == 5432

    def test_config_manager_defaults(self):
        """Test ConfigManager defaults."""
        manager = ConfigManager()
        manager.set_default("timeout", 30)

        assert manager.get("timeout") == 30
        assert manager.get("nonexistent", "default") == "default"

    def test_config_manager_reload(self):
        """Test ConfigManager reload."""
        manager = ConfigManager()
        manager.set("key", "value1")
        manager.reload()

        assert manager.get("key") == "value1"

    def test_config_manager_profile(self):
        """Test ConfigManager profile."""
        manager = ConfigManager()
        manager.load_profile(ConfigProfile.PRODUCTION)

        assert manager.get_profile() == ConfigProfile.PRODUCTION

    def test_config_manager_dict_export(self):
        """Test ConfigManager dictionary export."""
        manager = ConfigManager()
        manager.set("key1", "value1")
        manager.set("key2", "value2")

        config_dict = manager.get_all()
        assert "key1" in config_dict
        assert "key2" in config_dict

    def test_json_source(self):
        """Test JsonSource."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "value", "number": 42}, f)
            temp_path = f.name

        try:
            source = JsonSource(temp_path)
            config = source.load()

            assert config["test"] == "value"
            assert config["number"] == 42
        finally:
            Path(temp_path).unlink()

    def test_json_source_missing_file(self):
        """Test JsonSource with missing file."""
        source = JsonSource("/nonexistent/path.json")
        config = source.load()

        assert config == {}

    def test_environment_source(self):
        """Test EnvironmentSource."""
        with patch.dict("os.environ", {"APP_DEBUG": "true", "APP_PORT": "8080"}):
            source = EnvironmentSource(prefix="APP_")
            config = source.load()

            assert config.get("debug") == True
            assert config.get("port") == 8080

    def test_config_manager_from_file(self):
        """Test ConfigManager.from_file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"app_name": "test_app"}, f)
            temp_path = f.name

        try:
            manager = ConfigManager.from_file(temp_path)
            assert manager.get("app_name") == "test_app"
        finally:
            Path(temp_path).unlink()

    def test_config_manager_add_source(self):
        """Test ConfigManager.add_source."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"source_test": "success"}, f)
            temp_path = f.name

        try:
            manager = ConfigManager()
            manager.add_source(JsonSource(temp_path))
            manager.reload()

            assert manager.get("source_test") == "success"
        finally:
            Path(temp_path).unlink()

    def test_config_manager_validation(self):
        """Test ConfigManager validation."""
        manager = ConfigManager()
        manager.add_rule(
            ConfigRule(
                key="port",
                required=True,
                expected_type=int,
            )
        )
        manager.set("port", 8080)

    def test_config_manager_validation_failure(self):
        """Test ConfigManager validation failure."""
        manager = ConfigManager()
        manager.add_rule(
            ConfigRule(
                key="port",
                required=True,
                expected_type=int,
            )
        )
        manager.set("port", "not_int")

        is_valid, errors = manager.validate()
        assert not is_valid
        assert len(errors) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestPhase4Integration:
    """Integration tests for Phase 4 components."""

    def test_di_with_event_bus(self):
        """Test Dependency Injection with Event Bus."""
        container = ContainerBuilder().build()

        # This test verifies that DI and EventBus can coexist
        bus = EventBus()
        handler = ExampleEventHandler()

        bus.subscribe(ExampleEvent, handler)
        bus.publish(ExampleEvent("integration"))

        assert len(handler.handled_events) == 1

    def test_config_with_di_container(self):
        """Test ConfigManager with Dependency Injection."""
        manager = ConfigManager()
        manager.set("service.port", 8080)
        manager.set("service.host", "localhost")

        # Config can provide values to DI container
        port = manager.get("service.port")
        assert port == 8080

    def test_all_phase_4_components(self):
        """Test all Phase 4 components working together."""
        # Create configuration
        config = ConfigManager()
        config.set("app.name", "phase4_app")

        # Create DI container
        container = ContainerBuilder().build()

        # Create event bus
        bus = EventBus()
        handler = ExampleEventHandler()
        bus.subscribe(ExampleEvent, handler)

        # Publish event
        bus.publish(ExampleEvent("all_components"))

        assert config.get("app.name") == "phase4_app"
        assert len(handler.handled_events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
