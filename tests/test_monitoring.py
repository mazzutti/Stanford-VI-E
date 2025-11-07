"""
Tests for Phase 5.4: Logging & Monitoring
"""

import json
import time
import logging
import pytest
import threading
from threading import Thread
from datetime import datetime

from src.analysis.monitoring import (
    LogLevel,
    MetricType,
    LogEvent,
    MetricValue,
    PerformanceMetrics,
    StructuredLogger,
    MetricsCollector,
    PerformanceMonitor,
    SimpleHealthCheck,
    HealthCheckRegistry,
    monitor,
)


class TestLogLevel:
    """Test LogLevel enum."""

    def test_log_level_values(self):
        """Test log level enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestMetricType:
    """Test MetricType enum."""

    def test_metric_type_values(self):
        """Test metric type enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"


class TestLogEvent:
    """Test LogEvent dataclass."""

    def test_log_event_creation(self):
        """Test creating a log event."""
        event = LogEvent(
            timestamp="2023-01-01T00:00:00",
            level="INFO",
            message="Test message",
            service="test_service",
            context={"key": "value"},
        )

        assert event.timestamp == "2023-01-01T00:00:00"
        assert event.level == "INFO"
        assert event.message == "Test message"
        assert event.service == "test_service"
        assert event.context == {"key": "value"}

    def test_log_event_json_conversion(self):
        """Test converting log event to JSON."""
        event = LogEvent(
            timestamp="2023-01-01T00:00:00",
            level="INFO",
            message="Test message",
            service="test_service",
            context={"key": "value"},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["service"] == "test_service"
        assert parsed["context"]["key"] == "value"

    def test_log_event_string_representation(self):
        """Test string representation of log event."""
        event = LogEvent(
            timestamp="2023-01-01T00:00:00",
            level="INFO",
            message="Test message",
            service="test_service",
        )

        str_repr = str(event)
        assert "[2023-01-01T00:00:00] INFO: Test message" == str_repr


class TestMetricValue:
    """Test MetricValue dataclass."""

    def test_metric_value_creation(self):
        """Test creating a metric value."""
        now = time.time()
        metric = MetricValue(
            name="test_metric",
            type=MetricType.COUNTER,
            value=1.0,
            timestamp=now,
            tags={"service": "test"},
        )

        assert metric.name == "test_metric"
        assert metric.type == MetricType.COUNTER
        assert metric.value == 1.0
        assert metric.tags["service"] == "test"

    def test_metric_value_string_representation(self):
        """Test string representation of metric value."""
        metric = MetricValue(
            name="test_metric",
            type=MetricType.COUNTER,
            value=42.0,
            timestamp=time.time(),
            tags={"env": "test"},
        )

        str_repr = str(metric)
        assert "test_metric(counter): 42.0" in str_repr
        assert "env=test" in str_repr


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        start = time.time()
        metrics = PerformanceMetrics(name="test_op", start_time=start)

        assert metrics.name == "test_op"
        assert metrics.start_time == start
        assert metrics.end_time is None
        assert metrics.duration is None
        assert not metrics.success
        assert not metrics.is_complete

    def test_performance_metrics_completion(self):
        """Test completing performance metrics."""
        start = time.time()
        metrics = PerformanceMetrics(name="test_op", start_time=start)

        time.sleep(0.01)
        metrics.complete(success=True)

        assert metrics.end_time > start
        assert metrics.duration >= 0.01
        assert metrics.success
        assert metrics.is_complete

    def test_performance_metrics_with_error(self):
        """Test performance metrics with error."""
        metrics = PerformanceMetrics(name="test_op", start_time=time.time())
        metrics.complete(success=False, error="Test error")

        assert not metrics.success
        assert metrics.error == "Test error"
        assert metrics.is_complete

    def test_performance_metrics_string_representation(self):
        """Test string representation of performance metrics."""
        start = time.time()
        metrics = PerformanceMetrics(name="test_op", start_time=start)
        metrics.complete(success=True)

        str_repr = str(metrics)
        assert "✓" in str_repr
        assert "test_op" in str_repr


class TestStructuredLogger:
    """Test StructuredLogger."""

    def test_logger_creation(self):
        """Test creating a structured logger."""
        logger = StructuredLogger("test_service")

        assert logger.service_name == "test_service"
        assert logger.log_level == LogLevel.INFO
        assert logger.include_context is True

    def test_logger_debug_message(self):
        """Test logging debug message."""
        logger = StructuredLogger("test_service")
        event = logger.debug("Debug message", key="value")

        assert event.level == "DEBUG"
        assert event.message == "Debug message"
        assert event.context["key"] == "value"

    def test_logger_info_message(self):
        """Test logging info message."""
        logger = StructuredLogger("test_service")
        event = logger.info("Info message", operation="test")

        assert event.level == "INFO"
        assert event.message == "Info message"
        assert event.context["operation"] == "test"

    def test_logger_warning_message(self):
        """Test logging warning message."""
        logger = StructuredLogger("test_service")
        event = logger.warning("Warning message", alert_level=1)

        assert event.level == "WARNING"
        assert event.message == "Warning message"
        assert event.context["alert_level"] == 1

    def test_logger_error_message(self):
        """Test logging error message."""
        logger = StructuredLogger("test_service")
        event = logger.error("Error message", error_code=500)

        assert event.level == "ERROR"
        assert event.message == "Error message"
        assert event.context["error_code"] == 500

    def test_logger_critical_message(self):
        """Test logging critical message."""
        logger = StructuredLogger("test_service")
        event = logger.critical("Critical message", severity="high")

        assert event.level == "CRITICAL"
        assert event.message == "Critical message"
        assert event.context["severity"] == "high"

    def test_logger_with_exception(self):
        """Test logging with exception."""
        logger = StructuredLogger("test_service")
        exc = ValueError("Test error")
        event = logger.error("Operation failed", exception=exc)

        assert event.level == "ERROR"
        assert "ValueError" in event.exception or "Test error" in event.exception

    def test_logger_context_stack(self):
        """Test context stack functionality."""
        logger = StructuredLogger("test_service")

        logger.push_context(request_id="123")
        logger.push_context(user_id="user456")

        # Contexts should be available
        assert len(logger._context_stack) == 2

        logger.pop_context()
        assert len(logger._context_stack) == 1

        logger.pop_context()
        assert len(logger._context_stack) == 0

    def test_logger_context_manager(self):
        """Test logger as context manager."""
        with StructuredLogger("test_service") as logger:
            event = logger.info("Test message")
            assert event.level == "INFO"


class TestMetricsCollector:
    """Test MetricsCollector."""

    def test_collector_creation(self):
        """Test creating metrics collector."""
        collector = MetricsCollector("test_collector")
        assert collector.name == "test_collector"

    def test_record_counter(self):
        """Test recording counter metric."""
        collector = MetricsCollector()
        collector.record_counter("requests", 1)
        collector.record_counter("requests", 2)

        summary = collector.get_metrics_summary()
        assert summary["requests"]["type"] == "counter"
        assert summary["requests"]["count"] == 2
        assert summary["requests"]["sum"] == 3

    def test_record_gauge(self):
        """Test recording gauge metric."""
        collector = MetricsCollector()
        collector.record_gauge("memory", 100.0)
        collector.record_gauge("memory", 150.0)

        summary = collector.get_metrics_summary()
        assert summary["memory"]["type"] == "gauge"
        assert summary["memory"]["latest"] == 150.0
        assert summary["memory"]["count"] == 2

    def test_record_histogram(self):
        """Test recording histogram metric."""
        collector = MetricsCollector()
        collector.record_histogram("response_time", 0.1)
        collector.record_histogram("response_time", 0.2)
        collector.record_histogram("response_time", 0.3)

        summary = collector.get_metrics_summary()
        assert summary["response_time"]["type"] == "histogram"
        assert summary["response_time"]["count"] == 3
        assert summary["response_time"]["min"] == 0.1
        assert summary["response_time"]["max"] == 0.3

    def test_record_timer(self):
        """Test recording timer metric."""
        collector = MetricsCollector()
        collector.record_timer("operation_duration", 1.5)
        collector.record_timer("operation_duration", 2.0)

        summary = collector.get_metrics_summary()
        assert summary["operation_duration"]["type"] == "timer"
        assert summary["operation_duration"]["count"] == 2

    def test_get_metric(self):
        """Test getting single metric."""
        collector = MetricsCollector()
        collector.record_counter("test", 5)

        metric = collector.get_metric("test")
        assert metric is not None
        assert metric.value == 5

    def test_get_nonexistent_metric(self):
        """Test getting nonexistent metric."""
        collector = MetricsCollector()
        metric = collector.get_metric("nonexistent")

        assert metric is None

    def test_metrics_with_tags(self):
        """Test metrics with tags."""
        collector = MetricsCollector()
        collector.record_counter("requests", 1, tags={"endpoint": "/api/v1"})

        summary = collector.get_metrics_summary()
        assert summary["requests"]["tags"]["endpoint"] == "/api/v1"

    def test_metrics_summary_statistics(self):
        """Test metrics summary includes statistics."""
        collector = MetricsCollector()
        for val in [10, 20, 30, 40, 50]:
            collector.record_histogram("values", float(val))

        summary = collector.get_metrics_summary()
        metrics = summary["values"]

        assert metrics["sum"] == 150
        assert metrics["min"] == 10
        assert metrics["max"] == 50
        assert metrics["avg"] == 30

    def test_clear_metrics(self):
        """Test clearing metrics."""
        collector = MetricsCollector()
        collector.record_counter("test", 1)
        collector.record_gauge("gauge", 42)

        assert len(collector._metrics) > 0
        collector.clear()
        assert len(collector._metrics) == 0


class TestPerformanceMonitor:
    """Test PerformanceMonitor context manager."""

    def test_monitor_basic_usage(self):
        """Test basic monitor usage."""
        with PerformanceMonitor("test_operation") as monitor:
            time.sleep(0.01)

        assert monitor.metrics.name == "test_operation"
        assert monitor.metrics.duration >= 0.01
        assert monitor.metrics.success

    def test_monitor_with_exception(self):
        """Test monitor catches exceptions."""
        with pytest.raises(ValueError):
            with PerformanceMonitor("failing_operation") as monitor:
                raise ValueError("Test error")

        assert monitor.metrics.success is False
        assert "Test error" in monitor.metrics.error

    def test_monitor_with_logger(self):
        """Test monitor with structured logger."""
        logger = StructuredLogger("test_service")

        with PerformanceMonitor("test_op", logger=logger) as monitor:
            time.sleep(0.01)

        assert monitor.metrics.success
        assert monitor.metrics.duration >= 0.01

    def test_monitor_get_metrics(self):
        """Test getting metrics from monitor."""
        with PerformanceMonitor("test_op") as monitor:
            time.sleep(0.02)

        metrics = monitor.get_metrics()
        assert metrics.name == "test_op"
        assert metrics.duration >= 0.02


class TestHealthCheck:
    """Test health checks."""

    def test_simple_health_check_pass(self):
        """Test simple health check passes."""

        def check_func():
            return True

        check = SimpleHealthCheck("memory_check", check_func)
        assert check.name == "memory_check"
        assert check.check() is True

    def test_simple_health_check_fail(self):
        """Test simple health check fails."""

        def check_func():
            return False

        check = SimpleHealthCheck("memory_check", check_func)
        assert check.check() is False

    def test_health_check_with_exception(self):
        """Test health check handles exceptions."""

        def check_func():
            raise RuntimeError("Check error")

        check = SimpleHealthCheck("broken_check", check_func)
        assert check.check() is False


class TestHealthCheckRegistry:
    """Test health check registry."""

    def test_register_health_check(self):
        """Test registering health check."""
        registry = HealthCheckRegistry()
        check = SimpleHealthCheck("test_check", lambda: True)

        registry.register(check)
        assert "test_check" in registry._checks

    def test_unregister_health_check(self):
        """Test unregistering health check."""
        registry = HealthCheckRegistry()
        check = SimpleHealthCheck("test_check", lambda: True)

        registry.register(check)
        registry.unregister("test_check")

        assert "test_check" not in registry._checks

    def test_check_all_healthy(self):
        """Test check_all when all healthy."""
        registry = HealthCheckRegistry()
        registry.register(SimpleHealthCheck("check1", lambda: True))
        registry.register(SimpleHealthCheck("check2", lambda: True))

        results = registry.check_all()
        assert all(results.values())
        assert registry.is_healthy() is True

    def test_check_all_with_failure(self):
        """Test check_all with failure."""
        registry = HealthCheckRegistry()
        registry.register(SimpleHealthCheck("check1", lambda: True))
        registry.register(SimpleHealthCheck("check2", lambda: False))

        results = registry.check_all()
        assert results["check1"] is True
        assert results["check2"] is False
        assert registry.is_healthy() is False

    def test_check_all_empty_registry(self):
        """Test check_all with empty registry."""
        registry = HealthCheckRegistry()

        results = registry.check_all()
        assert results == {}
        assert registry.is_healthy() is True


class TestMonitorDecorator:
    """Test @monitor decorator."""

    def test_monitor_decorator_basic(self):
        """Test basic monitor decorator usage."""
        logger = StructuredLogger("test_service")
        metrics = MetricsCollector()

        @monitor(logger=logger, metrics=metrics)
        def monitored_service_function():
            time.sleep(0.01)
            return "result"

        result = monitored_service_function()

        assert result == "result"
        summary = metrics.get_metrics_summary()
        assert "monitored_service_function_duration" in summary

    def test_monitor_decorator_with_args(self):
        """Test monitor decorator with function arguments."""
        logger = StructuredLogger("test_service")

        @monitor(logger=logger)
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_monitor_decorator_exception(self):
        """Test monitor decorator with exception."""
        logger = StructuredLogger("test_service")

        @monitor(logger=logger)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_monitor_decorator_without_logger(self):
        """Test monitor decorator without logger."""

        @monitor()
        def simple_function():
            return 42

        result = simple_function()
        assert result == 42


class TestMonitoringThreadSafety:
    """Test thread-safety of monitoring components."""

    def test_logger_thread_safety(self):
        """Test logger is thread-safe."""
        logger = StructuredLogger("test_service")
        results = []

        def log_message(msg_id):
            event = logger.info(f"Message {msg_id}")
            results.append(event)

        threads = [Thread(target=log_message, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10

    def test_metrics_collector_thread_safety(self):
        """Test metrics collector is thread-safe."""
        collector = MetricsCollector()

        def record_metrics(count):
            for i in range(count):
                collector.record_counter("requests", 1)

        threads = [Thread(target=record_metrics, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = collector.get_metrics_summary()
        assert summary["requests"]["count"] == 50

    def test_health_check_registry_thread_safety(self):
        """Test health check registry is thread-safe."""
        registry = HealthCheckRegistry()
        counter = {"value": 0}

        def register_checks():
            for i in range(5):
                thread_id = threading.current_thread().ident
                check = SimpleHealthCheck(f"check_{thread_id}_{i}", lambda: True)
                registry.register(check)
                counter["value"] += 1

        threads = [Thread(target=register_checks) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All checks should be registered
        assert counter["value"] == 15


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        logger = StructuredLogger("service")
        metrics = MetricsCollector()
        registry = HealthCheckRegistry()

        # Add health checks
        registry.register(SimpleHealthCheck("db_check", lambda: True))
        registry.register(SimpleHealthCheck("api_check", lambda: True))

        # Log operation start
        logger.info("Starting operation", operation="batch_process")

        # Monitor operation
        with PerformanceMonitor("batch_process", logger=logger) as monitor:
            metrics.record_counter("items_processed", 10)
            time.sleep(0.01)

        # Record metrics
        metrics.record_histogram("processing_time", monitor.metrics.duration)

        # Check health
        assert registry.is_healthy()

        # Log completion
        logger.info("Operation complete", status="success")

        # Verify results
        summary = metrics.get_metrics_summary()
        assert "items_processed" in summary
        assert "processing_time" in summary

    def test_monitoring_with_multiple_services(self):
        """Test monitoring multiple services."""
        services = {}
        for i in range(3):
            services[f"service_{i}"] = {
                "logger": StructuredLogger(f"service_{i}"),
                "metrics": MetricsCollector(f"service_{i}"),
            }

        # Each service logs and collects metrics
        for name, components in services.items():
            components["logger"].info("Service started")
            components["metrics"].record_counter("startup", 1)

        # Verify all services have metrics
        for name, components in services.items():
            summary = components["metrics"].get_metrics_summary()
            assert "startup" in summary
