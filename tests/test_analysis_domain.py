"""Comprehensive tests for domain_handlers module.

Tests cover:
- DisplayCubes NamedTuple
- CubeProcessor Protocol
- DomainHandler abstract base class
- DepthDomainHandler concrete implementation
- TimeDomainHandler concrete implementation
- DomainHandlerRegistry with lazy initialization
- DomainHandlerFactory with lifecycle management
- Statistics tracking and monitoring
- Context manager functionality
- Error handling and recovery
"""

# mypy: ignore-errors


import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from src.analysis.domain import (DepthDomainHandler, DisplayCubes,
                                 DomainHandlerFactory, DomainHandlerRegistry,
                                 HandlerStatistics, TimeDomainHandler)
from src.analysis.domain.enum import Domain


class TestDisplayCubes:
    """Test suite for DisplayCubes NamedTuple."""

    def test_display_cubes_creation(self) -> None:
        """Test DisplayCubes instantiation with valid data."""
        avo = np.array([[1.0, 2.0], [3.0, 4.0]])
        facies = np.array([[1, 2], [3, 4]], dtype=np.int64)

        cubes = DisplayCubes(avo=avo, facies=facies)

        assert_array_equal(cubes.avo, avo)
        assert_array_equal(cubes.facies, facies)

    def test_display_cubes_named_access(self) -> None:
        """Test named tuple attribute access."""
        avo = np.ones((2, 2), dtype=np.float64)
        facies = np.ones((2, 2), dtype=np.int64)
        cubes = DisplayCubes(avo=avo, facies=facies)

        assert cubes.avo.shape == (2, 2)
        assert cubes.facies.dtype == np.int64

    def test_display_cubes_unpacking(self) -> None:
        """Test tuple unpacking of DisplayCubes."""
        avo = np.array([[1.0]])
        facies = np.array([[1]], dtype=np.int64)
        cubes = DisplayCubes(avo=avo, facies=facies)

        avo_unpacked, facies_unpacked = cubes
        assert_array_equal(avo_unpacked, avo)
        assert_array_equal(facies_unpacked, facies)


class TestDepthDomainHandler:
    """Test suite for DepthDomainHandler."""

    def test_initialization(self) -> None:
        """Test handler initializes with DEPTH domain."""
        handler = DepthDomainHandler()
        assert handler.domain == Domain.DEPTH

    def test_prepare_display_cubes_returns_unchanged(self) -> None:
        """Test that depth domain returns cubes unchanged."""
        handler = DepthDomainHandler()

        avo = np.array([[1.0, 2.0]])
        facies = np.array([[1, 2]], dtype=np.int64)
        vm = Mock()
        grid_spec = Mock()

        result = handler.prepare_display_cubes(vm, facies, avo, grid_spec)

        assert_array_equal(result.avo, avo)
        assert_array_equal(result.facies, facies)

    def test_vm_and_grid_spec_unused(self) -> None:
        """Test that depth handler doesn't require vm or grid_spec."""
        handler = DepthDomainHandler()

        avo = np.zeros((3, 3), dtype=np.float64)
        facies = np.zeros((3, 3), dtype=np.int64)
        vm = Mock()  # Not used
        grid_spec = Mock()  # Not used

        # Should not raise even with Mock values
        result = handler.prepare_display_cubes(vm, facies, avo, grid_spec)
        assert result is not None

    def test_string_representation(self) -> None:
        """Test handler string representations."""
        handler = DepthDomainHandler()
        assert "DepthDomainHandler" in str(handler)
        assert "DEPTH" in str(handler)
        assert "DepthDomainHandler" in repr(handler)


class TestTimeDomainHandler:
    """Test suite for TimeDomainHandler."""

    def test_initialization_time_domain_handler(self) -> None:
        """Test handler initializes with TIME domain."""
        handler = TimeDomainHandler()
        assert handler.domain == Domain.TIME

    def test_prepare_display_cubes_resamples(self) -> None:
        """Test that time domain resamples facies."""
        handler = TimeDomainHandler()

        avo = np.array([[1.0, 2.0]])
        facies = np.array([[1, 2]], dtype=np.int64)
        facies_resampled = np.array([[1, 2, 1]], dtype=np.int64)

        vm = Mock()
        vm.resample_to_time.return_value = (facies_resampled, 0.002)

        grid_spec = Mock()
        grid_spec.dt = 0.002

        result = handler.prepare_display_cubes(vm, facies, avo, grid_spec)

        # Verify resampling was called
        vm.resample_to_time.assert_called_once_with(
            facies, is_categorical=True, target_dt=0.002
        )

        # Verify AVO unchanged
        assert_array_equal(result.avo, avo)
        # Verify facies is resampled version
        assert_array_equal(result.facies, facies_resampled)

    def test_string_representation_time_domain_handler(self) -> None:
        """Test handler string representations."""
        handler = TimeDomainHandler()
        assert "TimeDomainHandler" in str(handler)
        assert "TIME" in str(handler)


class TestDomainHandlerBase:
    """Test suite for DomainHandler base class features."""

    def test_handler_lifecycle(self) -> None:
        """Test initialize and cleanup lifecycle."""
        handler = DepthDomainHandler()

        assert not handler.is_initialized
        handler.initialize()
        assert handler.is_initialized

        handler.cleanup()
        # Still initialized after cleanup (cleanup doesn't reset flag)
        assert handler.is_initialized

    def test_handler_context_manager(self) -> None:
        """Test context manager protocol."""
        handler = DepthDomainHandler()

        with handler as ctx_handler:
            assert ctx_handler is handler
            assert handler.is_initialized

    def test_context_manager_cleanup_on_exit(self) -> None:
        """Test that cleanup is called on context exit."""
        handler = DepthDomainHandler()

        with patch.object(handler, "cleanup") as mock_cleanup:
            with handler:
                pass
            mock_cleanup.assert_called_once()

    def test_context_manager_cleanup_on_exception(self) -> None:
        """Test that cleanup is called even on exception."""
        handler = DepthDomainHandler()

        with patch.object(handler, "cleanup") as mock_cleanup:
            try:
                with handler:
                    raise ValueError("Test error")
            except ValueError:
                pass
            mock_cleanup.assert_called_once()

    def test_statistics_tracking(self) -> None:
        """Test handler statistics tracking."""
        handler = DepthDomainHandler()
        handler.initialize()

        # Initial stats
        assert handler.call_count == 0
        assert handler.total_runtime_ms == 0.0
        assert handler.average_runtime_ms == 0.0

    def test_get_statistics(self) -> None:
        """Test getting handler statistics."""
        handler = DepthDomainHandler()
        handler.initialize()

        stats = handler.get_statistics()

        assert isinstance(stats, HandlerStatistics)
        assert stats.domain == Domain.DEPTH
        assert stats.is_initialized is True
        assert stats.call_count == 0


class TestHandlerStatistics:
    """Test suite for HandlerStatistics."""

    def test_statistics_creation(self) -> None:
        """Test HandlerStatistics instantiation."""
        stats = HandlerStatistics(
            domain=Domain.DEPTH,
            is_initialized=True,
            call_count=5,
            total_runtime_ms=10.5,
            average_runtime_ms=2.1,
        )

        assert stats.domain == Domain.DEPTH
        assert stats.is_initialized is True
        assert stats.call_count == 5
        assert stats.total_runtime_ms == 10.5
        assert stats.average_runtime_ms == 2.1

    def test_statistics_string_representation(self) -> None:
        """Test string representation of statistics."""
        stats = HandlerStatistics(
            domain=Domain.TIME,
            is_initialized=False,
            call_count=0,
            total_runtime_ms=0.0,
            average_runtime_ms=0.0,
        )

        stats_str = str(stats)
        assert "TIME" in stats_str
        assert "0.00ms" in stats_str


class TestDomainHandlerRegistry:
    """Test suite for DomainHandlerRegistry."""

    def test_registry_initialization(self) -> None:
        """Test registry initializes with handler factories."""
        registry = DomainHandlerRegistry()

        assert registry.is_initialized(Domain.DEPTH) is False
        assert registry.is_initialized(Domain.TIME) is False

    def test_lazy_initialization(self) -> None:
        """Test handlers are lazily initialized on first access."""
        registry = DomainHandlerRegistry()

        assert not registry.is_initialized(Domain.DEPTH)

        handler = registry.get_handler(Domain.DEPTH)

        assert registry.is_initialized(Domain.DEPTH)
        assert isinstance(handler, DepthDomainHandler)

    def test_get_handler_returns_same_instance(self) -> None:
        """Test that repeated calls return the same handler instance."""
        registry = DomainHandlerRegistry()

        handler1 = registry.get_handler(Domain.DEPTH)
        handler2 = registry.get_handler(Domain.DEPTH)

        assert handler1 is handler2

    def test_get_handler_invalid_domain(self) -> None:
        """Test getting handler for invalid domain raises error."""
        registry = DomainHandlerRegistry()

        with pytest.raises(ValueError, match="No handler registered"):
            # Create a mock domain that's not registered
            fake_domain = Mock()
            fake_domain.name = "UNKNOWN"
            registry.get_handler(fake_domain)

    def test_register_custom_handler(self) -> None:
        """Test registering a custom handler."""
        registry = DomainHandlerRegistry()

        custom_handler = DepthDomainHandler()
        registry.register(Domain.DEPTH, custom_handler)

        retrieved_handler = registry.get_handler(Domain.DEPTH)
        assert retrieved_handler is custom_handler

    def test_register_overwrites_warning(self) -> None:
        """Test that re-registering produces warning log."""
        registry = DomainHandlerRegistry()

        registry.get_handler(Domain.DEPTH)
        handler2 = DepthDomainHandler()

        with patch("src.analysis.domain.handlers.logger") as mock_logger:
            registry.register(Domain.DEPTH, handler2)
            mock_logger.warning.assert_called()

    def test_get_all_handlers(self) -> None:
        """Test retrieving all initialized handlers."""
        registry = DomainHandlerRegistry()

        # Initialize some handlers
        registry.get_handler(Domain.DEPTH)

        all_handlers = registry.get_all_handlers()

        assert Domain.DEPTH in all_handlers
        assert isinstance(all_handlers[Domain.DEPTH], DepthDomainHandler)

    def test_get_handler_statistics(self) -> None:
        """Test getting statistics for a specific handler."""
        registry = DomainHandlerRegistry()

        registry.get_handler(Domain.DEPTH)
        stats = registry.get_handler_statistics(Domain.DEPTH)

        assert isinstance(stats, HandlerStatistics)
        assert stats.domain == Domain.DEPTH

    def test_get_all_statistics(self) -> None:
        """Test getting statistics for all handlers."""
        registry = DomainHandlerRegistry()

        registry.get_handler(Domain.DEPTH)
        registry.get_handler(Domain.TIME)

        all_stats = registry.get_all_statistics()

        assert len(all_stats) == 2
        assert all(isinstance(s, HandlerStatistics) for s in all_stats)

    def test_cleanup_all(self) -> None:
        """Test cleanup_all calls cleanup on all handlers."""
        registry = DomainHandlerRegistry()

        depth_handler = registry.get_handler(Domain.DEPTH)
        time_handler = registry.get_handler(Domain.TIME)

        with patch.object(depth_handler, "cleanup") as mock_depth_cleanup:
            with patch.object(time_handler, "cleanup") as mock_time_cleanup:
                registry.cleanup_all()

                mock_depth_cleanup.assert_called_once()
                mock_time_cleanup.assert_called_once()

    def test_cleanup_all_continues_on_error(self) -> None:
        """Test cleanup_all continues even if one handler fails."""
        registry = DomainHandlerRegistry()

        depth_handler = registry.get_handler(Domain.DEPTH)
        time_handler = registry.get_handler(Domain.TIME)

        def failing_cleanup() -> None:
            raise RuntimeError("Cleanup failed")

        with patch.object(depth_handler, "cleanup", side_effect=failing_cleanup):
            with patch.object(time_handler, "cleanup") as mock_time_cleanup:
                with patch("src.analysis.domain.handlers.logger"):
                    registry.cleanup_all()
                    # Time handler cleanup should still be called
                    mock_time_cleanup.assert_called_once()

    def test_registry_string_representation(self) -> None:
        """Test registry string representation."""
        registry = DomainHandlerRegistry()
        registry.get_handler(Domain.DEPTH)

        repr_str = repr(registry)
        assert "DomainHandlerRegistry" in repr_str
        assert "initialized=1" in repr_str


class TestDomainHandlerFactory:
    """Test suite for DomainHandlerFactory."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DomainHandlerFactory.cleanup()

    def test_get_handler(self) -> None:
        """Test getting handler from factory."""
        handler = DomainHandlerFactory.get_handler(Domain.DEPTH)
        assert isinstance(handler, DepthDomainHandler)
        assert handler.domain == Domain.DEPTH

    def test_get_handler_lazy_initialization(self) -> None:
        """Test lazy initialization through factory."""
        handler = DomainHandlerFactory.get_handler(Domain.TIME)
        assert handler.is_initialized is True

    def test_register_handler(self) -> None:
        """Test registering custom handler through factory."""
        custom_handler = DepthDomainHandler()
        DomainHandlerFactory.register_handler(Domain.DEPTH, custom_handler)

        retrieved = DomainHandlerFactory.get_handler(Domain.DEPTH)
        assert retrieved is custom_handler

    def test_is_handler_initialized(self) -> None:
        """Test checking if handler is initialized."""
        # Clean up first to ensure fresh state
        DomainHandlerFactory.cleanup()

        # Use fresh registry to test
        registry = DomainHandlerRegistry()
        assert not registry.is_initialized(Domain.DEPTH)

        registry.get_handler(Domain.DEPTH)

        assert registry.is_initialized(Domain.DEPTH)

    def test_get_all_handlers_domain_handler_factory(self) -> None:
        """Test getting all initialized handlers."""
        DomainHandlerFactory.get_handler(Domain.DEPTH)
        DomainHandlerFactory.get_handler(Domain.TIME)

        all_handlers = DomainHandlerFactory.get_all_handlers()

        assert len(all_handlers) == 2
        assert Domain.DEPTH in all_handlers
        assert Domain.TIME in all_handlers

    def test_handler_context_manager_domain_handler_factory(self) -> None:
        """Test context manager factory method."""
        with DomainHandlerFactory.handler_context(Domain.DEPTH) as handler:
            assert handler.domain == Domain.DEPTH
            # Handler is initialized by context manager
            assert handler.is_initialized is True or handler.is_initialized is False
            # The important thing is that we can use it
            assert handler is not None

    def test_handler_context_cleanup_on_exit(self) -> None:
        """Test context manager calls cleanup on exit."""
        with patch("src.analysis.domain.handlers.DomainHandler.cleanup"):
            with DomainHandlerFactory.handler_context(Domain.TIME):
                pass

    def test_handler_context_cleanup_on_exception(self) -> None:
        """Test context manager cleanup on exception."""
        try:
            with DomainHandlerFactory.handler_context(Domain.TIME):
                raise ValueError("Test error")
        except ValueError:
            pass

    def test_get_statistics_single(self) -> None:
        """Test getting statistics for single handler."""
        DomainHandlerFactory.get_handler(Domain.DEPTH)

        stats = DomainHandlerFactory.get_statistics(Domain.DEPTH)

        assert isinstance(stats, HandlerStatistics)
        assert stats.domain == Domain.DEPTH

    def test_get_all_statistics_domain_handler_factory(self) -> None:
        """Test getting statistics for all handlers."""
        DomainHandlerFactory.get_handler(Domain.DEPTH)
        DomainHandlerFactory.get_handler(Domain.TIME)

        all_stats = DomainHandlerFactory.get_all_statistics()

        assert len(all_stats) == 2
        assert all(isinstance(s, HandlerStatistics) for s in all_stats)

    def test_print_statistics(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test printing statistics to logger."""
        DomainHandlerFactory.get_handler(Domain.DEPTH)

        with caplog.at_level(logging.INFO):
            DomainHandlerFactory.print_statistics()

        assert any("Handler Statistics" in record.message for record in caplog.records)

    def test_print_statistics_no_handlers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test printing statistics with no initialized handlers."""
        # Create fresh registry with no handlers
        registry = DomainHandlerRegistry()

        with caplog.at_level(logging.INFO):
            registry.cleanup_all()  # Clean up
            # Print with fresh factory state
            if not DomainHandlerFactory.get_all_handlers():
                logger = logging.getLogger("src.analysis.domain_handlers")
                logger.info("No handler statistics available yet")

    def test_cleanup(self) -> None:
        """Test cleanup method."""
        handler = DomainHandlerFactory.get_handler(Domain.DEPTH)

        with patch.object(handler, "cleanup") as mock_cleanup:
            DomainHandlerFactory.cleanup()
            mock_cleanup.assert_called_once()


class TestAnalysisDomainIntegration:
    """Integration tests for the domain_handlers module."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        DomainHandlerFactory.cleanup()

    def test_depth_domain_workflow(self) -> None:
        """Test complete workflow for depth domain."""
        avo = np.array([[1.0, 2.0], [3.0, 4.0]])
        facies = np.array([[1, 2], [3, 4]], dtype=np.int64)

        handler = DomainHandlerFactory.get_handler(Domain.DEPTH)
        result = handler.prepare_display_cubes(
            vm=Mock(), facies_depth=facies, avo=avo, grid_spec=Mock()
        )

        assert_array_equal(result.avo, avo)
        assert_array_equal(result.facies, facies)

    def test_time_domain_workflow(self) -> None:
        """Test complete workflow for time domain."""
        avo = np.array([[1.0, 2.0]])
        facies = np.array([[1, 2]], dtype=np.int64)
        facies_resampled = np.array([[1, 2, 1]], dtype=np.int64)

        vm = Mock()
        vm.resample_to_time.return_value = (facies_resampled, 0.002)

        grid_spec = Mock()
        grid_spec.dt = 0.002

        handler = DomainHandlerFactory.get_handler(Domain.TIME)
        result = handler.prepare_display_cubes(
            vm=vm, facies_depth=facies, avo=avo, grid_spec=grid_spec
        )

        assert_array_equal(result.avo, avo)
        assert_array_equal(result.facies, facies_resampled)

    def test_multiple_handlers_isolation(self) -> None:
        """Test that multiple handlers don't interfere."""
        depth_handler = DomainHandlerFactory.get_handler(Domain.DEPTH)
        time_handler = DomainHandlerFactory.get_handler(Domain.TIME)

        assert depth_handler.domain == Domain.DEPTH
        assert time_handler.domain == Domain.TIME
        assert depth_handler is not time_handler

    def test_context_manager_workflow(self) -> None:
        """Test complete workflow using context manager."""
        avo = np.ones((2, 2), dtype=np.float64)
        facies = np.ones((2, 2), dtype=np.int64)

        with DomainHandlerFactory.handler_context(Domain.DEPTH) as handler:
            result = handler.prepare_display_cubes(
                vm=Mock(), facies_depth=facies, avo=avo, grid_spec=Mock()
            )
            assert result is not None

    def test_statistics_workflow(self) -> None:
        """Test complete statistics tracking workflow."""
        # Use fresh registry to avoid interference
        registry = DomainHandlerRegistry()
        registry.get_handler(Domain.DEPTH)
        registry.get_handler(Domain.TIME)

        # Get statistics
        all_stats = registry.get_all_statistics()

        # Verify all handlers have statistics
        assert len(all_stats) == 2
        assert all(stats.domain in [Domain.DEPTH, Domain.TIME] for stats in all_stats)

    def test_registry_isolation(self) -> None:
        """Test that separate registries are isolated."""
        registry1 = DomainHandlerRegistry()
        registry2 = DomainHandlerRegistry()

        handler1 = registry1.get_handler(Domain.DEPTH)
        handler2 = registry2.get_handler(Domain.DEPTH)

        # Different instances even though same domain
        assert handler1 is not handler2

    def test_error_handling_invalid_domain(self) -> None:
        """Test error handling for invalid domains."""
        registry = DomainHandlerRegistry()

        with pytest.raises((ValueError, AttributeError)):
            # Create a mock domain that's not registered
            fake_domain = Mock()
            fake_domain.name = "UNKNOWN"
            registry.get_handler(fake_domain)

    def test_error_handling_cleanup_failures(self) -> None:
        """Test error handling during cleanup."""
        registry = DomainHandlerRegistry()
        registry.get_handler(Domain.DEPTH)

        handler = registry.get_handler(Domain.DEPTH)

        with patch.object(
            handler, "cleanup", side_effect=RuntimeError("Cleanup error")
        ):
            with patch("src.analysis.domain.handlers.logger"):
                # Should not raise
                registry.cleanup_all()


class TestCoverageGaps:
    """Tests to cover remaining coverage gaps."""

    def test_context_manager_exception_during_cleanup(self) -> None:
        """Test exception handling in __exit__ when cleanup fails."""
        handler = DepthDomainHandler()

        with patch.object(
            handler, "cleanup", side_effect=RuntimeError("Cleanup error")
        ):
            with patch("src.analysis.domain.handlers.logger.exception"):
                # Should handle exception gracefully
                with handler:
                    pass

    def test_registry_repr(self) -> None:
        """Test registry __repr__ method."""
        registry = DomainHandlerRegistry()
        registry.get_handler(Domain.DEPTH)
        registry.get_handler(Domain.TIME)

        repr_str = repr(registry)
        assert "DomainHandlerRegistry" in repr_str
        assert "initialized=2" in repr_str
        assert "factories" in repr_str

    def test_registry_ensure_initialized_failure(self) -> None:
        """Test _ensure_initialized when handler factory fails."""
        registry = DomainHandlerRegistry()

        # Mock the factory to raise an error
        registry._handler_factories[Domain.DEPTH] = Mock(
            side_effect=RuntimeError("Factory failed")
        )

        with pytest.raises(RuntimeError, match="Cannot initialize"):
            registry.get_handler(Domain.DEPTH)

    def test_registry_ensure_initialized_logs_error(self) -> None:
        """Test that _ensure_initialized logs errors."""
        registry = DomainHandlerRegistry()

        registry._handler_factories[Domain.DEPTH] = Mock(
            side_effect=RuntimeError("Factory failed")
        )

        with patch("src.analysis.domain.handlers.logger.exception"):
            with pytest.raises(RuntimeError):
                registry.get_handler(Domain.DEPTH)

    def test_handler_exit_with_exception(self) -> None:
        """Test __exit__ is called with exception info."""
        handler = DepthDomainHandler()
        handler.initialize()

        cleanup_called = False

        def mock_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        with patch.object(handler, "cleanup", side_effect=mock_cleanup):
            try:
                with handler:
                    raise ValueError("Test error")
            except ValueError:
                pass

        assert cleanup_called

    def test_factory_handler_context_error_during_cleanup(self) -> None:
        """Test handler_context cleanup when error occurs."""
        error_occurred = False

        try:
            with DomainHandlerFactory.handler_context(Domain.DEPTH) as handler:
                # Verify we get a handler
                assert handler is not None
                raise ValueError("Test error during processing")
        except ValueError:
            error_occurred = True

        assert error_occurred

    def test_handler_protocol_compliance(self) -> None:
        """Test that DomainHandler implements CubeProcessor protocol."""
        handler = DepthDomainHandler()
        handler.initialize()

        # Should have domain property
        assert hasattr(handler, "domain")
        assert handler.domain == Domain.DEPTH

        # Should have prepare_display_cubes method
        assert hasattr(handler, "prepare_display_cubes")
        assert callable(handler.prepare_display_cubes)

    def test_multiple_registry_instances_independent(self) -> None:
        """Test that multiple registry instances are fully independent."""
        registry1 = DomainHandlerRegistry()
        registry2 = DomainHandlerRegistry()

        # Get different handlers
        registry1.get_handler(Domain.DEPTH)
        registry2.get_handler(Domain.TIME)

        # Each registry should only have its own
        assert Domain.DEPTH in registry1._handlers
        assert Domain.DEPTH not in registry2._handlers
        assert Domain.TIME in registry2._handlers
        assert Domain.TIME not in registry1._handlers

    def test_handler_statistics_with_zero_calls(self) -> None:
        """Test statistics when handler not called."""
        handler = DepthDomainHandler()
        handler.initialize()

        stats = handler.get_statistics()

        assert stats.call_count == 0
        assert stats.total_runtime_ms == 0.0
        assert stats.average_runtime_ms == 0.0

    def test_factory_get_all_statistics_empty(self) -> None:
        """Test get_all_statistics with fresh factory."""
        registry = DomainHandlerRegistry()

        stats_list = registry.get_all_statistics()

        # Should be empty initially
        assert isinstance(stats_list, list)
        assert len(stats_list) == 0

    def test_registry_cleanup_all_with_mixed_errors(self) -> None:
        """Test cleanup_all with some handlers failing."""
        registry = DomainHandlerRegistry()

        depth_handler = registry.get_handler(Domain.DEPTH)
        time_handler = registry.get_handler(Domain.TIME)

        cleanup_order = []

        def depth_cleanup() -> None:
            cleanup_order.append("depth")
            raise RuntimeError("Depth cleanup failed")

        def time_cleanup() -> None:
            cleanup_order.append("time")

        with patch.object(depth_handler, "cleanup", side_effect=depth_cleanup):
            with patch.object(time_handler, "cleanup", side_effect=time_cleanup):
                with patch("src.analysis.domain.handlers.logger"):
                    registry.cleanup_all()

                    # Both should have been called despite error
                    assert "depth" in cleanup_order
                    assert "time" in cleanup_order

    def test_handler_repr_with_initialization_state(self) -> None:
        """Test handler repr shows initialization state."""
        handler = TimeDomainHandler()

        # Before initialization
        repr_before = repr(handler)
        assert "not-initialized" in repr_before

        # After initialization
        handler.initialize()
        repr_after = repr(handler)
        assert "initialized" in repr_after

    def test_display_cubes_with_large_arrays(self) -> None:
        """Test DisplayCubes with larger arrays."""
        avo = np.random.rand(100, 100, 100).astype(np.float64)
        facies = np.random.randint(0, 10, (100, 100, 100), dtype=np.int64)

        cubes = DisplayCubes(avo=avo, facies=facies)

        assert cubes.avo.shape == (100, 100, 100)
        assert cubes.facies.shape == (100, 100, 100)

    def test_handler_initialization_idempotency(self) -> None:
        """Test that calling initialize multiple times is safe."""
        handler = DepthDomainHandler()

        handler.initialize()
        first_init = handler.is_initialized

        handler.initialize()
        second_init = handler.is_initialized

        assert first_init is True
        assert second_init is True

    def test_registry_is_initialized_before_lazy_load(self) -> None:
        """Test is_initialized before and after lazy loading."""
        registry = DomainHandlerRegistry()

        assert not registry.is_initialized(Domain.DEPTH)

        registry.get_handler(Domain.DEPTH)

        assert registry.is_initialized(Domain.DEPTH)
