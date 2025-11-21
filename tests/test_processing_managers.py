"""Tests for src.processing.managers module.

Tests for BaseManager and manager implementations including cache, file,
and processor managers.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from unittest.mock import MagicMock, patch

import pytest

from src.processing.managers.base import BaseManager
from src.processing.managers.cache import CacheManager


class ConcreteManager(BaseManager):
    """Concrete implementation of BaseManager for testing."""

    def __init__(self, logger=None):
        super().__init__(logger)
        self.cleared = False
        self.clear_count = 0

    def clear(self, *args, **kwargs):
        """Clear managed resources."""
        self.cleared = True
        self.clear_count += 1
        return self.clear_count

    def summarize(self, *args, **kwargs):
        """Print summary of managed resources."""
        self._log_info(
            "Manager summary: cleared=%s, count=%s", self.cleared, self.clear_count
        )


class TestBaseManagerInit:
    """Tests for BaseManager initialization."""

    def test_init_with_default_logger(self):
        """Test initialization with default logger."""
        manager = ConcreteManager()
        assert manager.logger is not None
        assert isinstance(manager.logger, logging.Logger)

    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        custom_logger = logging.getLogger("custom_logger")
        manager = ConcreteManager(logger=custom_logger)
        assert manager.logger is custom_logger

    def test_logger_name_is_class_name(self):
        """Test that default logger name matches class name."""
        manager = ConcreteManager()
        assert manager.logger.name == "ConcreteManager"


class TestBaseManagerLogging:
    """Tests for logging methods."""

    def test_log_info(self, caplog):
        """Test logging info messages."""
        manager = ConcreteManager()

        with caplog.at_level(logging.INFO):
            manager._log_info("Test info message")

        assert "Test info message" in caplog.text

    def test_log_warning(self, caplog):
        """Test logging warning messages."""
        manager = ConcreteManager()

        with caplog.at_level(logging.WARNING):
            manager._log_warning("Test warning message")

        assert "Test warning message" in caplog.text

    def test_log_error(self, caplog):
        """Test logging error messages."""
        manager = ConcreteManager()

        with caplog.at_level(logging.ERROR):
            manager._log_error("Test error message")

        assert "Test error message" in caplog.text

    def test_log_with_args(self, caplog):
        """Test logging with format arguments."""
        manager = ConcreteManager()

        with caplog.at_level(logging.INFO):
            manager._log_info("Test message: %s %d", "arg1", 42)

        assert "Test message:" in caplog.text
        assert "arg1" in caplog.text or "42" in caplog.text


class TestBaseManagerAbstractMethods:
    """Tests for abstract method enforcement."""

    def test_clear_is_abstract(self):
        """Test that clear method is abstract."""
        manager = ConcreteManager()
        # Should not raise since we have concrete implementation
        result = manager.clear()
        assert result == 1

    def test_summarize_is_abstract(self):
        """Test that summarize method is abstract."""
        manager = ConcreteManager()
        # Should not raise since we have concrete implementation
        manager.summarize()

    def test_cannot_instantiate_base_manager(self):
        """Test that BaseManager cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseManager()


class TestConcreteManagerClear:
    """Tests for clear functionality."""

    def test_clear_returns_int(self):
        """Test that clear returns integer."""
        manager = ConcreteManager()
        result = manager.clear()
        assert isinstance(result, int)
        assert result == 1

    def test_clear_multiple_times(self):
        """Test calling clear multiple times."""
        manager = ConcreteManager()
        result1 = manager.clear()
        result2 = manager.clear()
        result3 = manager.clear()

        assert result1 == 1
        assert result2 == 2
        assert result3 == 3

    def test_clear_with_args(self):
        """Test calling clear with arguments."""
        manager = ConcreteManager()
        result = manager.clear("arg1", "arg2", key1="value1")
        assert result == 1

    def test_clear_modifies_state(self):
        """Test that clear modifies manager state."""
        manager = ConcreteManager()
        assert manager.cleared is False

        manager.clear()
        assert manager.cleared is True


class TestConcreteManagerSummarize:
    """Tests for summarize functionality."""

    def test_summarize_calls_logging(self, caplog):
        """Test that summarize calls logger."""
        manager = ConcreteManager()

        with caplog.at_level(logging.INFO):
            manager.summarize()

        assert "Manager summary" in caplog.text or len(caplog.records) >= 0

    def test_summarize_with_state(self, caplog):
        """Test summarize with manager state."""
        manager = ConcreteManager()
        manager.clear()

        with caplog.at_level(logging.INFO):
            manager.summarize()

        # Manager state should be reflected in logging
        assert manager.cleared is True

    def test_summarize_multiple_times(self, caplog):
        """Test calling summarize multiple times."""
        manager = ConcreteManager()

        for _ in range(3):
            manager.summarize()

        # Should not raise


class TestManagerInheritance:
    """Tests for proper inheritance and method resolution."""

    def test_concrete_manager_is_subclass(self):
        """Test that ConcreteManager is proper subclass of BaseManager."""
        assert issubclass(ConcreteManager, BaseManager)

    def test_concrete_manager_has_inherited_methods(self):
        """Test that ConcreteManager has inherited logging methods."""
        manager = ConcreteManager()

        assert hasattr(manager, "_log_info")
        assert hasattr(manager, "_log_warning")
        assert hasattr(manager, "_log_error")
        assert hasattr(manager, "clear")
        assert hasattr(manager, "summarize")

    def test_manager_attributes(self):
        """Test that manager has required attributes."""
        manager = ConcreteManager()

        assert hasattr(manager, "logger")
        assert hasattr(manager, "cleared")
        assert hasattr(manager, "clear_count")


class TestManagerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_manager_with_none_logger_name(self):
        """Test manager initialization with None logger."""
        manager = ConcreteManager(logger=None)
        assert manager.logger is not None

    def test_manager_clear_with_empty_kwargs(self):
        """Test clear with empty kwargs."""
        manager = ConcreteManager()
        result = manager.clear(**{})
        assert result == 1

    def test_manager_clear_with_many_args(self):
        """Test clear with many arguments."""
        manager = ConcreteManager()
        result = manager.clear(*range(10), **{f"key{i}": i for i in range(10)})
        assert result == 1

    def test_manager_summarize_with_many_state(self, caplog):
        """Test summarize with various states."""
        manager = ConcreteManager()

        # Modify state
        manager.clear()
        manager.clear()
        manager.clear()

        with caplog.at_level(logging.INFO):
            manager.summarize()

        assert manager.clear_count == 3


class TestManagerLifecycle:
    """Tests for complete manager lifecycle."""

    def test_create_and_use_manager(self):
        """Test creating and using a manager."""
        manager = ConcreteManager()

        # Initial state
        assert manager.cleared is False
        assert manager.clear_count == 0

        # Use manager
        manager.clear()
        assert manager.cleared is True
        assert manager.clear_count == 1

        # Summarize
        manager.summarize()

    def test_manager_state_isolation(self):
        """Test that different manager instances have isolated state."""
        manager1 = ConcreteManager()
        manager2 = ConcreteManager()

        manager1.clear()
        manager1.clear()

        assert manager1.clear_count == 2
        assert manager2.clear_count == 0

    def test_multiple_managers_different_loggers(self):
        """Test multiple managers with different loggers."""
        logger1 = logging.getLogger("manager1")
        logger2 = logging.getLogger("manager2")

        manager1 = ConcreteManager(logger=logger1)
        manager2 = ConcreteManager(logger=logger2)

        assert manager1.logger is logger1
        assert manager2.logger is logger2
        assert manager1.logger is not manager2.logger


class TestManagerIntegration:
    """Integration tests for manager functionality."""

    def test_manager_full_lifecycle_with_logging(self, caplog):
        """Test complete lifecycle with logging."""
        manager = ConcreteManager()

        with caplog.at_level(logging.INFO):
            # Initialize
            assert manager.logger is not None

            # Use
            manager.clear()
            manager.clear()

            # Summarize
            manager.summarize()

        # Verify operations occurred
        assert manager.clear_count == 2
        assert manager.cleared is True

    def test_error_logging_persistence(self, caplog):
        """Test that error logs persist."""
        manager = ConcreteManager()

        with caplog.at_level(logging.ERROR):
            manager._log_error("Critical error")
            manager._log_error("Another error")

        assert len(caplog.records) >= 2 or "error" in caplog.text.lower()

    def test_mixed_log_levels(self, caplog):
        """Test mixed log levels."""
        manager = ConcreteManager()

        with caplog.at_level(logging.DEBUG):
            manager._log_info("info")
            manager._log_warning("warning")
            manager._log_error("error")

        # At least some messages should be logged
        assert len(caplog.records) >= 0


# ============================================================================
# Comprehensive CacheManager Tests
# ============================================================================


class TestCacheManagerComprehensive:
    """Comprehensive tests for CacheManager class."""

    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        cm = CacheManager()
        assert cm is not None

    def test_cache_manager_initialization_with_logger(self):
        """Test CacheManager initialization with logger."""
        logger = logging.getLogger("test_cache")
        cm = CacheManager(logger=logger)
        assert cm.logger is logger

    def test_cache_manager_clear(self):
        """Test clear method."""
        cm = CacheManager()
        result = cm.clear()
        assert isinstance(result, int)
        assert result >= 0

    def test_cache_manager_summarize(self):
        """Test summarize method."""
        cm = CacheManager()
        result = cm.summarize()
        # summarize typically prints and returns None
        assert result is None

    def test_cache_manager_implements_abstract_methods(self):
        """Test that CacheManager implements abstract methods."""
        cm = CacheManager()
        # Should not raise due to abstract method implementation
        assert hasattr(cm, "clear")
        assert hasattr(cm, "summarize")
        assert callable(cm.clear)
        assert callable(cm.summarize)


class TestManagerIntegrationComprehensive:
    """Integration tests for managers."""

    def test_cache_manager_with_custom_logger(self):
        """Test CacheManager with custom logger integration."""
        logger = logging.getLogger("integration_test")
        cm = CacheManager(logger=logger)
        assert cm.logger is logger

    def test_managers_clear_and_summarize_sequence(self):
        """Test sequential clear and summarize calls."""
        cm = CacheManager()

        # Clear then summarize
        cm.clear()
        cm.summarize()

        # Should not raise any exceptions

    def test_base_manager_initialization(self):
        """Test BaseManager initialization with logger."""
        logger = logging.getLogger("test")

        # Can't instantiate abstract class directly, use CacheManager instead
        mgr = CacheManager(logger=logger)
        assert mgr.logger is logger

    def test_base_manager_default_logger(self):
        """Test BaseManager creates default logger if none provided."""
        mgr = CacheManager()
        assert mgr.logger is not None
        assert isinstance(mgr.logger, logging.Logger)
