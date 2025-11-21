"""Comprehensive tests for ProcessManager and ManagerHub facades."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from src.processing.managers.cache import CacheManager
from src.processing.managers.file import FileManager
from src.processing.managers.processor import ManagerHub, ProcessManager


# Concrete FileManager implementation for testing (FileManager is abstract)
class ConcreteFileManager(FileManager):
    """Concrete implementation of FileManager for testing."""

    def clear(self) -> int:
        """Clear files (no-op for testing)."""
        return 0

    def summarize(self) -> dict:
        """Summarize files (no-op for testing)."""
        return {}


class TestProcessManagerInitialization:
    """Test ProcessManager initialization."""

    def test_process_manager_basic_init(self):
        """Test basic ProcessManager initialization."""
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(file_manager=mock_file)
        assert pm.cache is not None
        assert pm.files is not None
        assert isinstance(pm.cache, CacheManager)
        assert pm.files is mock_file

    def test_process_manager_with_custom_cache(self):
        """Test ProcessManager with custom cache manager."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)
        assert pm.cache is mock_cache

    def test_process_manager_with_custom_file(self):
        """Test ProcessManager with custom file manager."""
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(file_manager=mock_file)
        assert pm.files is mock_file

    def test_process_manager_with_custom_logger(self):
        """Test ProcessManager with custom logger."""
        mock_logger = MagicMock(spec=logging.Logger)
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(logger=mock_logger, file_manager=mock_file)
        assert pm.logger is mock_logger

    def test_process_manager_creates_default_managers(self):
        """Test ProcessManager creates default managers when not provided."""
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(file_manager=mock_file)
        assert pm.cache is not None
        assert pm.files is not None

    def test_process_manager_has_logger(self):
        """Test ProcessManager has a logger."""
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(file_manager=mock_file)
        assert pm.logger is not None


class TestProcessManagerClear:
    """Test ProcessManager clear method delegation."""

    def test_clear_delegates_to_cache(self):
        """Test clear delegates to cache manager."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 5
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        result = pm.clear()

        mock_cache.clear.assert_called_once_with(
            patterns=None, cache_dir=None, prefix=""
        )
        assert result == 5

    def test_clear_with_patterns(self):
        """Test clear with glob patterns."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 3
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        patterns = ["*.pkl", "*.tmp"]
        result = pm.clear(patterns=patterns)

        mock_cache.clear.assert_called_once_with(
            patterns=patterns, cache_dir=None, prefix=""
        )
        assert result == 3

    def test_clear_with_cache_dir(self):
        """Test clear with custom cache directory."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 2
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        cache_dir = Path("/tmp/cache")
        result = pm.clear(cache_dir=cache_dir)

        mock_cache.clear.assert_called_once_with(
            patterns=None, cache_dir=cache_dir, prefix=""
        )
        assert result == 2

    def test_clear_with_prefix(self):
        """Test clear with log prefix."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 1
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        result = pm.clear(prefix="[CUSTOM]")

        mock_cache.clear.assert_called_once_with(
            patterns=None, cache_dir=None, prefix="[CUSTOM]"
        )
        assert result == 1

    def test_clear_with_all_args(self):
        """Test clear with all arguments."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 10
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        patterns = ["*.pkl"]
        cache_dir = Path("/tmp/cache")
        result = pm.clear(patterns=patterns, cache_dir=cache_dir, prefix="[TEST]")

        mock_cache.clear.assert_called_once_with(
            patterns=patterns, cache_dir=cache_dir, prefix="[TEST]"
        )
        assert result == 10


class TestProcessManagerOpenFile:
    """Test ProcessManager open_file method delegation."""

    def test_open_file_delegates_to_file_manager(self):
        """Test open_file delegates to file manager."""
        mock_file = MagicMock(spec=FileManager)
        mock_file.open.return_value = True
        pm = ProcessManager(file_manager=mock_file)

        result = pm.open_file("test.txt")

        mock_file.open.assert_called_once_with(filepath="test.txt", prefix="")
        assert result is True

    def test_open_file_with_description(self):
        """Test open_file with description."""
        mock_file = MagicMock(spec=FileManager)
        mock_file.open.return_value = True
        pm = ProcessManager(file_manager=mock_file)

        result = pm.open_file("test.txt")

        mock_file.open.assert_called_once_with(filepath="test.txt", prefix="")
        assert result is True

    def test_open_file_with_prefix(self):
        """Test open_file with prefix."""
        mock_file = MagicMock(spec=FileManager)
        mock_file.open.return_value = False
        pm = ProcessManager(file_manager=mock_file)

        result = pm.open_file("test.txt", prefix="[INFO]")

        mock_file.open.assert_called_once_with(filepath="test.txt", prefix="[INFO]")
        assert result is False

    def test_open_file_with_all_args(self):
        """Test open_file with all arguments."""
        mock_file = MagicMock(spec=FileManager)
        mock_file.open.return_value = True
        pm = ProcessManager(file_manager=mock_file)

        result = pm.open_file("test.txt", prefix="[DEBUG]")

        mock_file.open.assert_called_once_with(filepath="test.txt", prefix="[DEBUG]")
        assert result is True


class TestProcessManagerSummarize:
    """Test ProcessManager summarize method delegation."""

    def test_summarize_delegates_to_cache(self):
        """Test summarize delegates to cache manager."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        pm.summarize()

        mock_cache.summarize.assert_called_once_with(
            cache_dir=".cache", keys=None, prefix=""
        )

    def test_summarize_with_cache_dir(self):
        """Test summarize with custom cache directory."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        pm.summarize(cache_dir="/tmp/cache")

        mock_cache.summarize.assert_called_once_with(
            cache_dir="/tmp/cache", keys=None, prefix=""
        )

    def test_summarize_with_keys(self):
        """Test summarize with keys filter."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        keys = ["key1", "key2"]
        pm.summarize(keys=keys)

        mock_cache.summarize.assert_called_once_with(
            cache_dir=".cache", keys=keys, prefix=""
        )

    def test_summarize_with_prefix(self):
        """Test summarize with prefix."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        pm.summarize(prefix="[SUMMARY]")

        mock_cache.summarize.assert_called_once_with(
            cache_dir=".cache", keys=None, prefix="[SUMMARY]"
        )

    def test_summarize_with_all_args(self):
        """Test summarize with all arguments."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        keys = ["a", "b", "c"]
        pm.summarize(cache_dir="/tmp", keys=keys, prefix="[INFO]")

        mock_cache.summarize.assert_called_once_with(
            cache_dir="/tmp", keys=keys, prefix="[INFO]"
        )


class TestManagerHubInitialization:
    """Test ManagerHub initialization."""

    def test_manager_hub_basic_init(self):
        """Test basic ManagerHub initialization."""
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(file_manager=mock_file)
        assert hub.cache is not None
        assert hub.files is not None
        assert hub.processes is not None
        assert isinstance(hub.cache, CacheManager)
        assert hub.files is mock_file
        assert isinstance(hub.processes, ProcessManager)

    def test_manager_hub_with_custom_logger(self):
        """Test ManagerHub with custom logger."""
        mock_logger = MagicMock(spec=logging.Logger)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(logger=mock_logger, file_manager=mock_file)
        assert hub.logger is mock_logger

    def test_manager_hub_shares_managers_with_process_manager(self):
        """Test ManagerHub shares managers with ProcessManager."""
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(file_manager=mock_file)
        # ProcessManager should use the same cache and file managers as hub
        assert hub.processes.cache is hub.cache
        assert hub.processes.files is hub.files

    def test_manager_hub_creates_all_managers(self):
        """Test ManagerHub creates all manager types."""
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(file_manager=mock_file)
        assert hub.cache is not None
        assert hub.files is not None
        assert hub.processes is not None

    def test_manager_hub_with_custom_cache(self):
        """Test ManagerHub with custom cache manager."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)
        assert hub.cache is mock_cache

    def test_manager_hub_with_custom_file(self):
        """Test ManagerHub with custom file manager."""
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(file_manager=mock_file)
        assert hub.files is mock_file

    def test_manager_hub_with_custom_process_manager(self):
        """Test ManagerHub with custom ProcessManager."""
        mock_pm = MagicMock(spec=ProcessManager)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(process_manager=mock_pm, file_manager=mock_file)
        assert hub.processes is mock_pm


class TestManagerHubClear:
    """Test ManagerHub clear method."""

    def test_hub_clear_delegates_to_cache(self):
        """Test hub clear delegates to cache manager."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 7
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        result = hub.clear()

        mock_cache.clear.assert_called_once_with(
            patterns=None, cache_dir=None, prefix=""
        )
        assert result == 7

    def test_hub_clear_with_patterns(self):
        """Test hub clear with patterns."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 4
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        patterns = ["*.pkl"]
        result = hub.clear(patterns=patterns)

        mock_cache.clear.assert_called_once_with(
            patterns=patterns, cache_dir=None, prefix=""
        )
        assert result == 4

    def test_hub_clear_with_all_args(self):
        """Test hub clear with all arguments."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 6
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        patterns = ["*.tmp"]
        cache_dir = Path("/tmp")
        result = hub.clear(patterns=patterns, cache_dir=cache_dir, prefix="[HUB]")

        mock_cache.clear.assert_called_once_with(
            patterns=patterns, cache_dir=cache_dir, prefix="[HUB]"
        )
        assert result == 6


class TestManagerHubSummarize:
    """Test ManagerHub summarize method."""

    def test_hub_summarize_delegates_to_cache(self):
        """Test hub summarize delegates to cache manager."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        hub.summarize()

        mock_cache.summarize.assert_called_once_with(
            cache_dir=".cache", keys=None, prefix=""
        )

    def test_hub_summarize_with_cache_dir(self):
        """Test hub summarize with custom cache directory."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        hub.summarize(cache_dir="/tmp/cache")

        mock_cache.summarize.assert_called_once_with(
            cache_dir="/tmp/cache", keys=None, prefix=""
        )

    def test_hub_summarize_with_keys_and_prefix(self):
        """Test hub summarize with keys and prefix."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        keys = ["key1", "key2"]
        hub.summarize(keys=keys, prefix="[TEST]")

        mock_cache.summarize.assert_called_once_with(
            cache_dir=".cache", keys=keys, prefix="[TEST]"
        )

    def test_hub_summarize_with_all_args(self):
        """Test hub summarize with all arguments."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        keys = ["a", "b"]
        hub.summarize(cache_dir="/tmp", keys=keys, prefix="[INFO]")

        mock_cache.summarize.assert_called_once_with(
            cache_dir="/tmp", keys=keys, prefix="[INFO]"
        )


class TestProcessManagerIntegration:
    """Integration tests for ProcessManager."""

    def test_process_manager_workflow(self):
        """Test complete ProcessManager workflow."""
        mock_file = MagicMock(spec=FileManager)
        mock_file.open.return_value = True
        pm = ProcessManager(file_manager=mock_file)

        # All methods should be callable
        result_clear = pm.clear()
        assert isinstance(result_clear, int)

        result_open = pm.open_file("test.txt")
        assert isinstance(result_open, bool)

        pm.summarize()  # Should not raise

    def test_process_manager_with_mixed_managers(self):
        """Test ProcessManager with mix of default and custom managers."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 3
        mock_file = MagicMock(spec=FileManager)

        pm = ProcessManager(cache_manager=mock_cache, file_manager=mock_file)

        assert pm.cache is mock_cache
        assert pm.files is mock_file

        result = pm.clear()
        assert result == 3


class TestManagerHubIntegration:
    """Integration tests for ManagerHub."""

    def test_manager_hub_workflow(self):
        """Test complete ManagerHub workflow."""
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(file_manager=mock_file)

        # All methods should be callable
        result_clear = hub.clear()
        assert isinstance(result_clear, int)

        hub.summarize()  # Should not raise

        # Process manager should be accessible
        assert hub.processes is not None

    def test_manager_hub_manager_coordination(self):
        """Test managers coordinate through hub."""
        mock_cache = MagicMock(spec=CacheManager)
        mock_cache.clear.return_value = 5
        mock_file = MagicMock(spec=FileManager)

        hub = ManagerHub(cache_manager=mock_cache, file_manager=mock_file)

        # Both managers should be accessible via hub
        assert hub.cache is mock_cache
        assert hub.files is mock_file

        # Process manager should reference the same instances
        assert hub.processes.cache is mock_cache
        assert hub.processes.files is mock_file

    def test_manager_hub_custom_process_manager(self):
        """Test ManagerHub with custom ProcessManager."""
        mock_pm = MagicMock(spec=ProcessManager)
        mock_file = MagicMock(spec=FileManager)
        hub = ManagerHub(process_manager=mock_pm, file_manager=mock_file)

        assert hub.processes is mock_pm
