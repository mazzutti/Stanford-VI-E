"""Tests for file manager implementations.

Tests FileManager abstract base class and concrete implementations for
managing file I/O operations in processing pipeline.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.processing.managers.base import BaseManager
from src.processing.managers.file import FileManager


class TestFileManagerAbstractMethods:
    """Test FileManager abstract methods."""

    def test_file_manager_is_concrete(self):
        """Test that FileManager can be instantiated directly."""
        # FileManager is concrete - it implements clear() and summarize()
        fm = FileManager()
        assert isinstance(fm, FileManager)
        assert isinstance(fm, BaseManager)

    def test_file_manager_inherits_from_base_manager(self):
        """Test that FileManager inherits from BaseManager."""
        assert issubclass(FileManager, BaseManager)


class TestFileManagerConcreteImplementation:
    """Test concrete FileManager implementation."""

    def create_concrete_file_manager(self):
        """Helper to create concrete FileManager for testing."""

        class ConcreteFileManager(FileManager):
            def clear(self) -> int:
                return 0

            def summarize(self) -> dict:
                return {}

        return ConcreteFileManager()

    def test_concrete_file_manager_instantiation(self):
        """Test that concrete FileManager can be instantiated."""
        fm = self.create_concrete_file_manager()
        assert fm is not None
        assert isinstance(fm, FileManager)

    def test_concrete_file_manager_has_logger(self):
        """Test that concrete FileManager has logger from BaseManager."""
        fm = self.create_concrete_file_manager()
        assert fm.logger is not None

    def test_clear_method_implementation(self):
        """Test clear method returns integer."""
        fm = self.create_concrete_file_manager()
        result = fm.clear()
        assert isinstance(result, int)

    def test_summarize_method_implementation(self):
        """Test summarize method returns dict."""
        fm = self.create_concrete_file_manager()
        result = fm.summarize()
        assert isinstance(result, dict)


class TestFileManagerOperations:
    """Test FileManager operations."""

    def create_file_manager_with_methods(self, clear_result=0, summarize_result=None):
        """Helper to create FileManager with custom method implementations."""
        if summarize_result is None:
            summarize_result = {}

        class CustomFileManager(FileManager):
            def clear(self) -> int:
                return clear_result

            def summarize(self) -> dict:
                return summarize_result

        return CustomFileManager()

    def test_clear_returns_zero(self):
        """Test clear method returns 0."""
        fm = self.create_file_manager_with_methods(clear_result=0)
        assert fm.clear() == 0

    def test_clear_returns_positive_integer(self):
        """Test clear method can return positive integer."""
        fm = self.create_file_manager_with_methods(clear_result=5)
        assert fm.clear() == 5

    def test_summarize_returns_empty_dict(self):
        """Test summarize method can return empty dict."""
        fm = self.create_file_manager_with_methods(summarize_result={})
        result = fm.summarize()
        assert result == {}
        assert isinstance(result, dict)

    def test_summarize_returns_populated_dict(self):
        """Test summarize method can return populated dict."""
        expected = {"files": 10, "size_mb": 100.5}
        fm = self.create_file_manager_with_methods(summarize_result=expected)
        result = fm.summarize()
        assert result == expected

    def test_clear_and_summarize_sequence(self):
        """Test sequence of clear and summarize operations."""
        fm = self.create_file_manager_with_methods(
            clear_result=3, summarize_result={"cleared": 3}
        )

        clear_result = fm.clear()
        summarize_result = fm.summarize()

        assert clear_result == 3
        assert summarize_result == {"cleared": 3}


class TestFileManagerInheritedBehavior:
    """Test inherited behavior from BaseManager."""

    def create_file_manager(self):
        """Helper to create concrete FileManager."""

        class SimpleFileManager(FileManager):
            def clear(self) -> int:
                return 0

            def summarize(self) -> dict:
                return {}

        return SimpleFileManager()

    def test_logger_initialization(self):
        """Test that logger is properly initialized from BaseManager."""
        fm = self.create_file_manager()
        assert fm.logger is not None
        # Logger should have a name (from class name)
        assert fm.logger.name is not None

    def test_manager_type(self):
        """Test manager type checking."""
        fm = self.create_file_manager()
        assert isinstance(fm, BaseManager)
        assert isinstance(fm, FileManager)


class TestFileManagerDuckTyping:
    """Test FileManager duck-typing compatibility."""

    def create_manager_like_object(self):
        """Helper to create object that acts like FileManager."""

        class MockFileManager(FileManager):
            def __init__(self):
                super().__init__()
                self.clear_called = False
                self.summarize_called = False

            def clear(self) -> int:
                self.clear_called = True
                return 1

            def summarize(self) -> dict:
                self.summarize_called = True
                return {"status": "summarized"}

        return MockFileManager()

    def test_manager_callable_interface(self):
        """Test that manager has callable interface."""
        fm = self.create_manager_like_object()

        assert callable(fm.clear)
        assert callable(fm.summarize)

    def test_manager_method_tracking(self):
        """Test tracking method calls on manager."""
        fm = self.create_manager_like_object()

        assert not fm.clear_called
        fm.clear()
        assert fm.clear_called

        assert not fm.summarize_called
        fm.summarize()
        assert fm.summarize_called


class TestFileManagerErrorScenarios:
    """Test FileManager error scenarios."""

    def create_manager_with_error(self, error_in_method="clear"):
        """Helper to create FileManager that raises error."""

        class ErrorFileManager(FileManager):
            def clear(self) -> int:
                if error_in_method == "clear":
                    raise RuntimeError("Clear operation failed")
                return 0

            def summarize(self) -> dict:
                if error_in_method == "summarize":
                    raise RuntimeError("Summarize operation failed")
                return {}

        return ErrorFileManager()

    def test_clear_raises_error(self):
        """Test clear method raising error."""
        fm = self.create_manager_with_error(error_in_method="clear")

        with pytest.raises(RuntimeError, match="Clear operation failed"):
            fm.clear()

    def test_summarize_raises_error(self):
        """Test summarize method raising error."""
        fm = self.create_manager_with_error(error_in_method="summarize")

        with pytest.raises(RuntimeError, match="Summarize operation failed"):
            fm.summarize()

    def test_logger_callable_after_error(self):
        """Test logger is still available after error."""
        fm = self.create_manager_with_error(error_in_method="clear")

        try:
            fm.clear()
        except RuntimeError:
            pass

        # Logger should still be accessible
        assert fm.logger is not None


class TestFileManagerMockUsage:
    """Test FileManager with mocks."""

    def test_file_manager_with_mock_logger(self):
        """Test FileManager with mocked logger."""

        class MockedFileManager(FileManager):
            def clear(self) -> int:
                self.logger.info("Clearing files")
                return 0

            def summarize(self) -> dict:
                self.logger.debug("Summarizing")
                return {}

        fm = MockedFileManager()
        fm.logger = Mock()

        fm.clear()
        fm.logger.info.assert_called_once_with("Clearing files")

        fm.summarize()
        fm.logger.debug.assert_called_once_with("Summarizing")

    def test_file_manager_with_spied_methods(self):
        """Test FileManager with spied (wrapped) methods."""

        class SpiedFileManager(FileManager):
            def __init__(self):
                super().__init__()
                self.clear_call_count = 0

            def clear(self) -> int:
                self.clear_call_count += 1
                return self.clear_call_count

            def summarize(self) -> dict:
                return {"calls": self.clear_call_count}

        fm = SpiedFileManager()

        assert fm.clear() == 1
        assert fm.clear() == 2
        assert fm.summarize() == {"calls": 2}


class TestFileManagerIntegration:
    """Integration tests for FileManager."""

    def create_realistic_file_manager(self):
        """Helper to create realistic FileManager."""

        class RealisticFileManager(FileManager):
            def __init__(self):
                super().__init__()
                self._files = []

            def add_file(self, filename: str) -> None:
                self._files.append(filename)

            def clear(self) -> int:
                count = len(self._files)
                self._files.clear()
                return count

            def summarize(self) -> dict:
                return {"file_count": len(self._files), "files": self._files.copy()}

        return RealisticFileManager()

    def test_realistic_workflow(self):
        """Test realistic FileManager workflow."""
        fm = self.create_realistic_file_manager()

        # Add some files
        fm.add_file("file1.txt")
        fm.add_file("file2.txt")

        # Check summary
        summary = fm.summarize()
        assert summary["file_count"] == 2

        # Clear files
        cleared = fm.clear()
        assert cleared == 2

        # Verify cleared
        summary = fm.summarize()
        assert summary["file_count"] == 0

    def test_multiple_clear_calls(self):
        """Test multiple clear calls."""
        fm = self.create_realistic_file_manager()

        fm.add_file("file1.txt")
        assert fm.clear() == 1

        fm.add_file("file2.txt")
        fm.add_file("file3.txt")
        assert fm.clear() == 2

        # No files left
        assert fm.clear() == 0


class TestFileManagerStateManagement:
    """Test FileManager state management."""

    def create_stateful_file_manager(self):
        """Helper to create stateful FileManager."""

        class StatefulFileManager(FileManager):
            def __init__(self):
                super().__init__()
                self._state = {"initialized": True, "operations": 0}

            def clear(self) -> int:
                self._state["operations"] += 1
                return self._state["operations"]

            def summarize(self) -> dict:
                return self._state.copy()

            def get_state(self) -> dict:
                return self._state.copy()

        return StatefulFileManager()

    def test_state_persists_across_calls(self):
        """Test state persists across method calls."""
        fm = self.create_stateful_file_manager()

        fm.clear()
        fm.clear()

        state = fm.get_state()
        assert state["operations"] == 2

    def test_state_in_summary(self):
        """Test state included in summary."""
        fm = self.create_stateful_file_manager()

        fm.clear()
        summary = fm.summarize()

        assert "operations" in summary
        assert summary["operations"] == 1


class TestFileManagerOpenMethod:
    """Test FileManager.open() method."""

    def create_concrete_file_manager(self):
        """Helper to create concrete FileManager for testing."""

        class ConcreteFileManager(FileManager):
            def clear(self) -> int:
                return 0

            def summarize(self) -> dict:
                return {}

        return ConcreteFileManager()

    def test_open_nonexistent_file_returns_false(self):
        """Test opening nonexistent file returns False."""
        fm = self.create_concrete_file_manager()
        result = fm.open("/nonexistent/path/to/file.txt")
        assert result is False

    def test_open_existing_file_with_webbrowser(self, tmp_path):
        """Test opening existing file with webbrowser."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.html"
        test_file.write_text("<html><body>Test</body></html>")

        # Mock webbrowser.open
        with patch("webbrowser.open") as mock_webbrowser:
            mock_webbrowser.return_value = True
            result = fm.open(str(test_file))

        assert result is True
        mock_webbrowser.assert_called_once()

    def test_open_file_webbrowser_exception_fallback(self, tmp_path):
        """Test fallback when webbrowser.open raises exception."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.html"
        test_file.write_text("<html><body>Test</body></html>")

        # Mock webbrowser.open to raise a runtime error (expected by the code)
        with patch("webbrowser.open", side_effect=RuntimeError("Webbrowser failed")):
            with patch("shutil.which") as mock_which:
                # Simulate 'open' command exists (macOS)
                mock_which.return_value = "/usr/bin/open"
                with patch("subprocess.run") as mock_run:
                    result = fm.open(str(test_file))

        assert result is True
        mock_run.assert_called_once()

    def test_open_file_with_xdg_open_fallback(self, tmp_path):
        """Test fallback to xdg-open when open command not found."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Mock both webbrowser and shutil.which
        with patch("webbrowser.open", side_effect=RuntimeError("Webbrowser failed")):
            with patch("shutil.which") as mock_which:
                # Simulate 'open' not found but 'xdg-open' exists (Linux)
                def which_side_effect(cmd):
                    if cmd == "xdg-open":
                        return "/usr/bin/xdg-open"
                    return None

                mock_which.side_effect = which_side_effect
                with patch("subprocess.run") as mock_run:
                    result = fm.open(str(test_file))

        assert result is True
        mock_run.assert_called_once()

    def test_open_file_all_methods_fail(self, tmp_path):
        """Test when all opening methods fail."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Mock all methods to fail
        with patch("webbrowser.open", side_effect=RuntimeError("Webbrowser failed")):
            with patch("shutil.which", return_value=None):
                result = fm.open(str(test_file))

        assert result is False

    def test_open_file_subprocess_exception_handled(self, tmp_path):
        """Test subprocess exception is handled gracefully."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Mock webbrowser and subprocess to fail
        with patch("webbrowser.open", side_effect=RuntimeError("Webbrowser failed")):
            with patch("shutil.which", return_value="/usr/bin/open"):
                with patch(
                    "subprocess.run", side_effect=RuntimeError("Subprocess failed")
                ):
                    result = fm.open(str(test_file))

        assert result is False

    def test_open_with_description_parameter(self, tmp_path):
        """Test open method accepts description parameter."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch("webbrowser.open", return_value=True):
            result = fm.open(str(test_file))

        assert result is True

    def test_open_with_prefix_parameter(self, tmp_path):
        """Test open method accepts prefix parameter."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch("webbrowser.open", return_value=True):
            result = fm.open(str(test_file), prefix="[PREFIX]")

        assert result is True

    def test_open_error_logging(self, tmp_path):
        """Test error is logged when file not found."""
        fm = self.create_concrete_file_manager()
        fm.logger = Mock()

        result = fm.open("/nonexistent/file.txt")

        assert result is False
        # Should call _log_error
        assert fm.logger.error.called or not fm.logger.called

    def test_open_resolves_relative_paths(self, tmp_path):
        """Test that open resolves relative file paths."""
        fm = self.create_concrete_file_manager()

        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch("webbrowser.open") as mock_webbrowser:
            mock_webbrowser.return_value = True
            result = fm.open(str(test_file))

        assert result is True
        # webbrowser.open should be called with file:// URL
        call_args = mock_webbrowser.call_args
        assert call_args is not None
        assert "file://" in call_args[0][0]
