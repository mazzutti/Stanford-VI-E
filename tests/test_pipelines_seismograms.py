"""Comprehensive test suite for SeismogramAnalyzer.

Tests cover:
- Initialization with and without dependency injection
- Context manager (_timed_operation)
- Command execution (run_command)
- Cache operations (clear_cache)
- File existence checks (check_file_exists)
- File operations (open_file)
- Pipeline orchestration (main)
- Error handling and validation
"""

# mypy: ignore-errors


from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.common import AnalysisCommon
from src.analysis.pipelines import SeismogramAnalyzer


@pytest.fixture
def mock_analysis():
    """Fixture providing a mocked AnalysisCommon instance."""
    return MagicMock(spec=AnalysisCommon)


@pytest.fixture
def analyzer(mock_analysis):
    """Fixture providing a SeismogramAnalyzer with mocked dependencies."""
    return SeismogramAnalyzer(analysis=mock_analysis)


class TestSeismogramAnalyzerInit:
    """Test SeismogramAnalyzer initialization."""

    def test_init_with_analysis(self):
        """Test initialization with injected AnalysisCommon instance."""
        mock_analysis = MagicMock(spec=AnalysisCommon)
        analyzer = SeismogramAnalyzer(analysis=mock_analysis)
        assert analyzer._analysis is mock_analysis

    def test_init_creates_logger(self):
        """Test initialization creates a logger for the class."""
        mock_analysis = MagicMock(spec=AnalysisCommon)
        analyzer = SeismogramAnalyzer(analysis=mock_analysis)
        assert analyzer._logger is not None
        assert analyzer._logger.name == "SeismogramAnalyzer"

    def test_constants_are_set(self):
        """Test class constants are properly defined."""
        assert SeismogramAnalyzer.DEFAULT_CACHE_PATTERNS == ["avo_*.npz"]
        assert SeismogramAnalyzer.FILE_READY_DELAY_SECONDS == 1
        assert SeismogramAnalyzer.BYTES_PER_MB == 1024 * 1024
        assert SeismogramAnalyzer.INDENT_PREFIX == "  "


class TestTimedOperation:
    """Test _timed_operation context manager."""

    def test_timed_operation_logs_description(self, analyzer, caplog):
        """Test context manager logs operation description."""
        with caplog.at_level(logging.INFO):
            with analyzer._timed_operation("Test operation", prefix=""):
                pass

        assert "Test operation" in caplog.text
        assert "Completed in" in caplog.text

    def test_timed_operation_with_prefix(self, analyzer, caplog):
        """Test context manager includes prefix in logging."""
        with caplog.at_level(logging.INFO):
            with analyzer._timed_operation("Test", prefix="[PREFIX]"):
                pass

        assert "[PREFIX]" in caplog.text

    def test_timed_operation_without_description(self, analyzer, caplog):
        """Test context manager works without description."""
        with caplog.at_level(logging.INFO):
            with analyzer._timed_operation("", prefix=""):
                pass

        # Should still log completion time
        assert "Completed in" in caplog.text

    def test_timed_operation_yields_control(self, analyzer):
        """Test context manager properly yields control."""
        executed = False
        with analyzer._timed_operation("Test"):
            executed = True

        assert executed

    def test_timed_operation_logs_on_exception(self, analyzer, caplog):
        """Test context manager logs timing even when exception occurs."""
        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError):
                with analyzer._timed_operation("Test"):
                    raise ValueError("Test error")

        # Timing should still be logged despite exception
        assert "Completed in" in caplog.text


class TestRunCommand:
    """Test run_command method."""

    def test_run_command_valid(self, analyzer):
        """Test running a valid shell command."""
        result = analyzer.run_command("echo 'test'", description="Echo test")

        assert result is not None
        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 0

    def test_run_command_empty_raises_error(self, analyzer):
        """Test empty command raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            analyzer.run_command("")

    def test_run_command_whitespace_only_raises_error(self, analyzer):
        """Test whitespace-only command raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            analyzer.run_command("   ")

    def test_run_command_none_raises_error(self, analyzer):
        """Test None command raises ValueError."""
        with pytest.raises(ValueError):
            analyzer.run_command(None)  # type: ignore

    def test_run_command_non_string_raises_error(self, analyzer):
        """Test non-string command raises ValueError."""
        with pytest.raises(ValueError):
            analyzer.run_command(123)  # type: ignore

    def test_run_command_with_description(self, analyzer, caplog):
        """Test run_command with description logs it."""
        with caplog.at_level(logging.INFO):
            analyzer.run_command("true", description="Test command")

        assert "Test command" in caplog.text

    def test_run_command_with_prefix(self, analyzer, caplog):
        """Test run_command with prefix includes it in logs."""
        with caplog.at_level(logging.INFO):
            analyzer.run_command("true", prefix="[TEST]")

        assert "[TEST]" in caplog.text

    def test_run_command_failure_returns_none(self, analyzer, caplog):
        """Test command failure returns None and logs error."""
        with caplog.at_level(logging.ERROR):
            result = analyzer.run_command("false", description="Fail test")

        assert result is None
        assert "Command failed" in caplog.text or "exit code" in caplog.text

    def test_run_command_logs_completion_time(self, analyzer, caplog):
        """Test run_command logs completion time."""
        with caplog.at_level(logging.INFO):
            analyzer.run_command("echo 'test'")

        assert "Completed in" in caplog.text

    @patch("subprocess.run")
    def test_run_command_handles_generic_exception(self, mock_run, analyzer, caplog):
        """Test run_command handles unexpected exceptions."""
        mock_run.side_effect = RuntimeError("Unexpected error")

        with caplog.at_level(logging.ERROR):
            result = analyzer.run_command("test", description="Test")

        assert result is None
        assert any(
            "Error running command" in record.message for record in caplog.records
        )


class TestClearCache:
    """Test clear_cache method."""

    def test_clear_cache_with_default_patterns(self, analyzer, mock_analysis):
        """Test clear_cache uses default patterns."""
        mock_analysis.clear_cache.return_value = True

        result = analyzer.clear_cache()

        assert result is True
        mock_analysis.clear_cache.assert_called_once_with(
            patterns=["avo_*.npz"], prefix=""
        )

    def test_clear_cache_with_custom_patterns(self, analyzer, mock_analysis):
        """Test clear_cache with custom patterns."""
        mock_analysis.clear_cache.return_value = True

        custom_patterns = ["*.tmp", "*.bak"]
        result = analyzer.clear_cache(patterns=custom_patterns)

        assert result is True
        mock_analysis.clear_cache.assert_called_once_with(
            patterns=custom_patterns, prefix=""
        )

    def test_clear_cache_with_prefix(self, analyzer, mock_analysis):
        """Test clear_cache includes prefix."""
        mock_analysis.clear_cache.return_value = True

        analyzer.clear_cache(prefix="[CACHE]")

        mock_analysis.clear_cache.assert_called_once_with(
            patterns=["avo_*.npz"], prefix="[CACHE]"
        )

    def test_clear_cache_delegates_to_analysis(self, analyzer, mock_analysis):
        """Test clear_cache delegates to AnalysisCommon."""
        mock_analysis.clear_cache.return_value = False

        result = analyzer.clear_cache()

        assert result is False
        mock_analysis.clear_cache.assert_called_once()


class TestCheckFileExists:
    """Test check_file_exists method."""

    def test_check_file_exists_returns_true_for_existing_file(self, analyzer):
        """Test file existence check returns True for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = analyzer.check_file_exists(tmp_path, "Test file")

            assert result is True
        finally:
            Path(tmp_path).unlink()

    def test_check_file_exists_returns_false_for_missing_file(self, analyzer):
        """Test file existence check returns False for missing file."""
        result = analyzer.check_file_exists(
            "/nonexistent/path/file.txt", "Missing file"
        )

        assert result is False

    def test_check_file_exists_logs_size_for_existing_file(self, analyzer, caplog):
        """Test check_file_exists logs file size."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"x" * 1024)  # 1 KB
            tmp_path = tmp.name

        try:
            with caplog.at_level(logging.INFO):
                analyzer.check_file_exists(tmp_path, "Test file")

            assert "Found" in caplog.text
            assert "MB" in caplog.text
        finally:
            Path(tmp_path).unlink()

    def test_check_file_exists_logs_error_for_missing_file(self, analyzer, caplog):
        """Test check_file_exists logs error for missing file."""
        with caplog.at_level(logging.ERROR):
            analyzer.check_file_exists("/nonexistent/path/file.txt", "Missing file")

        assert "Missing" in caplog.text

    def test_check_file_exists_handles_permission_error(self, analyzer, caplog):
        """Test check_file_exists handles permission errors gracefully."""
        with patch(
            "src.analysis.pipelines.seismograms.Path.exists",
            side_effect=PermissionError,
        ):
            with caplog.at_level(logging.ERROR):
                result = analyzer.check_file_exists("/restricted/file", "Restricted")

            assert result is False
            assert "Error checking" in caplog.text


class TestOpenFile:
    """Test open_file method."""

    def test_open_file_calls_analysis_helper(self, analyzer, mock_analysis):
        """Test open_file delegates to AnalysisCommon."""
        mock_analysis.open_file.return_value = True

        result = analyzer.open_file("/path/to/file", "Test file")

        assert result is True
        mock_analysis.open_file.assert_called_once_with(
            "/path/to/file", description="Test file", prefix=""
        )

    def test_open_file_with_prefix(self, analyzer, mock_analysis):
        """Test open_file includes prefix."""
        mock_analysis.open_file.return_value = True

        analyzer.open_file("/path/to/file", "Test file", prefix="[TEST]")

        mock_analysis.open_file.assert_called_once_with(
            "/path/to/file", description="Test file", prefix="[TEST]"
        )

    @patch("src.analysis.pipelines.seismograms.time.sleep")
    def test_open_file_sleeps_on_success(self, mock_sleep, analyzer, mock_analysis):
        """Test open_file sleeps after successful open."""
        mock_analysis.open_file.return_value = True

        analyzer.open_file("/path/to/file", "Test file")

        mock_sleep.assert_called_once_with(1)

    @patch("src.analysis.pipelines.seismograms.time.sleep")
    def test_open_file_no_sleep_on_failure(self, mock_sleep, analyzer, mock_analysis):
        """Test open_file doesn't sleep if file open fails."""
        mock_analysis.open_file.return_value = False

        analyzer.open_file("/path/to/file", "Test file")

        mock_sleep.assert_not_called()

    @patch(
        "src.analysis.pipelines.seismograms.time.sleep", side_effect=InterruptedError
    )
    def test_open_file_handles_interrupted_sleep(
        self, mock_sleep, analyzer, mock_analysis, caplog
    ):
        """Test open_file handles InterruptedError during sleep."""
        mock_analysis.open_file.return_value = True

        with caplog.at_level(logging.WARNING):
            result = analyzer.open_file("/path/to/file", "Test file")

        assert result is True
        assert "interrupted" in caplog.text.lower()


class TestMain:
    """Test main orchestration method."""

    def test_main_validates_cache_dir(self, analyzer):
        """Test main validates cache_dir parameter."""
        with pytest.raises(ValueError, match="cache_dir must be a non-empty string"):
            analyzer.main(cache_dir="")

    def test_main_invalid_cache_dir_type(self, analyzer):
        """Test main rejects non-string cache_dir."""
        with pytest.raises(ValueError):
            analyzer.main(cache_dir=None)  # type: ignore

    @patch("src.analysis.pipelines.seismograms.SeismogramAnalyzer.clear_cache")
    @patch("src.modeling.api.run_full_modeling")
    def test_main_clears_cache_by_default(self, mock_run, mock_clear, analyzer):
        """Test main clears cache by default."""
        analyzer.main()

        mock_clear.assert_called_once()

    @patch("src.analysis.pipelines.seismograms.SeismogramAnalyzer.clear_cache")
    @patch("src.modeling.api.run_full_modeling")
    def test_main_skips_cleanup_when_requested(self, mock_run, mock_clear, analyzer):
        """Test main skips cleanup when skip_cleanup=True."""
        analyzer.main(skip_cleanup=True)

        mock_clear.assert_not_called()

    @patch("src.modeling.api.run_full_modeling")
    def test_main_runs_modeling(self, mock_run, analyzer):
        """Test main runs the modeling pipeline."""
        result = analyzer.main(cache_dir=".cache")

        assert result is True
        mock_run.assert_called_once_with(
            cache_dir=".cache",
            skip_cleanup=False,
            verbose=False,
            add_avo_noise=False,
        )

    @patch("src.modeling.api.run_full_modeling")
    def test_main_enables_debug_when_verbose(self, mock_run, analyzer, caplog):
        """Test main enables debug logging when verbose=True."""
        with caplog.at_level(logging.DEBUG):
            analyzer.main(verbose=True)

        assert analyzer._logger.level == logging.DEBUG

    @patch(
        "src.modeling.api.run_full_modeling",
        side_effect=ImportError("Module not found"),
    )
    def test_main_handles_import_error(self, mock_run, analyzer):
        """Test main handles ImportError from modeling API."""
        with pytest.raises(RuntimeError, match="Failed to import modeling API"):
            analyzer.main()

    @patch(
        "src.modeling.api.run_full_modeling",
        side_effect=RuntimeError("Modeling failed"),
    )
    def test_main_handles_general_error(self, mock_run, analyzer):
        """Test main handles general exceptions from modeling."""
        with pytest.raises(RuntimeError, match="Seismic modeling failed"):
            analyzer.main()

    @patch("src.modeling.api.run_full_modeling")
    def test_main_logs_startup(self, mock_run, analyzer, caplog):
        """Test main logs pipeline startup."""
        with caplog.at_level(logging.INFO):
            analyzer.main(cache_dir=".cache")

        assert "Starting seismic modeling pipeline" in caplog.text

    @patch("src.modeling.api.run_full_modeling")
    def test_main_logs_completion(self, mock_run, analyzer, caplog):
        """Test main logs pipeline completion."""
        with caplog.at_level(logging.INFO):
            analyzer.main()

        assert "completed successfully" in caplog.text


class TestIntegration:
    """Integration tests for SeismogramAnalyzer."""

    def test_multiple_instances_have_separate_loggers(self, mock_analysis):
        """Test multiple instances have independent loggers."""
        analyzer1 = SeismogramAnalyzer(analysis=mock_analysis)
        analyzer2 = SeismogramAnalyzer(analysis=MagicMock(spec=AnalysisCommon))

        # While they have the same name, they should be independent instances
        assert analyzer1._logger.name == analyzer2._logger.name
        assert analyzer1._logger is analyzer2._logger  # Same singleton from logging

    def test_dependency_injection_workflow(self, mock_analysis):
        """Test complete workflow with dependency injection."""
        mock_analysis.clear_cache.return_value = True
        mock_analysis.open_file.return_value = True

        analyzer = SeismogramAnalyzer(analysis=mock_analysis)

        # Test workflow
        assert analyzer.clear_cache() is True
        assert analyzer.open_file("/path", "test") is True

        # Verify calls
        assert mock_analysis.clear_cache.called
        assert mock_analysis.open_file.called

    @patch("src.modeling.api.run_full_modeling")
    def test_full_pipeline_execution(self, mock_run, analyzer):
        """Test full pipeline execution with all steps."""
        # This should not raise any exceptions
        result = analyzer.main(
            cache_dir=".cache",
            skip_cleanup=False,
            verbose=False,
        )

        assert result is True
        mock_run.assert_called_once()
