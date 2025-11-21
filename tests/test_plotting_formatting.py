"""Tests for plotting formatting utilities.

Tests verify text formatting functions for logging and display.
"""

import logging
import tempfile
from pathlib import Path

from src.plotting.helpers.formatting import (FormattingHelper,
                                             get_formatting_helper,
                                             print_angle_summary,
                                             print_cache_info, print_header,
                                             print_selected_angles)


class TestPrintHeader:
    """Test print_header function."""

    def test_print_header_logs_message(self, caplog):
        """Test that print_header logs the title."""
        with caplog.at_level(logging.INFO):
            print_header("Test Header")

        # Should contain the title and separator
        assert "Test Header" in caplog.text
        assert "=" * 70 in caplog.text

    def test_print_header_with_special_chars(self, caplog):
        """Test print_header with special characters."""
        with caplog.at_level(logging.INFO):
            print_header("Test: Header [Special]")

        assert "Test: Header [Special]" in caplog.text

    def test_print_header_empty_string(self, caplog):
        """Test print_header with empty string."""
        with caplog.at_level(logging.INFO):
            print_header("")

        # Should still log separators
        assert "=" * 70 in caplog.text


class TestPrintAngleSummary:
    """Test print_angle_summary function."""

    def test_print_angle_summary_basic(self, caplog):
        """Test basic angle summary printing."""
        import numpy as np

        angles = [0, 15, 30]
        volumes = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
        ]

        with caplog.at_level(logging.INFO):
            print_angle_summary(angles, volumes)

        # Should log header and angles
        assert "ANGLE-DEPENDENT SUMMARY" in caplog.text
        assert "0.0°" in caplog.text
        assert "15.0°" in caplog.text
        assert "30.0°" in caplog.text

    def test_print_angle_summary_with_stack(self, caplog):
        """Test angle summary with stack."""
        import numpy as np

        angles = [0, 15]
        volumes = [np.array([1, 2]), np.array([3, 4])]
        stack = np.array([5, 6])

        with caplog.at_level(logging.INFO):
            print_angle_summary(angles, volumes, stack=stack)

        assert "Stack" in caplog.text

    def test_print_angle_summary_with_gradient(self, caplog):
        """Test angle summary with gradient."""
        import numpy as np

        angles = [0, 15]
        volumes = [np.array([1, 2]), np.array([3, 4])]
        gradient = np.array([0.1, 0.2])

        with caplog.at_level(logging.INFO):
            print_angle_summary(angles, volumes, gradient=gradient)

        assert "Gradient" in caplog.text

    def test_print_angle_summary_with_both(self, caplog):
        """Test angle summary with both stack and gradient."""
        import numpy as np

        angles = [0, 15]
        volumes = [np.array([1, 2]), np.array([3, 4])]
        stack = np.array([5, 6])
        gradient = np.array([0.1, 0.2])

        with caplog.at_level(logging.INFO):
            print_angle_summary(angles, volumes, stack=stack, gradient=gradient)

        assert "Stack" in caplog.text
        assert "Gradient" in caplog.text


class TestPrintSelectedAngles:
    """Test print_selected_angles function."""

    def test_print_selected_angles_basic(self, caplog):
        """Test printing selected angles."""
        selected = [0, 15, 30]
        weights = [0.3, 0.5, 0.2]

        with caplog.at_level(logging.INFO):
            print_selected_angles(selected, weights)

        assert "Selected angles" in caplog.text
        assert "Weights" in caplog.text

    def test_print_selected_angles_single(self, caplog):
        """Test printing single selected angle."""
        with caplog.at_level(logging.INFO):
            print_selected_angles([15], [1.0])

        assert "Selected angles" in caplog.text

    def test_print_selected_angles_empty(self, caplog):
        """Test printing empty angles."""
        with caplog.at_level(logging.INFO):
            print_selected_angles([], [])

        assert "Selected angles" in caplog.text


class TestPrintCacheInfo:
    """Test print_cache_info function."""

    def test_print_cache_info_with_file(self, caplog):
        """Test printing cache info with valid file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cache_file = f.name
            f.write(b"test data" * 1000)

        try:
            with caplog.at_level(logging.INFO):
                print_cache_info(cache_file)

            assert "Saved multi-angle results to" in caplog.text
            assert "File size" in caplog.text
        finally:
            Path(cache_file).unlink()

    def test_print_cache_info_with_none(self, caplog):
        """Test that None cache file doesn't log."""
        with caplog.at_level(logging.INFO):
            print_cache_info(None)

        # Should not have logged anything
        assert "Saved" not in caplog.text

    def test_print_cache_info_with_empty_string(self, caplog):
        """Test that empty string cache file doesn't log."""
        with caplog.at_level(logging.INFO):
            print_cache_info("")

        assert "Saved" not in caplog.text

    def test_print_cache_info_missing_file(self, caplog):
        """Test with non-existent file."""
        with caplog.at_level(logging.INFO):
            print_cache_info("/nonexistent/path/to/file.npz")

        # Should log the message but handle file size error gracefully
        assert "Saved multi-angle results to" in caplog.text


class TestFormattingHelper:
    """Test FormattingHelper facade class."""

    def test_formatting_helper_print_header(self, caplog):
        """Test FormattingHelper.print_header."""
        helper = FormattingHelper()

        with caplog.at_level(logging.INFO):
            helper.print_header("Test")

        assert "Test" in caplog.text

    def test_formatting_helper_print_angle_summary(self, caplog):
        """Test FormattingHelper.print_angle_summary."""
        import numpy as np

        helper = FormattingHelper()
        angles = [0, 15]
        volumes = [np.array([1, 2]), np.array([3, 4])]

        with caplog.at_level(logging.INFO):
            helper.print_angle_summary(angles, volumes)

        assert "ANGLE-DEPENDENT SUMMARY" in caplog.text

    def test_formatting_helper_print_selected_angles(self, caplog):
        """Test FormattingHelper.print_selected_angles."""
        helper = FormattingHelper()

        with caplog.at_level(logging.INFO):
            helper.print_selected_angles([0, 15], [0.5, 0.5])

        assert "Selected angles" in caplog.text

    def test_formatting_helper_print_cache_info(self, caplog):
        """Test FormattingHelper.print_cache_info."""
        helper = FormattingHelper()

        with caplog.at_level(logging.INFO):
            helper.print_cache_info(None)

        # Should not raise


class TestGetFormattingHelper:
    """Test get_formatting_helper factory function."""

    def test_get_formatting_helper_returns_instance(self):
        """Test get_formatting_helper returns FormattingHelper instance."""
        helper = get_formatting_helper()

        assert helper is not None
        assert isinstance(helper, FormattingHelper)
        assert hasattr(helper, "print_header")

    def test_get_formatting_helper_returns_singleton(self):
        """Test that get_formatting_helper returns the same instance."""
        helper1 = get_formatting_helper()
        helper2 = get_formatting_helper()

        # Should be the same singleton instance
        assert helper1 is helper2
