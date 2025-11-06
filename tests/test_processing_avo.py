"""Comprehensive tests for AVOValidator and AVOValidityReport."""

import logging
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.processing.avo.validator import AVOValidator, AVOValidityReport
from src.processing.core.constants import (
    DEFAULT_MAX_AVO_ANGLE,
    DEFAULT_CONTRAST_THRESHOLD,
)


class TestAVOValidityReportCreation:
    """Test AVOValidityReport creation and properties."""

    def test_report_creation_with_all_fields(self):
        """Test creating a report with all fields."""
        report = AVOValidityReport(
            max_angle=30.0,
            contrast_vp=0.1,
            contrast_vs=0.15,
            contrast_rho=0.08,
            contrast_flag=False,
            angle_flag=False,
            suggested_angles=None,
        )
        assert report.max_angle == 30.0
        assert report.contrast_vp == 0.1
        assert report.contrast_vs == 0.15
        assert report.contrast_rho == 0.08
        assert report.contrast_flag is False
        assert report.angle_flag is False
        assert report.suggested_angles is None

    def test_report_creation_with_suggested_angles(self):
        """Test creating a report with suggested angles."""
        suggested = [0, 10, 15]
        report = AVOValidityReport(
            max_angle=45.0,
            contrast_vp=0.2,
            contrast_vs=0.18,
            contrast_rho=0.22,
            contrast_flag=True,
            angle_flag=True,
            suggested_angles=suggested,
        )
        assert report.suggested_angles == suggested

    def test_report_creation_minimal(self):
        """Test creating a minimal report."""
        report = AVOValidityReport(
            max_angle=20.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=False,
        )
        assert report is not None
        assert report.suggested_angles is None

    def test_report_dataclass_nature(self):
        """Test that report behaves as a dataclass."""
        report = AVOValidityReport(
            max_angle=25.0,
            contrast_vp=0.12,
            contrast_vs=0.11,
            contrast_rho=0.10,
            contrast_flag=False,
            angle_flag=False,
        )
        # Should be immutable-like behavior from dataclass
        assert hasattr(report, "max_angle")
        assert hasattr(report, "contrast_vp")


class TestAVOValidityReportIsValid:
    """Test AVOValidityReport is_valid method."""

    def test_is_valid_when_no_flags_and_low_contrast(self):
        """Test is_valid returns True when conditions are good."""
        report = AVOValidityReport(
            max_angle=20.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=False,
        )
        assert report.is_valid(contrast_threshold=0.1) is True

    def test_is_valid_false_when_contrast_flag_set(self):
        """Test is_valid returns False when contrast flag is set."""
        report = AVOValidityReport(
            max_angle=20.0,
            contrast_vp=0.15,
            contrast_vs=0.14,
            contrast_rho=0.16,
            contrast_flag=True,
            angle_flag=False,
        )
        assert report.is_valid(contrast_threshold=0.1) is False

    def test_is_valid_false_when_angle_flag_set(self):
        """Test is_valid returns False when angle flag is set."""
        report = AVOValidityReport(
            max_angle=45.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=True,
        )
        assert report.is_valid(contrast_threshold=0.1) is False

    def test_is_valid_with_both_flags(self):
        """Test is_valid returns False when both flags are set."""
        report = AVOValidityReport(
            max_angle=50.0,
            contrast_vp=0.2,
            contrast_vs=0.18,
            contrast_rho=0.22,
            contrast_flag=True,
            angle_flag=True,
        )
        assert report.is_valid(contrast_threshold=0.1) is False

    def test_is_valid_with_custom_threshold(self):
        """Test is_valid with custom contrast threshold."""
        report = AVOValidityReport(
            max_angle=20.0,
            contrast_vp=0.12,
            contrast_vs=0.11,
            contrast_rho=0.10,
            contrast_flag=False,
            angle_flag=False,
        )
        # With 0.1 threshold, max contrast (0.12) exceeds it
        assert report.is_valid(contrast_threshold=0.1) is False
        # With 0.15 threshold, max contrast (0.12) is within
        assert report.is_valid(contrast_threshold=0.15) is True

    def test_is_valid_checks_max_contrast(self):
        """Test is_valid checks against maximum contrast value."""
        report = AVOValidityReport(
            max_angle=20.0,
            contrast_vp=0.08,
            contrast_vs=0.09,
            contrast_rho=0.12,  # Highest
            contrast_flag=False,
            angle_flag=False,
        )
        # Should fail if max(0.08, 0.09, 0.12) > 0.1
        assert report.is_valid(contrast_threshold=0.1) is False


class TestAVOValidityReportPrintSummary:
    """Test AVOValidityReport print_summary method."""

    def test_print_summary_logs_header(self, caplog):
        """Test print_summary logs header."""
        report = AVOValidityReport(
            max_angle=25.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=False,
        )
        with caplog.at_level(logging.INFO):
            report.print_summary()
        assert "Aki-Richards Linearization Validity Summary" in caplog.text

    def test_print_summary_logs_angles(self, caplog):
        """Test print_summary logs max angle."""
        report = AVOValidityReport(
            max_angle=35.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=False,
        )
        with caplog.at_level(logging.INFO):
            report.print_summary()
        assert "35.0" in caplog.text

    def test_print_summary_logs_contrasts(self, caplog):
        """Test print_summary logs contrasts."""
        report = AVOValidityReport(
            max_angle=25.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=False,
        )
        with caplog.at_level(logging.INFO):
            report.print_summary()
        assert "Vp fractional contrast" in caplog.text
        assert "Vs fractional contrast" in caplog.text
        assert "Rho fractional contrast" in caplog.text

    def test_print_summary_warns_on_contrast_flag(self, caplog):
        """Test print_summary warns when contrast flag is set."""
        report = AVOValidityReport(
            max_angle=25.0,
            contrast_vp=0.15,
            contrast_vs=0.16,
            contrast_rho=0.14,
            contrast_flag=True,
            angle_flag=False,
        )
        with caplog.at_level(logging.WARNING):
            report.print_summary()
        assert "⚠️" in caplog.text or "contrasts" in caplog.text.lower()

    def test_print_summary_warns_on_angle_flag(self, caplog):
        """Test print_summary warns when angle flag is set."""
        report = AVOValidityReport(
            max_angle=45.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=True,
        )
        with caplog.at_level(logging.WARNING):
            report.print_summary()
        assert "⚠️" in caplog.text or "angle" in caplog.text.lower()

    def test_print_summary_suggests_angles(self, caplog):
        """Test print_summary suggests angles."""
        suggested = [0, 10, 20]
        report = AVOValidityReport(
            max_angle=45.0,
            contrast_vp=0.15,
            contrast_vs=0.16,
            contrast_rho=0.14,
            contrast_flag=True,
            angle_flag=True,
            suggested_angles=suggested,
        )
        with caplog.at_level(logging.INFO):
            report.print_summary()
        assert "Suggested safer angles" in caplog.text


class TestAVOValidityReportToDict:
    """Test AVOValidityReport to_dict method."""

    def test_to_dict_returns_dictionary(self):
        """Test to_dict returns a dictionary."""
        report = AVOValidityReport(
            max_angle=25.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=False,
        )
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_all_fields(self):
        """Test to_dict contains all fields."""
        report = AVOValidityReport(
            max_angle=25.0,
            contrast_vp=0.05,
            contrast_vs=0.06,
            contrast_rho=0.04,
            contrast_flag=False,
            angle_flag=False,
        )
        result = report.to_dict()
        assert "max_angle" in result
        assert "contrast_vp" in result
        assert "contrast_vs" in result
        assert "contrast_rho" in result
        assert "contrast_flag" in result
        assert "angle_flag" in result
        assert "suggested_angles" in result

    def test_to_dict_preserves_values(self):
        """Test to_dict preserves all values."""
        suggested = [0, 15]
        report = AVOValidityReport(
            max_angle=30.5,
            contrast_vp=0.055,
            contrast_vs=0.065,
            contrast_rho=0.045,
            contrast_flag=True,
            angle_flag=False,
            suggested_angles=suggested,
        )
        result = report.to_dict()
        assert result["max_angle"] == 30.5
        assert result["contrast_vp"] == 0.055
        assert result["contrast_vs"] == 0.065
        assert result["contrast_rho"] == 0.045
        assert result["contrast_flag"] is True
        assert result["angle_flag"] is False
        assert result["suggested_angles"] == suggested


class TestAVOValidatorInitialization:
    """Test AVOValidator initialization."""

    def test_validator_basic_init(self):
        """Test basic AVOValidator initialization."""
        validator = AVOValidator()
        assert validator is not None
        assert validator.max_angle == DEFAULT_MAX_AVO_ANGLE
        assert validator.contrast_threshold == DEFAULT_CONTRAST_THRESHOLD

    def test_validator_with_custom_max_angle(self):
        """Test AVOValidator with custom max_angle."""
        validator = AVOValidator(max_angle=45.0)
        assert validator.max_angle == 45.0

    def test_validator_with_custom_threshold(self):
        """Test AVOValidator with custom contrast threshold."""
        validator = AVOValidator(contrast_threshold=0.2)
        assert validator.contrast_threshold == 0.2

    def test_validator_with_custom_logger(self):
        """Test AVOValidator with custom logger."""
        mock_logger = MagicMock(spec=logging.Logger)
        validator = AVOValidator(logger=mock_logger)
        assert validator.logger is mock_logger

    def test_validator_has_default_logger(self):
        """Test AVOValidator has default logger."""
        validator = AVOValidator()
        assert validator.logger is not None


class TestAVOValidatorValidateMethod:
    """Test AVOValidator validate method."""

    def test_validate_with_valid_input(self):
        """Test validate with valid inputs."""
        validator = AVOValidator()
        vp = np.array([3000.0, 3300.0])
        vs = np.array([1500.0, 1650.0])
        rho = np.array([2300.0, 2400.0])

        report = validator.validate(vp, vs, rho)

        assert isinstance(report, AVOValidityReport)
        assert report is not None

    def test_validate_returns_report(self):
        """Test validate returns AVOValidityReport."""
        validator = AVOValidator()
        vp = np.array([3000.0, 3300.0])
        vs = np.array([1500.0, 1650.0])
        rho = np.array([2300.0, 2400.0])

        report = validator.validate(vp, vs, rho)

        assert isinstance(report, AVOValidityReport)
        assert hasattr(report, "contrast_vp")
        assert hasattr(report, "contrast_vs")
        assert hasattr(report, "contrast_rho")

    def test_validate_with_custom_max_angle(self):
        """Test validate with custom max_angle validator."""
        validator = AVOValidator(max_angle=25.0)
        vp = np.array([3000.0, 3300.0])
        vs = np.array([1500.0, 1650.0])
        rho = np.array([2300.0, 2400.0])

        report = validator.validate(vp, vs, rho)

        assert report.max_angle == 25.0

    def test_validate_with_small_contrasts(self):
        """Test validate with small property contrasts."""
        validator = AVOValidator()
        vp = np.array([3000.0, 3010.0])  # Small contrast
        vs = np.array([1500.0, 1505.0])
        rho = np.array([2300.0, 2305.0])

        report = validator.validate(vp, vs, rho)

        assert report.contrast_vp < 0.01
        assert report.contrast_vs < 0.01
        assert report.contrast_rho < 0.01

    def test_validate_with_large_contrasts(self):
        """Test validate with large property contrasts."""
        validator = AVOValidator(contrast_threshold=0.1)
        vp = np.array([2000.0, 4000.0])  # (4000-2000)/max(4000,1e-12) = 0.5
        vs = np.array([1000.0, 2000.0])
        rho = np.array([2000.0, 3000.0])

        report = validator.validate(vp, vs, rho)

        # Contrast formula: (max - min) / max(max_val, 1e-12)
        assert report.contrast_vp == 0.5
        assert report.contrast_vp > 0.1
        assert report.contrast_flag is True

    def test_validate_with_arrays(self):
        """Test validate with array inputs."""
        validator = AVOValidator()
        vp = np.array([3000.0, 3100.0, 3200.0, 3300.0])
        vs = np.array([1500.0, 1550.0, 1600.0, 1650.0])
        rho = np.array([2300.0, 2325.0, 2350.0, 2400.0])

        report = validator.validate(vp, vs, rho)

        assert isinstance(report, AVOValidityReport)
        assert report.contrast_vp > 0


class TestAVOValidatorEdgeCases:
    """Test AVOValidator edge cases."""

    def test_validate_with_zero_velocity(self):
        """Test validate with zero velocity (should handle gracefully)."""
        validator = AVOValidator()
        vp = np.array([0.0, 3000.0])
        vs = np.array([0.0, 1500.0])
        rho = np.array([0.0, 2300.0])

        # Should not raise, but contrast will be computed
        report = validator.validate(vp, vs, rho)
        assert isinstance(report, AVOValidityReport)

    def test_validate_with_equal_velocities(self):
        """Test validate with equal velocities (zero contrast)."""
        validator = AVOValidator()
        vp = np.array([3000.0, 3000.0])
        vs = np.array([1500.0, 1500.0])
        rho = np.array([2300.0, 2300.0])

        report = validator.validate(vp, vs, rho)

        assert report.contrast_vp == 0.0
        assert report.contrast_vs == 0.0
        assert report.contrast_rho == 0.0

    def test_validate_with_negative_velocities(self):
        """Test validate with negative velocities."""
        validator = AVOValidator()
        vp = np.array([-3000.0, 3000.0])
        vs = np.array([-1500.0, 1500.0])
        rho = np.array([-2300.0, 2300.0])

        # Should still compute contrasts using nanmax/nanmin
        report = validator.validate(vp, vs, rho)
        assert isinstance(report, AVOValidityReport)


class TestAVOValidatorConsistency:
    """Test AVOValidator consistency."""

    def test_multiple_validations_same_input(self):
        """Test multiple validations with same input give same result."""
        validator = AVOValidator()
        vp = np.array([3000.0, 3300.0])
        vs = np.array([1500.0, 1650.0])
        rho = np.array([2300.0, 2400.0])

        report1 = validator.validate(vp, vs, rho)
        report2 = validator.validate(vp, vs, rho)

        assert report1.contrast_vp == report2.contrast_vp
        assert report1.contrast_vs == report2.contrast_vs
        assert report1.contrast_rho == report2.contrast_rho

    def test_validation_order_independence(self):
        """Test validation order doesn't matter."""
        validator = AVOValidator()

        # Forward
        report1 = validator.validate(
            vp=np.array([3000.0, 3300.0]),
            vs=np.array([1500.0, 1650.0]),
            rho=np.array([2300.0, 2400.0]),
        )

        # Same arrays, should get same results
        report2 = validator.validate(
            vp=np.array([3000.0, 3300.0]),
            vs=np.array([1500.0, 1650.0]),
            rho=np.array([2300.0, 2400.0]),
        )

        assert report1.contrast_vp == report2.contrast_vp
