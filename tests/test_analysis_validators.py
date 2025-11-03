# mypy: ignore-errors
# mypy: ignore-errors
"""Tests for analysis validators module.

Tests cover:
- RangeValidator: correlation, p-value, and range validation
- CountValidator: array element count validation
- QuantileValidator: quantile range validation
- ValidationError: exception handling
"""
# type: ignore


import pytest
from src.analysis.validators import (
    RangeValidator,
    CountValidator,
    QuantileValidator,
    ValidationError,
)


class TestRangeValidator:
    """Tests for RangeValidator static methods."""

    def test_validate_correlation_valid(self) -> None:  # type: ignore
        """Test valid correlation values."""
    # type: ignore
        RangeValidator.validate_correlation(0.95)
        RangeValidator.validate_correlation(-0.95)
        RangeValidator.validate_correlation(0.0)
        RangeValidator.validate_correlation(1.0)
        RangeValidator.validate_correlation(-1.0)

    def test_validate_correlation_invalid_high(self) -> None:  # type: ignore
        """Test correlation above maximum."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_correlation(1.5)

    def test_validate_correlation_invalid_low(self) -> None:  # type: ignore
        """Test correlation below minimum."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_correlation(-1.5)

    def test_validate_correlation_with_custom_name(self) -> None:  # type: ignore
        """Test correlation validation with custom name."""
    # type: ignore
        RangeValidator.validate_correlation(0.5, name="pearson_r")
        RangeValidator.validate_correlation(0.5, name="spearman_rho")

    def test_validate_correlation_nan_not_allowed(self) -> None:  # type: ignore
        """Test NaN correlation not allowed by default."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_correlation(float("nan"))

    def test_validate_correlation_nan_allowed(self) -> None:  # type: ignore
        """Test NaN correlation allowed when specified."""
    # type: ignore
        RangeValidator.validate_correlation(float("nan"), allow_nan=True)

    def test_validate_pvalue_valid(self) -> None:  # type: ignore
        """Test valid p-values."""
    # type: ignore
        RangeValidator.validate_pvalue(0.05)
        RangeValidator.validate_pvalue(0.0)
        RangeValidator.validate_pvalue(1.0)
        RangeValidator.validate_pvalue(0.001)

    def test_validate_pvalue_invalid_high(self) -> None:  # type: ignore
        """Test p-value above maximum."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_pvalue(1.5)

    def test_validate_pvalue_invalid_low(self) -> None:  # type: ignore
        """Test p-value below minimum."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_pvalue(-0.1)

    def test_validate_pvalue_with_custom_name(self) -> None:  # type: ignore
        """Test p-value validation with custom name."""
    # type: ignore
        RangeValidator.validate_pvalue(0.05, name="pearson_pvalue")

    def test_validate_range_valid(self) -> None:  # type: ignore
        """Test valid range validation."""
    # type: ignore
        RangeValidator.validate_range(50, 0, 100, "percentage")
        RangeValidator.validate_range(0, 0, 100, "percentage")
        RangeValidator.validate_range(100, 0, 100, "percentage")

    def test_validate_range_invalid_high(self) -> None:  # type: ignore
        """Test value above range."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_range(150, 0, 100, "percentage")

    def test_validate_range_invalid_low(self) -> None:  # type: ignore
        """Test value below range."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_range(-10, 0, 100, "percentage")

    def test_validate_range_negative_values(self) -> None:  # type: ignore
        """Test range with negative boundaries."""
    # type: ignore
        RangeValidator.validate_range(-50, -100, 0, "temperature")
        RangeValidator.validate_range(-100, -200, -50, "depth")

    def test_validate_range_float_values(self) -> None:  # type: ignore
        """Test range with float values."""
    # type: ignore
        RangeValidator.validate_range(0.5, 0.0, 1.0, "probability")
        RangeValidator.validate_range(0.1, 0.0, 1.0, "probability")

    def test_validate_range_open_interval(self) -> None:  # type: ignore
        """Test range validation with open interval."""
    # type: ignore
        # (0, 100) - does not include endpoints
        RangeValidator.validate_range(50, 0, 100, "value", include_endpoints=False)

        # Should fail at endpoints for open interval
        with pytest.raises(ValidationError):
            RangeValidator.validate_range(0, 0, 100, "value", include_endpoints=False)

    def test_validate_range_nan_not_allowed(self) -> None:  # type: ignore
        """Test NaN in range validation."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_range(float("nan"), 0, 100, "value")

    def test_validate_range_nan_allowed(self) -> None:  # type: ignore
        """Test NaN allowed in range validation."""
    # type: ignore
        RangeValidator.validate_range(float("nan"), 0, 100, "value", allow_nan=True)


class TestCountValidator:
    """Tests for CountValidator static methods."""

    def test_validate_count_valid(self) -> None:  # type: ignore
        """Test valid counts."""
    # type: ignore
        CountValidator.validate_count(0)
        CountValidator.validate_count(1)
        CountValidator.validate_count(100)
        CountValidator.validate_count(1000000)

    def test_validate_count_negative(self) -> None:  # type: ignore
        """Test negative count raises error."""
    # type: ignore
        with pytest.raises(ValidationError):
            CountValidator.validate_count(-1)

    def test_validate_count_with_custom_name(self) -> None:  # type: ignore
        """Test count validation with custom name."""
    # type: ignore
        CountValidator.validate_count(100, name="samples")
        CountValidator.validate_count(50, name="observations")

    def test_validate_count_zero(self) -> None:  # type: ignore
        """Test zero count is valid."""
    # type: ignore
        CountValidator.validate_count(0)
        CountValidator.validate_count(0, name="empty_set")

    def test_validate_count_large_numbers(self) -> None:  # type: ignore
        """Test very large counts."""
    # type: ignore
        CountValidator.validate_count(1_000_000_000)
        CountValidator.validate_count(10**15)


class TestQuantileValidator:
    """Tests for QuantileValidator static methods."""

    def test_validate_quantile_valid(self) -> None:  # type: ignore
        """Test valid quantile values."""
    # type: ignore
        QuantileValidator.validate_quantile(0.25)
        QuantileValidator.validate_quantile(0.5)
        QuantileValidator.validate_quantile(0.75)
        QuantileValidator.validate_quantile(0.0)
        QuantileValidator.validate_quantile(1.0)

    def test_validate_quantile_out_of_range_high(self) -> None:  # type: ignore
        """Test quantile above 1.0."""
    # type: ignore
        with pytest.raises(ValidationError):
            QuantileValidator.validate_quantile(1.5)

    def test_validate_quantile_out_of_range_low(self) -> None:  # type: ignore
        """Test quantile below 0.0."""
    # type: ignore
        with pytest.raises(ValidationError):
            QuantileValidator.validate_quantile(-0.5)

    def test_validate_quantile_with_custom_name(self) -> None:  # type: ignore
        """Test quantile validation with custom name."""
    # type: ignore
        QuantileValidator.validate_quantile(0.5, name="median")
        QuantileValidator.validate_quantile(0.25, name="q25")

    def test_validate_quantile_order_increasing(self) -> None:  # type: ignore
        """Test quantile order validation."""
    # type: ignore
        QuantileValidator.validate_quantile_order(10, 15, 20)
        QuantileValidator.validate_quantile_order(0.25, 0.5, 0.75)

    def test_validate_quantile_order_equal(self) -> None:  # type: ignore
        """Test quantile order with equal values."""
    # type: ignore
        QuantileValidator.validate_quantile_order(10, 10, 10)
        QuantileValidator.validate_quantile_order(5, 5, 15)
        QuantileValidator.validate_quantile_order(5, 15, 15)

    def test_validate_quantile_order_not_increasing(self) -> None:  # type: ignore
        """Test quantile order fails when not increasing."""
    # type: ignore
        with pytest.raises(ValidationError):
            QuantileValidator.validate_quantile_order(20, 15, 10)

    def test_validate_quantile_order_strict(self) -> None:  # type: ignore
        """Test strict quantile order."""
    # type: ignore
        QuantileValidator.validate_quantile_order(10, 15, 20, allow_equal=False)

        # Equal values should fail with allow_equal=False
        with pytest.raises(ValidationError):
            QuantileValidator.validate_quantile_order(10, 10, 10, allow_equal=False)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_creation(self) -> None:  # type: ignore
        """Test creating ValidationError."""
    # type: ignore
        error = ValidationError("Test message")
        assert str(error) == "Test message"

    def test_validation_error_inheritance(self) -> None:  # type: ignore
        """Test ValidationError is an Exception."""
    # type: ignore
        error = ValidationError("Test")
        assert isinstance(error, Exception)

    def test_validation_error_raise_catch(self) -> None:  # type: ignore
        """Test raising and catching ValidationError."""
    # type: ignore
        with pytest.raises(ValidationError):
            raise ValidationError("Test error")

    def test_validation_error_as_exception(self) -> None:  # type: ignore
        """Test catching ValidationError as Exception."""
    # type: ignore
        with pytest.raises(Exception):
            raise ValidationError("Test error")


class TestRangeValidatorEdgeCases:
    """Tests for edge cases in RangeValidator."""

    def test_validate_correlation_boundaries(self) -> None:  # type: ignore
        """Test correlation at exact boundaries."""
    # type: ignore
        RangeValidator.validate_correlation(-1.0)
        RangeValidator.validate_correlation(1.0)

    def test_validate_pvalue_boundaries(self) -> None:  # type: ignore
        """Test p-value at exact boundaries."""
    # type: ignore
        RangeValidator.validate_pvalue(0.0)
        RangeValidator.validate_pvalue(1.0)

    def test_validate_range_inverted_bounds(self) -> None:  # type: ignore
        """Test range with inverted bounds."""
    # type: ignore
        with pytest.raises(ValidationError):
            RangeValidator.validate_range(50, 100, 0, "inverted")

    def test_validate_range_very_narrow(self) -> None:  # type: ignore
        """Test very narrow range."""
    # type: ignore
        RangeValidator.validate_range(0.5, 0.0, 1.0, "narrow")
        RangeValidator.validate_range(0.5000001, 0.5, 0.5000002, "very_narrow")

    def test_validate_correlation_near_boundaries(self) -> None:  # type: ignore
        """Test correlation near boundaries."""
    # type: ignore
        RangeValidator.validate_correlation(0.9999999)
        RangeValidator.validate_correlation(-0.9999999)

    def test_validate_pvalue_small_values(self) -> None:  # type: ignore
        """Test very small p-values."""
    # type: ignore
        RangeValidator.validate_pvalue(1e-10)
        RangeValidator.validate_pvalue(1e-300)


class TestCountValidatorEdgeCases:
    """Tests for edge cases in CountValidator."""

    def test_validate_count_zero(self) -> None:  # type: ignore
        """Test zero count."""
    # type: ignore
        CountValidator.validate_count(0)

    def test_validate_count_one(self) -> None:  # type: ignore
        """Test count of one."""
    # type: ignore
        CountValidator.validate_count(1)

    def test_validate_count_very_large(self) -> None:  # type: ignore
        """Test very large count."""
    # type: ignore
        CountValidator.validate_count(10**100)

    def test_validate_count_float_like_int(self) -> None:  # type: ignore
        """Test float that is an integer."""
    # type: ignore
        CountValidator.validate_count(100)


class TestQuantileValidatorEdgeCases:
    """Tests for edge cases in QuantileValidator."""

    def test_validate_quantile_zero(self) -> None:  # type: ignore
        """Test zero quantile."""
    # type: ignore
        QuantileValidator.validate_quantile(0.0)

    def test_validate_quantile_one(self) -> None:  # type: ignore
        """Test quantile of one."""
    # type: ignore
        QuantileValidator.validate_quantile(1.0)

    def test_validate_quantile_midpoint(self) -> None:  # type: ignore
        """Test quantile at midpoint."""
    # type: ignore
        QuantileValidator.validate_quantile(0.5)

    def test_validate_quantile_order_with_floats(self) -> None:  # type: ignore
        """Test quantile order with float values."""
    # type: ignore
        QuantileValidator.validate_quantile_order(0.1, 0.5, 0.9)
        QuantileValidator.validate_quantile_order(1.0, 5.0, 10.0)


class TestValidatorIntegration:
    """Integration tests combining multiple validators."""

    def test_validate_correlation_and_pvalue(self) -> None:  # type: ignore
        """Test validating both correlation and p-value."""
    # type: ignore
        RangeValidator.validate_correlation(0.8)
        RangeValidator.validate_pvalue(0.01)

    def test_validate_range_and_count(self) -> None:  # type: ignore
        """Test combining range and count validators."""
    # type: ignore
        RangeValidator.validate_range(100, 1, 1000, "count")
        CountValidator.validate_count(100)

    def test_validate_with_quantile_and_range(self) -> None:  # type: ignore
        """Test combining quantile and range validators."""
    # type: ignore
        QuantileValidator.validate_quantile(0.5)
        QuantileValidator.validate_quantile_order(0.25, 0.5, 0.75)
        RangeValidator.validate_range(0.5, 0.0, 1.0, "quantile_value")

    def test_typical_statistical_validation_workflow(self) -> None:  # type: ignore
        """Test typical statistical validation workflow."""
    # type: ignore
        # Validate correlation coefficient
        correlation = 0.85
        RangeValidator.validate_correlation(correlation)

        # Validate p-value
        pvalue = 0.001
        RangeValidator.validate_pvalue(pvalue)

        # Validate sample count
        n_samples = 1000
        CountValidator.validate_count(n_samples)

        # Validate confidence interval quantiles
        QuantileValidator.validate_quantile(0.025)
        QuantileValidator.validate_quantile(0.975)
        QuantileValidator.validate_quantile_order(0.025, 0.5, 0.975)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
