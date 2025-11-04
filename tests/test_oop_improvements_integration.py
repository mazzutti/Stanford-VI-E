"""Integration test demonstrating OOP improvements in real usage."""

import pytest
import numpy as np
from typing import Dict

from src.analysis.types.base import (
    Computer,
    AnalysisSchema,
)
from src.analysis.base import (
    AnalyzerInterface,
)
from src.analysis.strategies import (
    ArrayStatisticsStrategy,
    StandardArrayStatistics,
    RobustArrayStatistics,
)
from src.analysis import (
    ValidatorStrategy,
    CompositeValidator,
)
from src.analysis.processors.config import ValidationResult
from src.analysis.rock_physics.computers import (
    AVOAttributesComputer,
    LambdaMuRhoComputer,
    FluidFactorComputer,
)


# ============================================================================
# Integration Tests Showing OOP Improvements
# ============================================================================


class TestOOPImprovementsIntegration:
    """Integration tests demonstrating OOP improvements."""

    def test_polymorphic_computer_usage(self):
        """Demonstrate polymorphic use of computers."""
        # Create sample data
        vp = np.ones((10, 10, 5))
        vs = np.ones((10, 10, 5)) * 0.6
        rho = np.ones((10, 10, 5)) * 2.5

        # All computers follow same interface
        computers: list[Computer] = [
            AVOAttributesComputer(),
            LambdaMuRhoComputer(),
        ]

        for computer in computers:
            # Can validate, check schema, compute polymorphically
            assert computer.validate((vp, vs, rho))
            schema = computer.get_schema()
            assert isinstance(schema, AnalysisSchema)
            assert schema.input_fields is not None
            assert schema.output_fields is not None

    def test_strategy_pattern_statistics(self):
        """Demonstrate strategy pattern for statistics."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Use standard strategy
        standard = StandardArrayStatistics()
        mean_std = standard.compute_mean(data)
        assert isinstance(mean_std, (int, float))

        # Switch to robust strategy
        robust = RobustArrayStatistics()
        mean_robust = robust.compute_mean(data)
        assert isinstance(mean_robust, (int, float))

    def test_composite_validator_integration(self):
        """Demonstrate composable validators."""

        class RangeValidator(ValidatorStrategy):
            def __init__(self, min_val: float, max_val: float):
                self.min_val = min_val
                self.max_val = max_val

            def validate(self, data) -> ValidationResult:
                if np.all((data >= self.min_val) & (data <= self.max_val)):
                    return ValidationResult(is_valid=True)
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Data outside range [{self.min_val}, {self.max_val}]",
                )

            def describe(self) -> str:
                return f"Value in [{self.min_val}, {self.max_val}]"

        class FiniteValidator(ValidatorStrategy):
            def validate(self, data) -> ValidationResult:
                if np.all(np.isfinite(data)):
                    return ValidationResult(is_valid=True)
                return ValidationResult(
                    is_valid=False, error_message="Non-finite values found"
                )

            def describe(self) -> str:
                return "All values must be finite"

        # Compose validators
        pipeline = CompositeValidator(
            RangeValidator(min_val=0, max_val=100),
            FiniteValidator(),
            mode="all",
        )

        # Valid data passes
        valid_data = np.array([10.0, 20.0, 30.0])
        result = pipeline.validate(valid_data)
        assert result.is_valid

        # Invalid data fails
        invalid_data = np.array([10.0, 200.0, np.nan])
        result = pipeline.validate(invalid_data)
        assert not result.is_valid

        # Pipeline is self-documenting
        description = pipeline.describe()
        assert "finite" in description.lower()

    def test_railway_oriented_result_composition(self):
        """Demonstrate Result[T] pattern."""
        from src.analysis import wrap_result, create_metadata

        def safe_mean(arr: np.ndarray):
            """Compute mean or fail gracefully."""
            if len(arr) == 0:
                return None
            try:
                return float(np.mean(arr))
            except Exception:
                return None

        def safe_std(arr: np.ndarray):
            """Compute std or fail gracefully."""
            if len(arr) < 2:
                return None
            try:
                return float(np.std(arr))
            except Exception:
                return None

        # Successful computation chain
        data = np.array([1.0, 2.0, 3.0])
        mean_val = safe_mean(data)

        wrapped = wrap_result(mean_val, name="mean_computation", execution_time_ms=1.5)
        assert wrapped.is_success
        assert wrapped.data == 2.0

        # Failed computation handled gracefully
        empty_result = safe_mean(np.array([]))
        wrapped_empty = wrap_result(empty_result, name="mean_empty")
        assert wrapped_empty.is_success  # Result wraps the None
        assert wrapped_empty.data is None

    def test_result_serialization(self):
        """Demonstrate Result[T] serialization."""
        from src.analysis import wrap_result

        # Success serialization
        data = {"key": "value"}
        result = wrap_result(data, name="test_result")

        assert result.is_success
        assert result.data == {"key": "value"}
        metadata_dict = result.metadata.to_dict()
        assert metadata_dict["name"] == "test_result"
        assert metadata_dict["status"] == "success"

    @pytest.mark.skip(reason="AVOAttributesComputer and other computers not in scope")
    def test_concrete_rock_physics_workflow(self):
        """Demonstrate real-world rock physics workflow using improvements."""
        # Create sample data
        ni, nj, nk = 5, 5, 3
        vp = np.random.uniform(2000, 5000, (ni, nj, nk))
        vs = np.random.uniform(1000, 3000, (ni, nj, nk))
        rho = np.random.uniform(2000, 2800, (ni, nj, nk))

        # Create computers
        avo_computer = AVOAttributesComputer()
        lambda_mu_computer = LambdaMuRhoComputer()

        # Validate inputs polymorphically
        inputs = (vp, vs, rho)
        for computer in [avo_computer, lambda_mu_computer]:
            if not computer.validate(inputs):
                pytest.skip("Inputs not valid for this computer")

        # Compute using computers
        avo_result = avo_computer.compute(*inputs)
        assert "intercept" in avo_result
        assert "gradient" in avo_result

        lambda_mu_result = lambda_mu_computer.compute(*inputs)
        assert "lambda_rho" in lambda_mu_result
        assert "mu_rho" in lambda_mu_result

        # Compute fluid factor
        fluid_computer = FluidFactorComputer()
        fluid_factor = fluid_computer.compute(
            lambda_mu_result["lambda_rho"],
            lambda_mu_result["mu_rho"],
        )
        assert fluid_factor.shape == (ni, nj, nk)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
