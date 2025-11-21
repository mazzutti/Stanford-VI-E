"""Tests for src.processing.core.abstracts module.

Tests for abstract base classes that define contracts for processing components.
"""

from abc import ABC
from typing import Any

import numpy as np
import pytest

from src.processing.core.abstracts import (Manager, MaterialProperty,
                                           Processor, Resampler, Validator)


class TestProcessorAbstract:
    """Test Processor abstract base class."""

    def test_processor_is_abstract(self):
        """Test that Processor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Processor()

    def test_processor_requires_process_method(self):
        """Test that subclass must implement process method."""

        class IncompleteProcessor(Processor):
            pass

        with pytest.raises(TypeError):
            IncompleteProcessor()

    def test_processor_concrete_implementation(self):
        """Test concrete Processor implementation."""

        class ConcreteProcessor(Processor):
            def process(self, data, **kwargs):
                return data * 2

        processor = ConcreteProcessor()
        result = processor.process(5)
        assert result == 10

    def test_processor_with_kwargs(self):
        """Test Processor process method with kwargs."""

        class MultiplyProcessor(Processor):
            def process(self, data, **kwargs):
                factor = kwargs.get("factor", 1)
                return data * factor

        processor = MultiplyProcessor()
        result = processor.process(5, factor=3)
        assert result == 15

    def test_processor_with_array_data(self):
        """Test Processor with numpy array data."""

        class ArrayProcessor(Processor):
            def process(self, data, **kwargs):
                return np.array(data) * 2

        processor = ArrayProcessor()
        result = processor.process([1, 2, 3])
        np.testing.assert_array_equal(result, [2, 4, 6])


class TestManagerAbstract:
    """Test Manager abstract base class."""

    def test_manager_is_abstract(self):
        """Test that Manager cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Manager()

    def test_manager_requires_clear_method(self):
        """Test that subclass must implement clear method."""

        class IncompleteManager(Manager):
            def summarize(self):
                pass

        with pytest.raises(TypeError):
            IncompleteManager()

    def test_manager_requires_summarize_method(self):
        """Test that subclass must implement summarize method."""

        class IncompleteManager(Manager):
            def clear(self):
                return 0

        with pytest.raises(TypeError):
            IncompleteManager()

    def test_manager_concrete_implementation(self):
        """Test concrete Manager implementation."""

        class ConcreteManager(Manager):
            def __init__(self):
                self.items = []

            def clear(self):
                count = len(self.items)
                self.items.clear()
                return count

            def summarize(self):
                return f"Manager has {len(self.items)} items"

        manager = ConcreteManager()
        manager.items = ["a", "b", "c"]
        assert manager.clear() == 3
        assert len(manager.items) == 0

    def test_manager_clear_returns_int(self):
        """Test that Manager.clear returns an integer."""

        class SimpleManager(Manager):
            def clear(self, *args, **kwargs):
                return 42

            def summarize(self, *args, **kwargs):
                pass

        manager = SimpleManager()
        result = manager.clear()
        assert isinstance(result, int)
        assert result == 42

    def test_manager_summarize_can_print(self):
        """Test Manager.summarize can output information."""

        class VerboseManager(Manager):
            def __init__(self):
                self.output = []

            def clear(self, *args, **kwargs):
                return 0

            def summarize(self, *args, **kwargs):
                self.output.append("Summary called")

        manager = VerboseManager()
        manager.summarize()
        assert len(manager.output) == 1


class TestResamplerAbstract:
    """Test Resampler abstract base class."""

    def test_resampler_is_abstract(self):
        """Test that Resampler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Resampler()

    def test_resampler_requires_resample_method(self):
        """Test that subclass must implement resample method."""

        class IncompleteResampler(Resampler):
            def inverse_resample(self, data, plan):
                pass

        with pytest.raises(TypeError):
            IncompleteResampler()

    def test_resampler_requires_inverse_resample_method(self):
        """Test that subclass must implement inverse_resample method."""

        class IncompleteResampler(Resampler):
            def resample(self, data, plan):
                pass

        with pytest.raises(TypeError):
            IncompleteResampler()

    def test_resampler_concrete_implementation(self):
        """Test concrete Resampler implementation."""

        class SimpleResampler(Resampler):
            def resample(self, data, plan):
                return np.array(data) * 2

            def inverse_resample(self, data, plan):
                return np.array(data) / 2

        resampler = SimpleResampler()
        data = np.array([1, 2, 3])
        plan = None

        forward = resampler.resample(data, plan)
        backward = resampler.inverse_resample(forward, plan)

        np.testing.assert_array_equal(data, backward)

    def test_resampler_accepts_array_like(self):
        """Test Resampler works with array-like inputs."""

        class ListResampler(Resampler):
            def resample(self, data, plan):
                return np.array(data)

            def inverse_resample(self, data, plan):
                return np.array(data)

        resampler = ListResampler()
        result = resampler.resample([1, 2, 3], None)
        assert isinstance(result, np.ndarray)

    def test_resampler_plan_parameter(self):
        """Test Resampler respects plan parameter."""

        class PlanAwareResampler(Resampler):
            def resample(self, data, plan):
                if plan is None:
                    return np.array(data)
                return np.array(data) * plan.get("factor", 1)

            def inverse_resample(self, data, plan):
                if plan is None:
                    return np.array(data)
                return np.array(data) / plan.get("factor", 1)

        resampler = PlanAwareResampler()
        plan = {"factor": 2}
        result = resampler.resample([1, 2], plan)
        np.testing.assert_array_equal(result, [2, 4])


class TestMaterialPropertyAbstract:
    """Test MaterialProperty abstract base class."""

    def test_material_property_is_abstract(self):
        """Test that MaterialProperty cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MaterialProperty()

    def test_material_property_requires_get_data(self):
        """Test that subclass must implement get_data method."""

        class IncompleteMaterial(MaterialProperty):
            def set_data(self, data):
                pass

            def ensure_units(self):
                pass

            def validate(self):
                pass

        with pytest.raises(TypeError):
            IncompleteMaterial()

    def test_material_property_requires_all_methods(self):
        """Test that all abstract methods are required."""
        required_methods = ["get_data", "set_data", "ensure_units", "validate"]

        for method in required_methods:
            incomplete_dict = {
                m: lambda *a, **kw: None for m in required_methods if m != method
            }

            class IncompleteMaterial(MaterialProperty):
                pass

            for m, func in incomplete_dict.items():
                setattr(IncompleteMaterial, m, func)

            with pytest.raises(TypeError):
                IncompleteMaterial()

    def test_material_property_concrete_implementation(self):
        """Test concrete MaterialProperty implementation."""

        class SimpleMaterial(MaterialProperty):
            def __init__(self, data):
                self._data = data

            def get_data(self):
                return self._data

            def set_data(self, data):
                self._data = data

            def ensure_units(self):
                return False

            def validate(self):
                if not np.all(np.isfinite(self._data)):
                    raise ValueError("Data contains non-finite values")

        material = SimpleMaterial(np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(material.get_data(), [1.0, 2.0, 3.0])
        material.validate()

    def test_material_property_set_data(self):
        """Test MaterialProperty.set_data method."""

        class MutableMaterial(MaterialProperty):
            def __init__(self):
                self._data = np.array([1.0, 2.0])

            def get_data(self):
                return self._data

            def set_data(self, data):
                self._data = np.array(data)

            def ensure_units(self):
                return False

            def validate(self):
                pass

        material = MutableMaterial()
        material.set_data([3.0, 4.0])
        np.testing.assert_array_equal(material.get_data(), [3.0, 4.0])

    def test_material_property_ensure_units(self):
        """Test MaterialProperty.ensure_units method."""

        class ConvertibleMaterial(MaterialProperty):
            def __init__(self):
                self._data = np.array([1.0, 2.0])
                self._needs_conversion = True

            def get_data(self):
                return self._data

            def set_data(self, data):
                self._data = data

            def ensure_units(self):
                if self._needs_conversion:
                    self._data = self._data * 1000
                    self._needs_conversion = False
                    return True
                return False

            def validate(self):
                pass

        material = ConvertibleMaterial()
        assert material.ensure_units() is True
        np.testing.assert_array_equal(material.get_data(), [1000.0, 2000.0])
        assert material.ensure_units() is False

    def test_material_property_validate_raises(self):
        """Test MaterialProperty.validate raises on invalid data."""

        class StrictMaterial(MaterialProperty):
            def __init__(self, data):
                self._data = data

            def get_data(self):
                return self._data

            def set_data(self, data):
                self._data = data

            def ensure_units(self):
                return False

            def validate(self):
                if np.any(self._data < 0):
                    raise ValueError("Data contains negative values")

        material = StrictMaterial(np.array([-1.0, 2.0]))
        with pytest.raises(ValueError, match="negative"):
            material.validate()


class TestValidatorAbstract:
    """Test Validator abstract base class."""

    def test_validator_is_abstract(self):
        """Test that Validator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Validator()

    def test_validator_requires_validate_method(self):
        """Test that subclass must implement validate method."""

        class IncompleteValidator(Validator):
            def is_valid(self):
                pass

        with pytest.raises(TypeError):
            IncompleteValidator()

    def test_validator_requires_is_valid_method(self):
        """Test that subclass must implement is_valid method."""

        class IncompleteValidator(Validator):
            def validate(self):
                pass

        with pytest.raises(TypeError):
            IncompleteValidator()

    def test_validator_concrete_implementation(self):
        """Test concrete Validator implementation."""

        class SimpleValidator(Validator):
            def validate(self, *args, **kwargs):
                return {"valid": True, "errors": []}

            def is_valid(self, *args, **kwargs):
                return True

        validator = SimpleValidator()
        assert validator.is_valid() is True
        result = validator.validate()
        assert result["valid"] is True

    def test_validator_validate_returns_dict(self):
        """Test Validator.validate returns a dictionary."""

        class DictValidator(Validator):
            def validate(self, *args, **kwargs):
                return {
                    "valid": True,
                    "message": "All checks passed",
                    "warnings": [],
                }

            def is_valid(self, *args, **kwargs):
                return True

        validator = DictValidator()
        result = validator.validate()
        assert isinstance(result, dict)
        assert "valid" in result

    def test_validator_is_valid_returns_bool(self):
        """Test Validator.is_valid returns a boolean."""

        class BoolValidator(Validator):
            def validate(self, *args, **kwargs):
                return {"valid": True}

            def is_valid(self, *args, **kwargs):
                return True

        validator = BoolValidator()
        result = validator.is_valid()
        assert isinstance(result, bool)

    def test_validator_with_data_checking(self):
        """Test Validator that checks data validity."""

        class DataValidator(Validator):
            def validate(self, data, *args, **kwargs):
                errors = []
                if data is None:
                    errors.append("Data is None")
                elif np.any(~np.isfinite(data)):
                    errors.append("Data contains non-finite values")

                return {"valid": len(errors) == 0, "errors": errors}

            def is_valid(self, data, *args, **kwargs):
                result = self.validate(data)
                return result["valid"]

        validator = DataValidator()

        # Valid data
        assert validator.is_valid(np.array([1.0, 2.0, 3.0])) is True

        # Invalid data
        assert validator.is_valid(np.array([1.0, np.nan])) is False

    def test_validator_validate_with_kwargs(self):
        """Test Validator.validate accepts kwargs."""

        class FlexibleValidator(Validator):
            def validate(self, *args, **kwargs):
                strict_mode = kwargs.get("strict", False)
                return {"valid": True, "strict": strict_mode}

            def is_valid(self, *args, **kwargs):
                return self.validate(*args, **kwargs)["valid"]

        validator = FlexibleValidator()
        result = validator.validate(strict=True)
        assert result["strict"] is True


class TestAbstractInheritance:
    """Test abstract class inheritance patterns."""

    def test_multiple_abstract_inheritance(self):
        """Test class implementing multiple abstract base classes."""

        class MultiProcessor(Processor, Validator):
            def process(self, data, **kwargs):
                return data

            def validate(self, *args, **kwargs):
                return {"valid": True}

            def is_valid(self, *args, **kwargs):
                return True

        processor = MultiProcessor()
        assert processor.process(5) == 5
        assert processor.is_valid() is True

    def test_abstract_subclass_hierarchy(self):
        """Test abstract class inheritance hierarchy."""

        class AbstractProcessor(Processor):
            def common_method(self):
                return "common"

        class ConcreteProcessor(AbstractProcessor):
            def process(self, data, **kwargs):
                return data

        processor = ConcreteProcessor()
        assert processor.common_method() == "common"
        assert processor.process(5) == 5

    def test_abstract_with_mixin_methods(self):
        """Test abstract class with mixin-like methods."""

        class ProcessorWithLogging(Processor):
            def __init__(self):
                self.log = []

            def log_message(self, msg):
                self.log.append(msg)

        class LoggingProcessor(ProcessorWithLogging):
            def process(self, data, **kwargs):
                self.log_message("Processing data")
                return data

        processor = LoggingProcessor()
        result = processor.process(42)
        assert result == 42
        assert "Processing data" in processor.log
