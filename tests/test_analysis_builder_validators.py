# mypy: ignore-errors
# mypy: ignore-errors
"""Comprehensive unit tests for factories/validators module.

Tests cover the TypeValidator class and BuilderValidationError class
with various type validation scenarios including regular types, callables,
Protocols, and edge cases.
"""


import logging
from typing import Callable, Protocol
from unittest import mock

import pytest

from src.analysis.exceptions import BuilderFrozenError, BuilderValidationError
from src.analysis.factories.validators import TypeValidator


class TestValidateTypeBasic:
    """Tests for basic type validation."""

    def test_validate_type_with_none_value(self) -> None:
        """Test that None values are always acceptable."""
        TypeValidator.validate(None, int, "test_field")
        TypeValidator.validate(None, str, "test_field")
        TypeValidator.validate(None, list, "test_field")
        # Should not raise any exception

    def test_validate_type_with_matching_type(self) -> None:
        """Test validation with matching types."""
        TypeValidator.validate(42, int, "count")
        TypeValidator.validate("hello", str, "name")
        TypeValidator.validate([1, 2, 3], list, "items")
        TypeValidator.validate({"key": "value"}, dict, "config")

    def test_validate_type_with_subclass(self) -> None:
        """Test validation with subclass instances."""

        class CustomInt(int):
            pass

        TypeValidator.validate(CustomInt(5), int, "number")


class TestValidateCallable:
    """Tests for callable validation."""

    def test_validate_callable_with_function(self) -> None:
        """Test validation of regular functions."""

        def my_func() -> None:
            pass

        TypeValidator.validate(my_func, Callable, "handler")

    def test_validate_callable_with_lambda(self) -> None:
        """Test validation of lambda functions."""
        TypeValidator.validate(lambda x: x + 1, Callable, "operation")

    def test_validate_callable_with_method(self) -> None:
        """Test validation of methods."""

        class MyClass:
            def my_method(self) -> None:
                pass

        obj = MyClass()
        TypeValidator.validate(obj.my_method, Callable, "callback")

    def test_validate_callable_with_callable_class(self) -> None:
        """Test validation of callable class instances."""

        class CallableClass:
            def __call__(self) -> None:
                pass

        obj = CallableClass()
        TypeValidator.validate(obj, Callable, "handler")

    def test_validate_callable_with_non_callable(self) -> None:
        """Test validation fails for non-callable."""
        with pytest.raises(TypeError, match="Expected callable"):
            TypeValidator.validate(42, Callable, "handler")

        with pytest.raises(TypeError, match="Expected callable"):
            TypeValidator.validate("not callable", Callable, "callback")

    def test_validate_callable_error_includes_field_name(self) -> None:
        """Test error message includes field name for callable validation."""
        with pytest.raises(TypeError, match="my_callback"):
            TypeValidator.validate(42, Callable, "my_callback")


class TestValidateType:
    """Tests for validating the type type itself."""

    def test_validate_type_with_int_class(self) -> None:
        """Test validation of actual type objects."""
        TypeValidator.validate(int, type, "type_field")

    def test_validate_type_with_custom_class(self) -> None:
        """Test validation of custom class objects."""

        class MyClass:
            pass

        TypeValidator.validate(MyClass, type, "type_field")

    def test_validate_type_with_builtin_type(self) -> None:
        """Test validation of builtin types."""
        TypeValidator.validate(str, type, "field")
        TypeValidator.validate(list, type, "field")
        TypeValidator.validate(dict, type, "field")

    def test_validate_type_fails_for_non_type(self) -> None:
        """Test validation fails for non-type objects."""
        with pytest.raises(TypeError, match="Expected type"):
            TypeValidator.validate(42, type, "type_field")

        with pytest.raises(TypeError, match="Expected type"):
            TypeValidator.validate("not a type", type, "type_field")


class TestAnalysisBuilderValidatorsBuilderValidationError:
    """Tests for BuilderValidationError exception."""

    def test_builder_validation_error_basic(self) -> None:
        """Test BuilderValidationError creation with message only."""
        error = BuilderValidationError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.missing_deps == []

    def test_builder_validation_error_with_dependencies(self) -> None:
        """Test BuilderValidationError with missing dependencies."""
        missing = ["dep1", "dep2", "dep3"]
        error = BuilderValidationError("Missing dependencies", missing_deps=missing)
        assert str(error) == "Missing dependencies"
        assert error.missing_deps == missing

    def test_builder_validation_error_inherits_from_value_error(self) -> None:
        """Test that BuilderValidationError is a ValueError."""
        error = BuilderValidationError("test")
        assert isinstance(error, ValueError)

    def test_builder_validation_error_with_none_deps(self) -> None:
        """Test that None missing_deps defaults to empty list."""
        error = BuilderValidationError("message", missing_deps=None)
        assert error.missing_deps == []

    def test_builder_validation_error_empty_deps(self) -> None:
        """Test with explicitly empty dependencies list."""
        error = BuilderValidationError("message", missing_deps=[])
        assert error.missing_deps == []

    def test_builder_validation_error_can_be_raised(self) -> None:
        """Test that BuilderValidationError can be raised and caught."""
        with pytest.raises(BuilderValidationError) as exc_info:
            raise BuilderValidationError("Test error", missing_deps=["a", "b"])

        assert "Test error" in str(exc_info.value)
        assert exc_info.value.missing_deps == ["a", "b"]

    def test_builder_validation_error_message_formatting(self) -> None:
        """Test various message formats."""
        messages = [
            "Single error",
            "Error with: special characters!",
            "Error with\nmultiline\nmessage",
            "",
        ]
        for msg in messages:
            error = BuilderValidationError(msg)
            assert str(error) == msg


class TestBuilderFrozenError:
    """Tests for BuilderFrozenError exception."""

    def test_builder_frozen_error_basic(self) -> None:
        """Test BuilderFrozenError creation."""
        error = BuilderFrozenError("Builder is frozen")
        assert str(error) == "Builder is frozen"

    def test_builder_frozen_error_inherits_from_runtime_error(self) -> None:
        """Test that BuilderFrozenError is a RuntimeError."""
        error = BuilderFrozenError("test")
        assert isinstance(error, RuntimeError)

    def test_builder_frozen_error_can_be_raised(self) -> None:
        """Test that BuilderFrozenError can be raised and caught."""
        with pytest.raises(BuilderFrozenError):
            raise BuilderFrozenError("Frozen!")

    def test_builder_frozen_error_no_message(self) -> None:
        """Test BuilderFrozenError with no message."""
        error = BuilderFrozenError()
        assert str(error) == ""


class TestValidationWorkflow:
    """Integration tests for common validation workflows."""

    def test_validate_builder_field_types(self) -> None:
        """Test validating multiple builder fields."""
        fields = {
            "name": ("my_builder", str),
            "size": (100, int),
            "callback": (lambda: None, Callable),
        }

        for field_name, (value, expected_type) in fields.items():
            TypeValidator.validate(value, expected_type, field_name)

    def test_validate_optional_fields(self) -> None:
        """Test validating optional fields."""
        # None is acceptable for any type
        TypeValidator.validate(None, str, "optional_name")
        TypeValidator.validate(None, int, "optional_count")
        TypeValidator.validate(None, list, "optional_items")

    def test_validation_error_recovery(self) -> None:
        """Test error handling in validation."""
        try:
            TypeValidator.validate(42, str, "name")
        except TypeError as e:
            assert "name" in str(e)
            assert "Expected str" in str(e)
            # Recovery
            TypeValidator.validate("valid", str, "name")

    def test_complex_type_hierarchy(self) -> None:
        """Test validation with complex type hierarchies."""

        class Base:
            pass

        class Derived(Base):
            pass

        # Derived instance matches Base type
        TypeValidator.validate(Derived(), Base, "obj")
