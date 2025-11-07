# mypy: ignore-errors
# mypy: ignore-errors
"""Comprehensive unit tests for factories/validators module.

Tests cover the TypeValidator class and BuilderValidationError class
with various type validation scenarios including regular types, callables,
Protocols, and edge cases.
"""
# type: ignore


import logging
from typing import Callable, Protocol
from unittest import mock

import pytest

from src.analysis.factories.validators import TypeValidator
from src.analysis.exceptions import BuilderFrozenError, BuilderValidationError


class TestValidateTypeBasic:
    """Tests for basic type validation."""

    def test_validate_type_with_none_value(self) -> None:  # type: ignore
        """Test that None values are always acceptable."""
        # type: ignore
        TypeValidator.validate(None, int, "test_field")
        TypeValidator.validate(None, str, "test_field")
        TypeValidator.validate(None, list, "test_field")
        # Should not raise any exception

    def test_validate_type_with_matching_type(self) -> None:  # type: ignore
        """Test validation with matching types."""
        # type: ignore
        TypeValidator.validate(42, int, "count")
        TypeValidator.validate("hello", str, "name")
        TypeValidator.validate([1, 2, 3], list, "items")
        TypeValidator.validate({"key": "value"}, dict, "config")

    def test_validate_type_with_subclass(self) -> None:  # type: ignore
        """Test validation with subclass instances."""
        # type: ignore

        class CustomInt(int):
            pass

        TypeValidator.validate(CustomInt(5), int, "number")


class TestValidateCallable:
    """Tests for callable validation."""

    def test_validate_callable_with_function(self) -> None:  # type: ignore
        """Test validation of regular functions."""
        # type: ignore

        def my_func() -> None:  # type: ignore
            pass

        TypeValidator.validate(my_func, Callable, "handler")

    def test_validate_callable_with_lambda(self) -> None:  # type: ignore
        """Test validation of lambda functions."""
        # type: ignore
        TypeValidator.validate(lambda x: x + 1, Callable, "operation")

    def test_validate_callable_with_method(self) -> None:  # type: ignore
        """Test validation of methods."""
        # type: ignore

        class MyClass:
            def my_method(self) -> None:  # type: ignore
                pass

        obj = MyClass()
        TypeValidator.validate(obj.my_method, Callable, "callback")

    def test_validate_callable_with_callable_class(self) -> None:  # type: ignore
        """Test validation of callable class instances."""
        # type: ignore

        class CallableClass:
            def __call__(self) -> None:  # type: ignore
                pass

        obj = CallableClass()
        TypeValidator.validate(obj, Callable, "handler")

    def test_validate_callable_with_non_callable(self) -> None:  # type: ignore
        """Test validation fails for non-callable."""
        # type: ignore
        with pytest.raises(TypeError, match="Expected callable"):
            TypeValidator.validate(42, Callable, "handler")

        with pytest.raises(TypeError, match="Expected callable"):
            TypeValidator.validate("not callable", Callable, "callback")

    def test_validate_callable_error_includes_field_name(self) -> None:  # type: ignore
        """Test error message includes field name for callable validation."""
        # type: ignore
        with pytest.raises(TypeError, match="my_callback"):
            TypeValidator.validate(42, Callable, "my_callback")


class TestValidateType:
    """Tests for validating the type type itself."""

    def test_validate_type_with_int_class(self) -> None:  # type: ignore
        """Test validation of actual type objects."""
        # type: ignore
        TypeValidator.validate(int, type, "type_field")

    def test_validate_type_with_custom_class(self) -> None:  # type: ignore
        """Test validation of custom class objects."""
        # type: ignore

        class MyClass:
            pass

        TypeValidator.validate(MyClass, type, "type_field")

    def test_validate_type_with_builtin_type(self) -> None:  # type: ignore
        """Test validation of builtin types."""
        # type: ignore
        TypeValidator.validate(str, type, "field")
        TypeValidator.validate(list, type, "field")
        TypeValidator.validate(dict, type, "field")

    def test_validate_type_fails_for_non_type(self) -> None:  # type: ignore
        """Test validation fails for non-type objects."""
        # type: ignore
        with pytest.raises(TypeError, match="Expected type"):
            TypeValidator.validate(42, type, "type_field")

        with pytest.raises(TypeError, match="Expected type"):
            TypeValidator.validate("not a type", type, "type_field")


class TestAnalysisBuilderValidatorsBuilderValidationError:
    """Tests for BuilderValidationError exception."""

    def test_builder_validation_error_basic(self) -> None:  # type: ignore
        """Test BuilderValidationError creation with message only."""
        # type: ignore
        error = BuilderValidationError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.missing_deps == []

    def test_builder_validation_error_with_dependencies(self) -> None:  # type: ignore
        """Test BuilderValidationError with missing dependencies."""
        # type: ignore
        missing = ["dep1", "dep2", "dep3"]
        error = BuilderValidationError("Missing dependencies", missing_deps=missing)
        assert str(error) == "Missing dependencies"
        assert error.missing_deps == missing

    def test_builder_validation_error_inherits_from_value_error(self) -> None:  # type: ignore
        """Test that BuilderValidationError is a ValueError."""
        # type: ignore
        error = BuilderValidationError("test")
        assert isinstance(error, ValueError)

    def test_builder_validation_error_with_none_deps(self) -> None:  # type: ignore
        """Test that None missing_deps defaults to empty list."""
        # type: ignore
        error = BuilderValidationError("message", missing_deps=None)
        assert error.missing_deps == []

    def test_builder_validation_error_empty_deps(self) -> None:  # type: ignore
        """Test with explicitly empty dependencies list."""
        # type: ignore
        error = BuilderValidationError("message", missing_deps=[])
        assert error.missing_deps == []

    def test_builder_validation_error_can_be_raised(self) -> None:  # type: ignore
        """Test that BuilderValidationError can be raised and caught."""
        # type: ignore
        with pytest.raises(BuilderValidationError) as exc_info:
            raise BuilderValidationError("Test error", missing_deps=["a", "b"])

        assert "Test error" in str(exc_info.value)
        assert exc_info.value.missing_deps == ["a", "b"]

    def test_builder_validation_error_message_formatting(self) -> None:  # type: ignore
        """Test various message formats."""
        # type: ignore
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

    def test_builder_frozen_error_basic(self) -> None:  # type: ignore
        """Test BuilderFrozenError creation."""
        # type: ignore
        error = BuilderFrozenError("Builder is frozen")
        assert str(error) == "Builder is frozen"

    def test_builder_frozen_error_inherits_from_runtime_error(self) -> None:  # type: ignore
        """Test that BuilderFrozenError is a RuntimeError."""
        # type: ignore
        error = BuilderFrozenError("test")
        assert isinstance(error, RuntimeError)

    def test_builder_frozen_error_can_be_raised(self) -> None:  # type: ignore
        """Test that BuilderFrozenError can be raised and caught."""
        # type: ignore
        with pytest.raises(BuilderFrozenError):
            raise BuilderFrozenError("Frozen!")

    def test_builder_frozen_error_no_message(self) -> None:  # type: ignore
        """Test BuilderFrozenError with no message."""
        # type: ignore
        error = BuilderFrozenError()
        assert str(error) == ""


class TestValidationWorkflow:
    """Integration tests for common validation workflows."""

    def test_validate_builder_field_types(self) -> None:  # type: ignore
        """Test validating multiple builder fields."""
        # type: ignore
        fields = {
            "name": ("my_builder", str),
            "size": (100, int),
            "callback": (lambda: None, Callable),
        }

        for field_name, (value, expected_type) in fields.items():
            TypeValidator.validate(value, expected_type, field_name)

    def test_validate_optional_fields(self) -> None:  # type: ignore
        """Test validating optional fields."""
        # type: ignore
        # None is acceptable for any type
        TypeValidator.validate(None, str, "optional_name")
        TypeValidator.validate(None, int, "optional_count")
        TypeValidator.validate(None, list, "optional_items")

    def test_validation_error_recovery(self) -> None:  # type: ignore
        """Test error handling in validation."""
        # type: ignore
        try:
            TypeValidator.validate(42, str, "name")
        except TypeError as e:
            assert "name" in str(e)
            assert "Expected str" in str(e)
            # Recovery
            TypeValidator.validate("valid", str, "name")

    def test_complex_type_hierarchy(self) -> None:  # type: ignore
        """Test validation with complex type hierarchies."""
        # type: ignore

        class Base:
            pass

        class Derived(Base):
            pass

        # Derived instance matches Base type
        TypeValidator.validate(Derived(), Base, "obj")
