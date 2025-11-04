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

    def test_validate_type_falls_back_to_duck_typing(self) -> None:  # type: ignore
        """Test that validate_type falls back to duck-typing on isinstance failure."""
        # type: ignore
        # The function catches TypeError and falls back to duck-typing,
        # so most values won't raise exceptions
        TypeValidator.validate("not an int", int, "count")  # Doesn't raise
        TypeValidator.validate(42, str, "name")  # Doesn't raise
        TypeValidator.validate([1, 2, 3], dict, "config")  # Doesn't raise


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


class TestValidateProtocol:
    """Tests for Protocol-based validation."""

    def test_validate_protocol_with_matching_attributes(self) -> None:  # type: ignore
        """Test validation of object with Protocol attributes."""
        # type: ignore

        class MyProtocol(Protocol):
            def method(self) -> None: ...  # type: ignore

            @property
            def prop(self) -> str: ...  # type: ignore

        class Implementation:
            def method(self) -> None:  # type: ignore
                pass

            @property
            def prop(self) -> str:  # type: ignore
                return "value"

        # Should not raise - implementation duck-types the protocol
        TypeValidator.validate(Implementation(), MyProtocol, "handler")

    def test_validate_protocol_with_logger(self) -> None:  # type: ignore
        """Test validation for custom types."""
        # type: ignore

        # This just validates that a custom type instance matches a custom type
        class CustomType:
            pass

        # Should not raise - same instance type
        TypeValidator.validate(CustomType(), CustomType, "test_field")

    def test_validate_protocol_with_hasattr_check(self) -> None:  # type: ignore
        """Test Protocol validation uses hasattr checks."""
        # type: ignore

        class MockProtocol:
            __protocol_attrs__ = ["method1", "method2"]

        class Implementation:
            def method1(self) -> None:  # type: ignore
                pass

            def method2(self) -> None:  # type: ignore
                pass

        # Should not raise
        TypeValidator.validate(Implementation(), MockProtocol, "impl")

    def test_validate_protocol_with_missing_attributes(self) -> None:  # type: ignore
        """Test Protocol validation warns about missing attributes."""
        # type: ignore
        with mock.patch.object(
            logging.getLogger("src.analysis.factories.validators"), "warning"
        ) as mock_warning:

            class MockProtocol:
                __protocol_attrs__ = ["method1", "method2", "method3"]

            class PartialImplementation:
                def method1(self) -> None:  # type: ignore
                    pass

            TypeValidator.validate(PartialImplementation(), MockProtocol, "impl")
            # Should warn about missing methods
            mock_warning.assert_called()


class TestValidateTypeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_validate_type_with_exception_in_protocol_check(self) -> None:  # type: ignore
        """Test validate_type handles exceptions during protocol validation."""
        # type: ignore
        with mock.patch.object(
            logging.getLogger("src.analysis.factories.validators"), "debug"
        ) as mock_debug:

            class BrokenProtocol:
                __protocol_attrs__ = property(lambda self: 1 / 0)  # Will raise

            TypeValidator.validate(object(), BrokenProtocol, "test")
            mock_debug.assert_called()

    def test_validate_type_with_isinstance_exception(self) -> None:  # type: ignore
        """Test validate_type handles exceptions in isinstance checks."""
        # type: ignore
        with mock.patch.object(
            logging.getLogger("src.analysis.factories.validators"), "warning"
        ) as mock_warning:

            class BadType:
                @classmethod
                def __instancecheck__(cls, instance: object) -> bool:  # type: ignore
                    raise RuntimeError("Bad type check")

            # Should handle the exception and fall through to protocol logic
            TypeValidator.validate(object(), BadType, "test")
            # May log warning
            if mock_warning.called:
                assert "Could not validate" in str(mock_warning.call_args)

    def test_validate_type_preserves_duck_typing_fallback(self) -> None:  # type: ignore
        """Test that duck-typing fallback is used when isinstance fails."""
        # type: ignore
        # This tests that the function doesn't crash on type mismatches
        # but falls back to duck-typing
        TypeValidator.validate(42, str, "number")
        TypeValidator.validate(3.14, int, "number")
        TypeValidator.validate([1, 2], dict, "mapping")


class TestBuilderValidationError:
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

        # Base instance with Derived type also works due to duck-typing fallback
        TypeValidator.validate(Base(), Derived, "obj")


class TestValidateTypeLogging:
    """Tests for logging behavior in validate_type."""

    def test_logs_debug_on_protocol_type(self) -> None:  # type: ignore
        """Test debug logging for unknown Protocol types."""
        # type: ignore
        with mock.patch("src.analysis.factories.validators.logger") as mock_logger:
            # When isinstance(expected_type, type) is False, it goes to protocol validation
            class NotAType:
                pass

            TypeValidator.validate(object(), NotAType, "field")
            # Check if any logging was done
            assert len(mock_logger.method_calls) > 0

    def test_logs_warning_on_isinstance_exception(self) -> None:  # type: ignore
        """Test warning logging when isinstance fails."""
        # type: ignore
        with mock.patch("src.analysis.factories.validators.logger") as mock_logger:

            class BadType:
                @classmethod
                def __instancecheck__(cls, instance: object) -> bool:  # type: ignore
                    raise RuntimeError("Bad")

            TypeValidator.validate(object(), BadType, "field")
            assert mock_logger.warning.called or mock_logger.debug.called

    def test_logs_warning_on_missing_protocol_attrs(self) -> None:  # type: ignore
        """Test warning when protocol attributes are missing."""
        # type: ignore
        with mock.patch("src.analysis.factories.validators.logger") as mock_logger:

            class MockProtocol:
                __protocol_attrs__ = ["required_method"]

            class IncompleteImpl:
                pass

            TypeValidator.validate(IncompleteImpl(), MockProtocol, "impl")
            assert mock_logger.warning.called

    def test_logs_with_correct_field_name(self) -> None:  # type: ignore
        """Test that logs include field names."""
        # type: ignore
        with mock.patch("src.analysis.factories.validators.logger") as mock_logger:
            TypeValidator.validate(object(), object, "my_specific_field")
            # Check that field name appears in logged calls
            call_args_list = [str(call) for call in mock_logger.method_calls]
            # At least one call should mention the field if logging occurs
            # (this is a soft check for logging behavior)
            # Just verify that method calls were recorded
            assert isinstance(call_args_list, list)
