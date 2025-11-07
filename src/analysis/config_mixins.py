"""Reusable mixins for configuration classes.

This module provides common patterns for configuration validation
to avoid duplication across different analyzer configurations.
"""

from abc import abstractmethod


class ValidatableConfigMixin:
    """Mixin providing standard configuration validation pattern.

    Provides is_valid() method that catches validation errors from
    _validate_params() abstract method. Subclasses must implement
    _validate_params() which raises ValueError for invalid configs.

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class MyConfig(ValidatableConfigMixin):
        ...     value: int
        ...
        ...     def _validate_params(self) -> None:
        ...         if self.value < 0:
        ...             raise ValueError("value must be non-negative")
        >>> cfg = MyConfig(value=-1)
        >>> cfg.is_valid()
        False
    """

    @abstractmethod
    def _validate_params(self) -> None:
        """Validate configuration parameters.

        Should raise ValueError if any parameter is invalid.
        Do not catch exceptions here.

        Raises
        ------
        ValueError
            If any parameter fails validation.
        """
        pass

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns
        -------
        bool
            True if all parameters pass validation, False otherwise.
        """
        try:
            self._validate_params()
            return True
        except ValueError:
            return False
