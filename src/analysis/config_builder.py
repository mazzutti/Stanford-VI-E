"""Fluent configuration builder for analysis components.

This module provides a unified ConfigBuilder pattern that simplifies creating
and configuring analysis configurations. Eliminates manual dictionary construction
and provides type-safe configuration building through method chaining.

Design Pattern:
    - ConfigBuilder[T]: Generic configuration builder with fluent API
    - Configuration validation: Built-in validation during building
    - Type safety: Generic type preserves config type information
    - Method chaining: Fluent interface for readable config construction

Benefits:
    - Eliminates ~80 lines of manual configuration code
    - Type-safe configuration (compile-time checking possible)
    - Cleaner, more readable configuration construction
    - Built-in validation and defaults
    - Easy to test with partial configurations

Example:
    >>> from src.analysis.facies import FaciesAnalysisConfig
    >>> from src.analysis import ConfigBuilder
    >>>
    >>> config = (ConfigBuilder(FaciesAnalysisConfig)
    ...     .set("cache_dir", ".cache")
    ...     .set("dilation_window", 5)
    ...     .set("verbose", True)
    ...     .build())
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from src.core.configuration import ConfigRule, ConfigValidator

__all__ = [
    "ConfigBuilder",
    "Configurable",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic config type


class Configurable(Protocol):
    """Protocol for objects that can be configured.

    Objects implementing this protocol can be configured via ConfigBuilder.
    """

    def configure(self, config: Any) -> None:
        """Configure the object.

        Parameters
        ----------
        config : Any
            Configuration object
        """
        raise NotImplementedError()


@dataclass
class ConfigBuilder(Generic[T]):
    """Fluent builder for configuration objects.

    Provides a clean, chainable API for constructing configuration objects
    with optional validation and defaults using BaseConfig's validation framework.

    Type Parameters
    ---------------
    T : TypeVar
        Type of configuration object being built

    Attributes
    ----------
    config_class : Type[T]
        Configuration class to instantiate
    values : dict
        Configuration values being accumulated
    defaults : dict
        Default values for fields
    _validator : ConfigValidator
        Shared validation framework from BaseConfig

    Examples
    --------
    Basic usage:
        >>> from src.analysis.facies import FaciesAnalysisConfig
        >>> config = (ConfigBuilder(FaciesAnalysisConfig)
        ...     .set("cache_dir", ".cache")
        ...     .set("dilation_window", 5)
        ...     .build())

    With validation:
        >>> builder = ConfigBuilder(FaciesAnalysisConfig)
        >>> builder.add_validator(
        ...     "dilation_window",
        ...     lambda v: v > 0 or "must be positive"
        ... )
        >>> config = builder.set("dilation_window", 5).build()  # Valid
    """

    config_class: type[T]
    values: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    defaults: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    _validator: ConfigValidator = field(default_factory=ConfigValidator)

    def set(self, key: str, value: Any) -> ConfigBuilder[T]:
        """Set a configuration value.

        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Configuration value

        Returns
        -------
        ConfigBuilder[T]
            Self for method chaining

        Example
        -------
        >>> builder.set("timeout", 60.0)
        """
        self.values[key] = value
        logger.debug("Set %s=%s", key, value)
        return self

    # ConfigBuilder is intentionally terse; small helper functions are preferred

    def set_multiple(self, **kwargs: Any) -> ConfigBuilder[T]:
        """Set multiple configuration values at once.

        Parameters
        ----------
        **kwargs : Any
            Configuration key-value pairs

        Returns
        -------
        ConfigBuilder[T]
            Self for method chaining

        Example
        -------
        >>> builder.set_multiple(timeout=60.0, retries=3)
        """
        for key, value in kwargs.items():
            self.set(key, value)
        return self

    def set_default(self, key: str, value: Any) -> ConfigBuilder[T]:
        """Set a default value for a configuration key.

        Defaults are used if value not explicitly set during build().

        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Default value

        Returns
        -------
        ConfigBuilder[T]
            Self for method chaining

        Example
        -------
        >>> builder.set_default("timeout", 60.0)
        """
        self.defaults[key] = value
        return self

    def set_defaults(self, **kwargs: Any) -> ConfigBuilder[T]:
        """Set multiple default values.

        Parameters
        ----------
        **kwargs : Any
            Default key-value pairs

        Returns
        -------
        ConfigBuilder[T]
            Self for method chaining
        """
        self.defaults.update(kwargs)
        return self

    def add_validator(
        self,
        key: str,
        validator: Callable[[Any], bool],
    ) -> ConfigBuilder[T]:
        """Add a validation function for a configuration key.

        Uses BaseConfig's validation framework for consistent validation behavior.

        Parameters
        ----------
        key : str
            Configuration key to validate
        validator : callable
            Function that returns True if value is valid, False otherwise

        Returns
        -------
        ConfigBuilder[T]
            Self for method chaining

        Example
        -------
        >>> builder.add_validator("timeout", lambda v: v > 0)
        """
        rule = ConfigRule(key=key, validators=[validator])
        self._validator.add_rule(rule)
        return self

    def add_validators(
        self, validators: dict[str, Callable[[Any], bool]]
    ) -> ConfigBuilder[T]:
        """Add multiple validation functions.

        Parameters
        ----------
        validators : dict
            Dictionary of key -> validator function

        Returns
        -------
        ConfigBuilder[T]
            Self for method chaining
        """
        for key, validator in validators.items():
            self.add_validator(key, validator)
        return self

    def build(self) -> T:
        """Build the configuration object.

        Combines explicitly set values with defaults, validates using BaseConfig's
        validation framework, and instantiates the configuration class.

        Returns
        -------
        T
            Configured object instance

        Raises
        ------
        ValueError
            If validation fails
        TypeError
            If required fields are missing

        Example
        -------
        >>> config = builder.set("key", "value").build()
        """
        # Start with defaults
        final_values = self.defaults.copy()
        final_values.update(self.values)

        # Validate using BaseConfig's validator
        is_valid, errors = self._validator.validate(final_values)
        if not is_valid:
            raise ValueError(f"Validation errors: {errors}")

        # Instantiate config class
        try:
            # Declare instance variable once to avoid redefinition warnings
            instance: T
            if is_dataclass(self.config_class):
                # For dataclasses, use only fields that exist
                config_fields = {f.name for f in fields(self.config_class)}
                filtered_values = {
                    k: v for k, v in final_values.items() if k in config_fields
                }
                instance = self.config_class(**filtered_values)
            else:
                instance = self.config_class(**final_values)

            logger.info(
                "Built %s with %d configuration values",
                self.config_class.__name__,
                len(final_values),
            )
            return instance

        except TypeError as e:
            logger.error("Failed to build %s: %s", self.config_class.__name__, e)
            raise ValueError(
                f"Cannot build {self.config_class.__name__}: Missing required fields? "
                f"Configured: {list(final_values.keys())}. Error: {e}"
            ) from e

    def clone(self) -> ConfigBuilder[T]:
        """Create a copy of this builder for reuse/variation.

        Returns
        -------
        ConfigBuilder[T]
            New builder with same configuration

        Example
        -------
        >>> builder1 = ConfigBuilder(MyConfig).set("x", 1)
        >>> builder2 = builder1.clone().set("y", 2)
        >>> config1 = builder1.build()
        >>> config2 = builder2.build()
        """
        new_builder = ConfigBuilder(
            config_class=self.config_class,
            values=self.values.copy(),
            defaults=self.defaults.copy(),
        )
        # Copy validator rules. Accessing the `_validator` internals is
        # intentional here to duplicate rule objects into the cloned
        # builder. Keep this narrow and justified to avoid exposing the
        # ConfigValidator internals more widely.

        # Temporarily allow protected-access for the narrow copy operation.
        # pylint: disable=protected-access
        new_builder._validator = ConfigValidator()
        for rule in self._validator.rules.values():
            new_builder._validator.add_rule(rule)
        # pylint: enable=protected-access
        return new_builder

    def to_dict(self) -> dict[str, Any]:
        """Get current configuration as dictionary.

        Returns
        -------
        dict
            Dictionary of configuration values (defaults + values)
        """
        result = self.defaults.copy()
        result.update(self.values)
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        config_name = self.config_class.__name__
        value_count = len(self.values)
        default_count = len(self.defaults)
        return (
            f"ConfigBuilder({config_name}, "
            f"values={value_count}, defaults={default_count})"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return self.__repr__()


# Convenience factory functions


def build_config(config_class: type[T], **kwargs: Any) -> T:
    """Quick factory to build a configuration.

    Parameters
    ----------
    config_class : Type[T]
        Configuration class to build
    **kwargs : Any
        Configuration values

    Returns
    -------
    T
        Built configuration

    Example
    -------
    >>> config = build_config(MyConfig, timeout=60, retries=3)
    """
    builder = ConfigBuilder(config_class)
    for key, value in kwargs.items():
        builder.set(key, value)
    return builder.build()


def config_with_defaults(
    config_class: type[T], defaults: dict[str, Any], **kwargs: Any
) -> T:
    """Build configuration with defaults.

    Parameters
    ----------
    config_class : Type[T]
        Configuration class
    defaults : dict
        Default values
    **kwargs : Any
        Override values

    Returns
    -------
    T
        Built configuration
    """
    return (
        ConfigBuilder(config_class)
        .set_defaults(**defaults)
        .set_multiple(**kwargs)
        .build()
    )
