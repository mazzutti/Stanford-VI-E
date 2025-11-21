"""Unified configuration framework for application settings.

This module provides base classes and utilities for managing configuration
across the application with support for validation, profiles, and multiple sources.

Consolidates common patterns from config_manager.py and config_builder.py into
a unified BaseConfig framework that eliminates code duplication while providing
consistent behavior across all configuration types.

Design Patterns:
    - Template Method: BaseConfig provides default implementations
    - Strategy: Different validation and loading strategies
    - Builder: Fluent API for configuration construction
    - Composite: Nested configuration structures

Modules:
    - BaseConfig: Abstract base class for all configurations
    - ConfigProfile: Environment profiles (dev, staging, prod)
    - ConfigValidator: Validation rule management
    - ConfigRule: Individual validation rules
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

logger = logging.getLogger(__name__)

# Allow small helper classes and avoid noisy warnings for simple dataclasses
# that intentionally expose few public methods.

# Some small helper classes in this module intentionally expose a single
# public method. Suppress the too-few-public-methods warning for these
# lightweight types to avoid noisy linting while preserving real issues.

T = TypeVar("T")

__all__ = [
    "ConfigProfile",
    "ConfigRule",
    "ConfigValidator",
    "BaseConfig",
    "ConfigSource",
    "ConfigSourceRegistry",
]

class ConfigProfile(Enum):
    """Configuration profiles for different environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def from_string(cls, value: str) -> ConfigProfile:
        """Create profile from string.

        Parameters
        ----------
        value : str
            Profile name (case-insensitive)

        Returns
        -------
        ConfigProfile
            Corresponding profile enum value

        Raises
        ------
        ValueError
            If profile name not recognized
        """
        try:
            return cls[value.upper()]
        except KeyError as exc:
            valid = [p.value for p in cls]
            raise ValueError(
                f"Invalid profile '{value}'. Valid profiles: {valid}"
            ) from exc

# Type alias for validator callables to improve type inference
Validator = Callable[[Any], bool]

@dataclass
class ConfigRule:
    """Configuration validation rule.

    Attributes
    ----------
    key : str
        Configuration key to validate
    required : bool
        Whether value is required (not None)
    expected_type : Type, optional
        Expected type of value
    validators : list[Callable], optional
        Custom validation functions
    description : str
        Description of rule
    """

    key: str
    required: bool = False
    expected_type: type[Any] | None = None
    validators: list[Validator] = field(
        default_factory=lambda: cast(list[Validator], [])
    )
    description: str = ""

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """Validate a configuration value.

        Parameters
        ----------
        value : Any
            Value to validate

        Returns
        -------
        tuple[bool, str | None]
            (is_valid, error_message)

        """
        if value is None:
            if self.required:
                return False, f"Required configuration missing: {self.key}"
            return True, None

        if self.expected_type and not isinstance(value, self.expected_type):
            return False, (
                f"Configuration {self.key} should be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        for validator in self.validators:
            try:
                if not validator(value):
                    return False, f"Validation failed for {self.key}: {value}"
            except Exception as e:
                # validators are user-provided; treat exceptions as validation
                # failures and include the exception message in the error.
                return False, f"Validation error for {self.key}: {e}"

        return True, None

class ConfigValidator:
    """Validates configuration against defined rules.

    Attributes
    ----------
    rules : dict[str, ConfigRule]
        Mapping of config keys to validation rules
    """

    def __init__(self) -> None:
        """Initialize validator."""
        self.rules: dict[str, ConfigRule] = {}

    def add_rule(self, rule: ConfigRule) -> ConfigValidator:
        """Add validation rule.

        Parameters
        ----------
        rule : ConfigRule
            Rule to add

        Returns
        -------
        ConfigValidator
            Self for chaining
        """
        self.rules[rule.key] = rule
        return self

    def add_rules(self, rules: list[ConfigRule]) -> ConfigValidator:
        """Add multiple validation rules.

        Parameters
        ----------
        rules : list[ConfigRule]
            Rules to add

        Returns
        -------
        ConfigValidator
            Self for chaining
        """
        for rule in rules:
            self.add_rule(rule)
        return self

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration to validate

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_errors)
        """
        errors: list[str] = []

        for key, rule in self.rules.items():
            value = config.get(key)
            is_valid, error = rule.validate(value)

            if not is_valid:
                # error is str | None from ConfigRule.validate, but when
                # is_valid is False we expect a string message. Cast to str
                # to satisfy static analysis.
                errors.append(cast(str, error))
                logger.error(cast(str, error))

        return (not errors), errors

    def clear(self) -> None:
        """Clear all rules."""
        self.rules.clear()

class ConfigSource(ABC):
    """Abstract base for configuration sources.

    Configuration can be loaded from various sources (files, environment, etc).
    Implementations should override load() to provide source-specific loading.
    """

    @abstractmethod
    def load(self) -> dict[str, Any]:
        """Load configuration.

        Returns
        -------
        dict[str, Any]
            Loaded configuration dictionary
        """

class ConfigSourceRegistry:
    """Registry for configuration sources.

    Provides factory methods for common source types and manages source instances.

    Examples
    --------
    >>> registry = ConfigSourceRegistry()
    >>> json_source = registry.create_json_source("config.json")
    >>> env_source = registry.create_env_source("APP_")
    """

    @staticmethod
    def create_json_source(path: str | Path) -> ConfigSource:
        """Create JSON configuration source.

        Parameters
        ----------
        path : str | Path
            Path to JSON file

        Returns
        -------
        ConfigSource
            JSON source instance

        """

        class _JsonSourceImpl(ConfigSource):
            def __init__(self, p: str | Path) -> None:
                self._path = Path(p)

            def load(self) -> dict[str, Any]:
                if not self._path.exists():
                    logger.warning("JSON config file not found: %s", self._path)
                    return {}
                try:
                    with open(self._path, encoding="utf-8") as f:
                        raw = json.load(f)
                except (json.JSONDecodeError, OSError, ValueError) as e:
                    logger.error("Failed to load JSON config %s: %s", self._path, e)
                    return {}
                if not isinstance(raw, dict):
                    logger.warning(
                        "JSON config at %s did not contain a mapping; ignoring",
                        self._path,
                    )
                    return {}
                return cast(dict[str, Any], raw)

        return _JsonSourceImpl(path)

    @staticmethod
    def create_yaml_source(path: str | Path) -> ConfigSource:
        """Create YAML configuration source.

        Parameters
        ----------
        path : str | Path
            Path to YAML file

        Returns
        -------
        ConfigSource
            YAML source instance

        """

        class _YamlSourceImpl(ConfigSource):
            def __init__(self, p: str | Path) -> None:
                self._path = Path(p)

            def load(self) -> dict[str, Any]:
                if yaml is None:
                    raise RuntimeError(
                        "PyYAML is not installed; cannot load YAML configs"
                    )
                if not self._path.exists():
                    logger.warning("YAML config file not found: %s", self._path)
                    return {}
                try:
                    with open(self._path, encoding="utf-8") as f:
                        raw = yaml.safe_load(f)
                except (yaml.YAMLError, OSError, ValueError) as e:
                    logger.error("Failed to load YAML config %s: %s", self._path, e)
                    return {}
                if not isinstance(raw, dict):
                    logger.warning(
                        "YAML config at %s did not contain a mapping; ignoring",
                        self._path,
                    )
                    return {}
                return cast(dict[str, Any], raw)

        return _YamlSourceImpl(path)

    @staticmethod
    def create_env_source(prefix: str = "APP_") -> ConfigSource:
        """Create environment variable configuration source.

        Parameters
        ----------
        prefix : str
            Environment variable prefix

        Returns
        -------
        ConfigSource
            Environment source instance
        """

        class _EnvSourceImpl(ConfigSource):
            def __init__(self, pfx: str = "APP_") -> None:
                self._prefix = pfx

            def load(self) -> dict[str, Any]:
                config: dict[str, Any] = {}
                for key, value in os.environ.items():
                    if key.startswith(self._prefix):
                        config_key = key[len(self._prefix) :].lower().replace("_", ".")
                        try:
                            config[config_key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            config[config_key] = value
                logger.info("Loaded %d settings from environment", len(config))
                return config

        return _EnvSourceImpl(prefix)

class BaseConfig(ABC):
    """Abstract base class for configuration objects.

    Provides shared configuration management functionality including:
    - Dictionary-style access with dot notation
    - Validation against rules
    - Profile management
    - Merge and override capabilities
    - Source-based loading

    This base class consolidates common patterns from ConfigManager and
    ConfigBuilder to reduce code duplication while providing consistent behavior.

    Attributes
    ----------
    _config : dict[str, Any]
        Main configuration dictionary
    _defaults : dict[str, Any]
        Default configuration values
    _overrides : dict[str, Any]
        Runtime override values
    _validator : ConfigValidator
        Validation rule manager
    _profile : ConfigProfile
        Current environment profile
    """

    def __init__(self) -> None:
        """Initialize configuration."""
        self._config: dict[str, Any] = {}
        self._defaults: dict[str, Any] = {}
        self._overrides: dict[str, Any] = {}
        self._validator = ConfigValidator()
        self._profile = ConfigProfile.DEVELOPMENT
        logger.debug("Initialized %s", self.__class__.__name__)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation.

        Supports nested access using dot notation: "section.subsection.key"

        Parameters
        ----------
        key : str
            Configuration key (dot notation supported)
        default : Any, optional
            Default value if not found

        Returns
        -------
        Any
            Configuration value or default
        """
        value = self._config

        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return self._defaults.get(key, default)

        return value

    def set(self, key: str, value: Any) -> BaseConfig:
        """Set configuration value with dot notation.

        Supports nested setting using dot notation: "section.subsection.key"

        Parameters
        ----------
        key : str
            Configuration key (dot notation supported)
        value : Any
            Value to set

        Returns
        -------
        BaseConfig
            Self for chaining
        """
        parts = key.split(".")

        # Set in overrides
        current = self._overrides
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

        # Also set in main config
        current = self._config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

        logger.debug("Configuration set: %s=%s", key, value)
        return self

    def set_default(self, key: str, value: Any) -> BaseConfig:
        """Set default configuration value.

        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Default value

        Returns
        -------
        BaseConfig
            Self for chaining
        """
        self._defaults[key] = value
        return self

    def set_defaults(self, **kwargs: Any) -> BaseConfig:
        """Set multiple default values.

        Parameters
        ----------
        **kwargs : Any
            Default key-value pairs

        Returns
        -------
        BaseConfig
            Self for chaining
        """
        self._defaults.update(kwargs)
        return self

    def add_rule(self, rule: ConfigRule) -> BaseConfig:
        """Add validation rule.

        Parameters
        ----------
        rule : ConfigRule
            Rule to add

        Returns
        -------
        BaseConfig
            Self for chaining
        """
        self._validator.add_rule(rule)
        return self

    def add_rules(self, rules: list[ConfigRule]) -> BaseConfig:
        """Add multiple validation rules.

        Parameters
        ----------
        rules : list[ConfigRule]
            Rules to add

        Returns
        -------
        BaseConfig
            Self for chaining
        """
        self._validator.add_rules(rules)
        return self

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_errors)
        """
        return self._validator.validate(self._config)

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        is_valid, _ = self.validate()
        return is_valid

    def load_profile(self, profile: str | ConfigProfile) -> BaseConfig:
        """Set configuration profile.

        Parameters
        ----------
        profile : str | ConfigProfile
            Profile name or enum value

        Returns
        -------
        BaseConfig
            Self for chaining

        """
        if isinstance(profile, str):
            profile = ConfigProfile.from_string(profile)

        self._profile = profile
        logger.info("Configuration profile set to: %s", profile.value)
        return self

    def get_profile(self) -> ConfigProfile:
        """Get current configuration profile.

        Returns
        -------
        ConfigProfile
            Current profile
        """
        return self._profile

    def get_all(self) -> dict[str, Any]:
        """Get entire configuration.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary (copy)
        """
        return self._config.copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary
        """
        return self._config.copy()

    def _merge_config(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Recursively merge source configuration into target.

        Parameters
        ----------
        target : dict[str, Any]
            Target configuration
        source : dict[str, Any]
            Source configuration to merge
        """
        for key, value in source.items():
            if (
                isinstance(value, dict)
                and key in target
                and isinstance(target[key], dict)
            ):
                # Cast both sides to dict[str, Any] so the type checker knows
                # we are passing the correct types to the recursive call.
                self._merge_config(
                    cast(dict[str, Any], target[key]),
                    cast(dict[str, Any], value),
                )
            else:
                target[key] = value

    def __repr__(self) -> str:
        """Return string representation."""
        config_count = len(self._config)
        profile = self._profile.value
        return f"{self.__class__.__name__}(profile={profile}, settings={config_count})"

    def __str__(self) -> str:
        """Return human-readable string."""
        return self.__repr__()
