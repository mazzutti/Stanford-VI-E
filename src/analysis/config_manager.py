"""Configuration Management System for Dynamic Application Configuration

This module provides comprehensive configuration management with support for
multiple formats, runtime overrides, validation, and profile-based configurations.

Patterns Used:
  - Configuration: Externalize settings
  - Strategy: Different config sources (YAML, JSON, ENV)
  - Validation: Enforce config constraints
  - Factory: Create configs from different sources

Example:
    >>> from src.analysis.config_manager import ConfigManager, ConfigProfile
    >>>
    >>> # Load from file
    >>> manager = ConfigManager.from_file("config/settings.yaml")
    >>>
    >>> # Get values
    >>> cache_size = manager.get("cache.size", 1000)
    >>>
    >>> # Override at runtime
    >>> manager.set("cache.size", 5000)
    >>>
    >>> # Use profiles
    >>> manager.load_profile("production")
    >>>
    >>> # Validate configuration
    >>> if manager.validate():
    ...     app.start(manager)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from enum import Enum
import os

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigProfile",
    "ConfigValidator",
    "ConfigSource",
    "ConfigManager",
    "EnvironmentSource",
    "JsonSource",
    "YamlSource",
]


class ConfigProfile(Enum):
    """Configuration profiles for different environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigRule:
    """Configuration validation rule."""

    key: str
    required: bool = False
    expected_type: Optional[Type] = None
    validators: List[Callable] = None
    description: str = ""

    def __post_init__(self):
        """Initialize validators list."""
        if self.validators is None:
            self.validators = []

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a configuration value.

        Args:
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
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
                return False, f"Validation error for {self.key}: {e}"

        return True, None


class ConfigValidator:
    """Validates configuration against defined rules."""

    def __init__(self):
        """Initialize validator."""
        self.rules: Dict[str, ConfigRule] = {}

    def add_rule(self, rule: ConfigRule) -> ConfigValidator:
        """Add validation rule.

        Args:
            rule: Configuration rule

        Returns:
            Self for chaining
        """
        self.rules[rule.key] = rule
        return self

    def validate(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for key, rule in self.rules.items():
            value = config.get(key)
            is_valid, error = rule.validate(value)

            if not is_valid:
                errors.append(error)
                logger.error(error)

        return len(errors) == 0, errors


class ConfigSource(ABC):
    """Abstract base for configuration sources."""

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration.

        Returns:
            Configuration dictionary
        """
        pass


class EnvironmentSource(ConfigSource):
    """Load configuration from environment variables."""

    def __init__(self, prefix: str = "APP_"):
        """Initialize environment source.

        Args:
            prefix: Environment variable prefix
        """
        self.prefix = prefix

    def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        config = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to lowercase with dots
                config_key = key[len(self.prefix) :].lower().replace("_", ".")

                # Try to parse value as JSON
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config[config_key] = value

        logger.info(f"Loaded {len(config)} settings from environment")
        return config


class JsonSource(ConfigSource):
    """Load configuration from JSON file."""

    def __init__(self, path: Union[str, Path]):
        """Initialize JSON source.

        Args:
            path: Path to JSON file
        """
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        """Load configuration from JSON file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file not found
            json.JSONDecodeError: If file invalid JSON
        """
        if not self.path.exists():
            logger.warning(f"JSON config file not found: {self.path}")
            return {}

        with open(self.path, "r") as f:
            config = json.load(f)

        logger.info(f"Loaded configuration from {self.path}")
        return config


class YamlSource(ConfigSource):
    """Load configuration from YAML file."""

    def __init__(self, path: Union[str, Path]):
        """Initialize YAML source.

        Args:
            path: Path to YAML file
        """
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file not found
            yaml.YAMLError: If file invalid YAML
        """
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, using JSON instead")
            return {}

        if not self.path.exists():
            logger.warning(f"YAML config file not found: {self.path}")
            return {}

        with open(self.path, "r") as f:
            config = yaml.safe_load(f) or {}

        logger.info(f"Loaded configuration from {self.path}")
        return config


class ConfigManager:
    """Centralized configuration management with validation and profiles."""

    def __init__(self):
        """Initialize configuration manager."""
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}
        self._validator = ConfigValidator()
        self._profile = ConfigProfile.DEVELOPMENT
        self._sources: List[ConfigSource] = []
        logger.info("ConfigManager initialized")

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        file_type: Optional[str] = None,
    ) -> ConfigManager:
        """Create manager from configuration file.

        Args:
            path: Path to config file
            file_type: File type ('json' or 'yaml'), auto-detected if None

        Returns:
            Configured ConfigManager instance
        """
        path = Path(path)

        if file_type is None:
            file_type = path.suffix.lstrip(".").lower()

        manager = cls()

        if file_type == "json":
            source = JsonSource(path)
        elif file_type == "yaml" or file_type == "yml":
            source = YamlSource(path)
        else:
            raise ValueError(f"Unknown config file type: {file_type}")

        manager._sources.append(source)
        manager.reload()

        return manager

    def add_source(self, source: ConfigSource) -> ConfigManager:
        """Add configuration source.

        Args:
            source: Configuration source

        Returns:
            Self for chaining
        """
        self._sources.append(source)
        return self

    def add_rule(self, rule: ConfigRule) -> ConfigManager:
        """Add validation rule.

        Args:
            rule: Configuration rule

        Returns:
            Self for chaining
        """
        self._validator.add_rule(rule)
        return self

    def set_default(self, key: str, value: Any) -> ConfigManager:
        """Set default configuration value.

        Args:
            key: Configuration key
            value: Default value

        Returns:
            Self for chaining
        """
        self._defaults[key] = value
        return self

    def reload(self) -> ConfigManager:
        """Reload configuration from sources.

        Returns:
            Self for chaining
        """
        self._config = self._defaults.copy()

        for source in self._sources:
            try:
                config = source.load()
                self._merge_config(self._config, config)
            except Exception as e:
                logger.error(f"Error loading config from {source}: {e}")

        # Apply overrides
        self._merge_config(self._config, self._overrides)

        logger.info("Configuration reloaded")
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation.

        Args:
            key: Configuration key (supports dot notation: "section.key")
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        value = self._config

        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                # Check defaults if not found in config
                default_value = self._defaults.get(key, default)
                return default_value

        return value

    def set(self, key: str, value: Any) -> ConfigManager:
        """Set configuration value with dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Returns:
            Self for chaining
        """
        parts = key.split(".")
        current = self._overrides

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

        # Also set in main config for nested access
        current = self._config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

        logger.debug(f"Configuration set: {key}={value}")
        return self

    def load_profile(self, profile: Union[str, ConfigProfile]) -> ConfigManager:
        """Load configuration profile.

        Args:
            profile: Profile name or ConfigProfile enum

        Returns:
            Self for chaining
        """
        if isinstance(profile, str):
            profile = ConfigProfile(profile.lower())

        self._profile = profile
        logger.info(f"Configuration profile set to: {profile.value}")

        return self

    def validate(self) -> bool:
        """Validate current configuration.

        Returns:
            True if valid, False otherwise
        """
        is_valid, errors = self._validator.validate(self._config)

        if is_valid:
            logger.info("Configuration validation passed")
        else:
            for error in errors:
                logger.error(error)

        return is_valid

    def get_profile(self) -> ConfigProfile:
        """Get current configuration profile.

        Returns:
            Current profile
        """
        return self._profile

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary.

        Returns:
            Configuration dictionary (copy)
        """
        return self._config.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def _merge_config(self, target: Dict, source: Dict) -> None:
        """Recursively merge source config into target.

        Args:
            target: Target configuration dictionary
            source: Source configuration dictionary
        """
        for key, value in source.items():
            if (
                isinstance(value, dict)
                and key in target
                and isinstance(target[key], dict)
            ):
                self._merge_config(target[key], value)
            else:
                target[key] = value

    def __repr__(self) -> str:
        return (
            f"ConfigManager("
            f"profile={self._profile.value}, "
            f"settings={len(self._config)})"
        )
