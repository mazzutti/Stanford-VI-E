"""Configuration Management System for Dynamic Application Configuration

This module provides comprehensive configuration management with support for
multiple formats, runtime overrides, validation, and profile-based configurations.

Consolidates configuration management by inheriting from BaseConfig,
eliminating duplicated get/set/validate logic while keeping specialized
features like source-based loading and profile management.

Patterns Used:
  - Strategy: Different config sources (YAML, JSON, ENV)
  - Factory: Create configs from different sources
  - Composite: Multiple sources with merging

Example:
    >>> from src.analysis.config_manager import ConfigManager
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
    >>> if manager.is_valid():
    ...     app.start(manager)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import yaml

from src.core.configuration import BaseConfig, ConfigProfile, ConfigRule

logger = logging.getLogger(__name__)

# Many tiny source classes are intentionally minimal; suppress the
# too-few-public-methods warning for these light-weight types.


__all__ = [
    "ConfigSource",
    "ConfigManager",
    "EnvironmentSource",
    "JsonSource",
    "YamlSource",
]


class ConfigSource(ABC):
    """Abstract base for configuration sources."""

    @abstractmethod
    def load(self) -> dict[str, Any]:
        """Load configuration.

        Returns:
            Configuration dictionary
        """


class EnvironmentSource(ConfigSource):
    """Load configuration from environment variables."""

    def __init__(self, prefix: str = "APP_"):
        """Initialize environment source.

        Args:
            prefix: Environment variable prefix
        """
        self.prefix = prefix

    def load(self) -> dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        config: dict[str, Any] = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to lowercase with dots
                config_key = key[len(self.prefix) :].lower().replace("_", ".")

                # Try to parse value as JSON
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config[config_key] = value

        logger.info("Loaded %d settings from environment", len(config))
        return config


class JsonSource(ConfigSource):
    """Load configuration from JSON file."""

    def __init__(self, path: str | Path):
        """Initialize JSON source.

        Args:
            path: Path to JSON file
        """
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        """Load configuration from JSON file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file not found
            json.JSONDecodeError: If file invalid JSON
        """
        if not self.path.exists():
            logger.warning("JSON config file not found: %s", self.path)
            return {}

        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            logger.warning(
                "JSON config at %s did not contain a mapping; ignoring", self.path
            )
            return {}

        config = cast(dict[str, Any], raw)

        logger.info("Loaded configuration from %s", self.path)
        return config


class YamlSource(ConfigSource):
    """Load configuration from YAML file."""

    def __init__(self, path: str | Path):
        """Initialize YAML source.

        Args:
            path: Path to YAML file
        """
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file not found
            yaml.YAMLError: If file invalid YAML
        """

        if not self.path.exists():
            logger.warning("YAML config file not found: %s", self.path)
            return {}

        with open(self.path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            logger.warning(
                "YAML config at %s did not contain a mapping; ignoring", self.path
            )
            return {}

        config = cast(dict[str, Any], raw)

        logger.info("Loaded configuration from %s", self.path)
        return config


class ConfigManager(BaseConfig):
    """Centralized configuration management with validation and profiles.

    Inherits from BaseConfig to leverage shared configuration functionality
    (get/set/validate/profiles) while adding specialized features for
    loading configurations from multiple sources.
    """

    def __init__(self) -> None:
        """Initialize configuration manager."""
        super().__init__()
        self._sources: list[ConfigSource] = []
        logger.info("ConfigManager initialized")

    # ConfigManager intentionally provides small source wrapper classes; keep
    # the implementations concise to remain easily testable.

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        file_type: str | None = None,
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
            source: ConfigSource = JsonSource(path)
        elif file_type in ("yaml", "yml"):
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
        # add_rule is now inherited from BaseConfig via _validator
        super().add_rule(rule)
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
            except Exception as exc:
                # Configuration sources may fail for many reasons (I/O,
                # parsing, user-provided loader errors). Log and continue so
                # that one bad source does not prevent other sources from
                # contributing to the final configuration.
                logger.error("Error loading config from %s: %s", source, exc)

        # Apply overrides
        self._merge_config(self._config, self._overrides)

        logger.info("Configuration reloaded")
        return self

    def get_profile(self) -> ConfigProfile:
        """Get current configuration profile.

        Returns:
            Current profile
        """
        return self._profile

    def get_all(self) -> dict[str, Any]:
        """Get entire configuration dictionary.

        Returns:
            Configuration dictionary (copy)
        """
        return self._config.copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def _merge_config(self, target: dict[str, Any], source: dict[str, Any]) -> None:
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
                # Cast both sides to dict[str, Any] so the type checker knows they are mappings
                self._merge_config(
                    cast(dict[str, Any], target[key]), cast(dict[str, Any], value)
                )
            else:
                target[key] = value

    def __repr__(self) -> str:
        return (
            f"ConfigManager("
            f"profile={self._profile.value}, "
            f"settings={len(self._config)})"
        )
