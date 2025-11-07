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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type, Union
from pathlib import Path
import logging
import json
import os

from src.core import BaseConfig, ConfigProfile, ConfigRule

logger = logging.getLogger(__name__)

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


class ConfigManager(BaseConfig):
    """Centralized configuration management with validation and profiles.

    Inherits from BaseConfig to leverage shared configuration functionality
    (get/set/validate/profiles) while adding specialized features for
    loading configurations from multiple sources.
    """

    def __init__(self) -> None:
        """Initialize configuration manager."""
        super().__init__()
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
            except Exception as e:
                logger.error(f"Error loading config from {source}: {e}")

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
