"""Configuration management for the processing module.

Provides centralized configuration for backend verbosity and logging.
"""

from __future__ import annotations

import logging
import os

__all__ = ["ProcessingConfig"]

logger = logging.getLogger(__name__)


class ProcessingConfig:
    """Configuration manager for processing operations.

    Manages settings like backend verbosity and logging control,
    initialized from environment variables.
    """

    _instance: "ProcessingConfig | None" = None

    def __new__(cls) -> ProcessingConfig:
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration (only once)."""
        if self._initialized:
            return
        self._initialized = True

        self._backend_verbose = os.environ.get(
            "RESAMPLE_BACKEND_VERBOSE", "0"
        ).lower() in ("1", "true", "yes")
        self._module_logger = logging.getLogger("src.processing")
        if self._backend_verbose:
            self._module_logger.setLevel(logging.DEBUG)

    @property
    def backend_verbose(self) -> bool:
        """Check if backend verbose logging is enabled."""
        return self._backend_verbose

    @backend_verbose.setter
    def backend_verbose(self, enabled: bool) -> None:
        """Set backend verbose logging.

        Args:
            enabled: True to enable verbose logging
        """
        self._backend_verbose = bool(enabled)
        if enabled:
            self._module_logger.setLevel(logging.DEBUG)
            self._module_logger.debug("Backend verbose logging enabled")
        else:
            self._module_logger.setLevel(logging.INFO)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
