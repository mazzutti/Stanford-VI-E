"""Base Plotter class providing common functionality for all plotters.

This module consolidates common plotting patterns into a base class that
all specific plotter implementations can inherit from.
"""

from abc import ABC
import logging

logger = logging.getLogger(__name__)


class BasePlotter(ABC):
    """Abstract base class for all plotting implementations.

    Provides common initialization and utility methods for plotter subclasses.
    Implements the OOP pattern with consistent interface across all plotters.
    """

    def __init__(self):
        """Initialize the plotter with common defaults."""
        self.logger = logging.getLogger(self.__class__.__module__)

    def _log_debug(self, msg: str, *args):
        """Log a debug message."""
        self.logger.debug(msg, *args)

    def _log_info(self, msg: str, *args):
        """Log an info message."""
        self.logger.info(msg, *args)

    def _log_warning(self, msg: str, *args):
        """Log a warning message."""
        self.logger.warning(msg, *args)
