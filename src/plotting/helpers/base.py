"""Base Plotter class providing common functionality for all plotters.

Simplified base class using composition with components for better separation
of concerns. All specific plotter implementations inherit from this class.
"""

import logging
from abc import ABC


from src.plotting.helpers.config import setup_matplotlib

logger = logging.getLogger(__name__)


class BasePlotter(ABC):
    """Abstract base class for all plotting implementations.

    Uses composition with specialized components for better testability and
    maintainability. Provides minimal shared functionality.

    Attributes:
        logger: Logger instance for this plotter
        backend: Matplotlib backend name
    """

    def __init__(self, backend: str | None = "Agg"):
        """Initialize the plotter.

        Args:
            backend: Matplotlib backend to use (default: "Agg")
        """
        self.logger = logging.getLogger(self.__class__.__module__)
        self.backend = backend
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """Initialize matplotlib with the configured backend."""
        setup_matplotlib(self.backend)

    def _log_debug(self, msg: str) -> None:
        """Log a debug message."""
        self.logger.debug(msg)

    def _log_info(self, msg: str) -> None:
        """Log an info message."""
        self.logger.info(msg)

    def _log_warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)
