"""Base Plotter class providing common functionality for all plotters.

Simplified base class using composition with components for better separation
of concerns. All specific plotter implementations inherit from this class.
"""

import logging
from abc import ABC
from typing import Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BasePlotter(ABC):
    """Abstract base class for all plotting implementations.

    Uses composition with specialized components for better testability and
    maintainability. Provides minimal shared functionality.

    Attributes:
        logger: Logger instance for this plotter
        backend: Matplotlib backend name
    """

    def __init__(self, backend: Optional[str] = "Agg"):
        """Initialize the plotter.

        Args:
            backend: Matplotlib backend to use (default: "Agg")
        """
        self.logger = logging.getLogger(self.__class__.__module__)
        self.backend = backend
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """Initialize matplotlib with the configured backend."""
        import matplotlib

        if self.backend:
            try:
                matplotlib.use(self.backend)
            except Exception:
                pass
        self._configure_matplotlib_defaults()

    @staticmethod
    def _configure_matplotlib_defaults() -> None:
        """Apply standard matplotlib defaults once."""
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "sans-serif",
        ]
        plt.rcParams["image.interpolation"] = "bilinear"
        plt.rcParams["image.resample"] = True
        plt.rcParams["image.composite_image"] = True

    def _log_debug(self, msg: str) -> None:
        """Log a debug message."""
        self.logger.debug(msg)

    def _log_info(self, msg: str) -> None:
        """Log an info message."""
        self.logger.info(msg)

    def _log_warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)
