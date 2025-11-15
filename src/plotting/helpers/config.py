"""Plotting configuration and defaults.

Centralizes all plotting parameters to replace scattered plot_kwargs dictionaries.
This makes configurations easier to understand, test, and extend.

Also provides matplotlib initialization utilities.
"""

from dataclasses import dataclass, field
from typing import Any, cast
import logging
import matplotlib.pyplot as plt
import numpy as np
from types import ModuleType


@dataclass
class PlotConfig:
    """Central configuration for all plotting operations.

    This class replaces scattered plot_kwargs dictionaries with a unified,
    type-safe configuration object.
    """

    # Scale and labeling
    k_scale: float = 1.0
    k_label: str = "K"
    k_unit: str = ""

    # Visual properties
    cmap: str = "RdBu"
    is_categorical: bool = False
    # When plotting categorical data, n_categories and category_labels
    # allow specifying the number of categories and human-friendly labels
    n_categories: int | None = None
    category_labels: dict[int, str] | None = None
    show_colorbar: bool = True

    # Axis styling
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    fontsize_title: int = 12
    fontsize_labels: int = 10

    # Grid
    grid: bool = True
    grid_alpha: float = 0.3

    # Colorbar
    colorbar_label: str = "Value"
    percentile: float = 99.5

    # Additional kwargs for matplotlib
    # Use Any for extra kwargs to avoid partially-unknown dict typing from callers
    extra_kwargs: dict[str, Any] = field(
        default_factory=lambda: cast(dict[str, Any], {})
    )

    @classmethod
    def default(cls) -> "PlotConfig":
        """Create a PlotConfig with default values."""
        return cls()

    @classmethod
    def for_categorical(cls) -> "PlotConfig":
        """Create a PlotConfig optimized for categorical data."""
        return cls(is_categorical=True, cmap="tab10")

    @classmethod
    def for_seismic(cls, k_scale: float = 1.0, k_unit: str = "ms") -> "PlotConfig":
        """Create a PlotConfig optimized for seismic data."""
        return cls(
            k_scale=k_scale,
            k_label="Time",
            k_unit=k_unit,
            cmap="seismic",
            colorbar_label="Amplitude",
        )

    @classmethod
    def for_attributes(
        cls, attribute_name: str = "Attribute", cmap: str = "viridis"
    ) -> "PlotConfig":
        """Create a PlotConfig optimized for rock physics attributes."""
        return cls(
            cmap=cmap,
            colorbar_label=attribute_name,
            fontsize_title=11,
        )

    def update(self, **kwargs: Any) -> "PlotConfig":
        """Create a new PlotConfig with updated values.

        Args:
            **kwargs: Fields to update

        Returns:
            New PlotConfig with updated values
        """
        import dataclasses

        current_dict = dataclasses.asdict(self)
        current_dict.update(kwargs)
        return PlotConfig(**current_dict)

    def to_imshow_kwargs(self) -> dict[str, Any]:
        """Convert config to kwargs suitable for imshow."""
        return {"cmap": self.cmap, "origin": "upper", **self.extra_kwargs}

    def __repr__(self) -> str:
        """String representation showing main config values."""
        return (
            f"PlotConfig(k_scale={self.k_scale}, cmap={self.cmap}, "
            f"is_categorical={self.is_categorical}, title='{self.title}')"
        )


# ============================================================================
# Matplotlib Initialization Utilities
# ============================================================================

__all__ = [
    "PlotConfig",
    "setup_matplotlib",
    "init_plotting",
]


def setup_matplotlib(backend: str | None = "Agg") -> None:
    """Setup matplotlib with backend and default configuration.

    Args:
        backend: Matplotlib backend to use (default: "Agg")
    """
    import matplotlib

    if backend:
        try:
            matplotlib.use(backend)
        except Exception:
            pass

    # Apply standard matplotlib defaults
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


def init_plotting(backend: str | None = "Agg") -> tuple[ModuleType, ModuleType]:
    """Initialize matplotlib and return (plt, np) for convenience.

    Args:
        backend: Matplotlib backend to use (default: "Agg")

    Returns:
        Tuple of (matplotlib.pyplot, numpy)
    """
    setup_matplotlib(backend)
    return plt, np


# Module logger
logger = logging.getLogger(__name__)
