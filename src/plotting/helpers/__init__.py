"""Helper modules for plotting under src.plotting.helpers.

Modules:
- base: BasePlotter abstract class
- config: PlotConfig and matplotlib initialization
- components: Reusable plotting components (SliceExtractor, DataNormalizer, etc.)
- formatting: Formatting utilities
"""

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.config import PlotConfig, init_plotting, setup_matplotlib
from src.plotting.helpers.components import (
    SliceExtractor,
    DataNormalizer,
    AxisStyler,
    ImageRenderer,
)

__all__ = [
    "BasePlotter",
    "PlotConfig",
    "init_plotting",
    "setup_matplotlib",
    "SliceExtractor",
    "DataNormalizer",
    "AxisStyler",
    "ImageRenderer",
]
