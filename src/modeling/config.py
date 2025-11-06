"""Configuration and defaults for AVO modeling pipeline.

Centralizes all default values, making the pipeline simpler and easier to configure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.io.grid import GridSpec
from src.signal import RickerWavelet

__all__ = ["ModelingDefaults", "ModelingConfig"]


@dataclass(frozen=True)
class ModelingDefaults:
    """Default parameters for the complete modeling pipeline.

    Provides sensible defaults for all pipeline stages: data loading,
    resampling, and synthesis. Can be overridden when needed.
    """

    # Grid specification
    grid_shape: tuple[int, int, int] = (150, 200, 200)
    grid_dz: float = 1.0
    grid_dt: float = 0.001

    # Dataset loading
    data_path: str = "."
    vp_folder: str = "P-wave Velocity"
    vs_folder: str = "S-wave Velocity"
    rho_folder: str = "Density"
    facies_folder: str = "Facies"

    # AVO synthesis
    angles: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)
    peak_frequency: float = 26.0

    # Cache
    cache_dir: str = ".cache"

    @property
    def grid_spec(self) -> GridSpec:
        """Create GridSpec from defaults."""
        return GridSpec(self.grid_shape, dz=self.grid_dz, dt=self.grid_dt)

    @property
    def file_map(self) -> dict[str, str]:
        """Create file mapping from defaults."""
        return {
            "vp": self.vp_folder,
            "vs": self.vs_folder,
            "rho": self.rho_folder,
            "facies": self.facies_folder,
        }

    @property
    def wavelet(self) -> NDArray[np.floating[Any]]:
        """Create default Ricker wavelet."""
        ricker = RickerWavelet(f_peak=self.peak_frequency, dt=self.grid_dt)
        return np.asarray(ricker.samples, dtype=np.float64)


@dataclass
class ModelingConfig:
    """Configuration for a modeling pipeline run.

    Allows customization of all pipeline parameters while maintaining
    sensible defaults through ModelingDefaults.
    """

    defaults: ModelingDefaults = field(default_factory=ModelingDefaults)
    add_noise: bool = False
    use_quality_weighting: bool = True
    snr_db: float = 20.0
    cache_enabled: bool = True
