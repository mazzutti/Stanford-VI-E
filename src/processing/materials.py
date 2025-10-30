"""Small wrappers for S-wave velocity and density models.

Provide consistent unit-handling similar to `VelocityModel` for Vp.
These wrappers are intentionally tiny and only provide the common conversions
used in this codebase (km/s -> m/s for velocities, g/cm^3 -> kg/m^3 for
density) plus basic validation.
"""

from dataclasses import dataclass
from numpy.typing import ArrayLike

import numpy as np
from src.utils.units import UnitRegistry
import logging

__all__ = ["VsModel", "DensityModel"]

# Module logger
logger = logging.getLogger(__name__)


@dataclass
class VsModel:
    vs: ArrayLike

    def to_m_per_s(self) -> None:
        """Convert vs in-place from km/s to m/s if values appear to be in km/s."""
        out, converted = UnitRegistry.ensure_m_per_s(self.vs, copy_on_convert=False)
        self.vs = out

    def ensure_m_per_s(self) -> bool:
        """Convert vs to m/s if needed and return True if conversion occurred."""
        out, converted = UnitRegistry.ensure_m_per_s(self.vs, copy_on_convert=False)
        self.vs = out
        return bool(converted)

    def validate(self) -> None:
        if not np.all(np.isfinite(self.vs)):
            raise ValueError("vs contains non-finite values")
        if np.any(self.vs <= 0.0):
            raise ValueError("vs contains non-positive values")


@dataclass
class DensityModel:
    rho: ArrayLike

    def to_kg_per_m3(self) -> None:
        """Convert density from g/cm^3 to kg/m^3 if needed."""
        out, converted = UnitRegistry.ensure_kg_per_m3(self.rho, copy_on_convert=False)
        self.rho = out

    def ensure_kg_per_m3(self) -> bool:
        """Convert rho to kg/m3 if needed and return True if conversion occurred."""
        out, converted = UnitRegistry.ensure_kg_per_m3(self.rho, copy_on_convert=False)
        self.rho = out
        return bool(converted)

    def validate(self) -> None:
        if not np.all(np.isfinite(self.rho)):
            raise ValueError("rho contains non-finite values")
        if np.any(self.rho <= 0.0):
            raise ValueError("rho contains non-positive values")
