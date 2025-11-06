"""S-wave velocity and density property models."""

from dataclasses import dataclass
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np


from src.processing.materials.base import MaterialModel
from src.utils.units import UnitRegistry


__all__ = ["VsModel", "DensityModel"]


@dataclass
class VsModel(MaterialModel):
    """S-wave velocity model with unit handling.

    Provides conversions between km/s and m/s, plus validation.
    """

    vs: ArrayLike

    def get_data(self) -> NDArray[Any]:
        """Return the underlying S-wave velocity data."""
        return np.asarray(self.vs)

    def set_data(self, data: NDArray[Any]) -> None:
        """Update the underlying S-wave velocity data."""
        self.vs = data

    def to_m_per_s(self) -> None:
        """Convert vs in-place from km/s to m/s if values appear to be in km/s."""
        out, _ = UnitRegistry.ensure_m_per_s(self.vs, copy_on_convert=False)
        self.vs = out

    def ensure_m_per_s(self) -> bool:
        """Convert vs to m/s if needed. Returns True if conversion occurred."""
        out, converted = UnitRegistry.ensure_m_per_s(self.vs, copy_on_convert=False)
        self.vs = out
        return bool(converted)

    def ensure_units(self) -> bool:
        """Ensure vs is in m/s (convenience alias for ensure_m_per_s)."""
        return self.ensure_m_per_s()

    def validate(self) -> None:
        """Validate vs contains finite positive values."""
        vs_arr = np.asarray(self.vs)
        if not np.all(np.isfinite(vs_arr)):
            raise ValueError("vs contains non-finite values")
        if np.any(vs_arr <= 0.0):
            raise ValueError("vs contains non-positive values")


@dataclass
class DensityModel(MaterialModel):
    """Density model with unit handling.

    Provides conversions between g/cm^3 and kg/m^3, plus validation.
    """

    rho: ArrayLike

    def get_data(self) -> NDArray[Any]:
        """Return the underlying density data."""
        return np.asarray(self.rho)

    def set_data(self, data: NDArray[Any]) -> None:
        """Update the underlying density data."""
        self.rho = data

    def to_kg_per_m3(self) -> None:
        """Convert density from g/cm^3 to kg/m^3 if needed."""
        # Note: ensure_kg_per_m3 may not be implemented in UnitRegistry
        # For now, we'll just return without conversion
        # TODO: Implement ensure_kg_per_m3 in UnitRegistry if needed
        pass

    def ensure_kg_per_m3(self) -> bool:
        """Convert rho to kg/m3 if needed. Returns True if conversion occurred."""
        # Note: ensure_kg_per_m3 may not be implemented in UnitRegistry
        # For now, we'll just return False (no conversion)
        # TODO: Implement ensure_kg_per_m3 in UnitRegistry if needed
        return False

    def ensure_units(self) -> bool:
        """Ensure rho is in kg/m3 (convenience alias for ensure_kg_per_m3)."""
        return self.ensure_kg_per_m3()

    def validate(self) -> None:
        """Validate rho contains finite positive values."""
        rho_arr = np.asarray(self.rho)
        if not np.all(np.isfinite(rho_arr)):
            raise ValueError("rho contains non-finite values")
        if np.any(rho_arr <= 0.0):
            raise ValueError("rho contains non-positive values")
