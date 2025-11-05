"""Rock physics convenience model.

Small wrapper to hold vp/vs/rho/facies together and centralize unit handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.io.grid import GridSpec
from src.processing.velocity import VelocityModel
from src.processing.materials import VsModel, DensityModel
from src.io.disk_cache import DiskCache
from src.utils.quantity import Quantity
from src.processing._singleton import SingletonFactory
import logging

logger = logging.getLogger(__name__)

__all__ = ["RockPhysicsModel", "get_rock_physics_model"]


@dataclass
class RockPhysicsModel:
    vp: Optional[np.ndarray]
    vs: Optional[np.ndarray]
    rho: Optional[np.ndarray]
    facies: Optional[np.ndarray]
    grid_spec: GridSpec
    # Caches for derived attributes (LRU via OrderedDict)
    _derived_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _refl_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    # Technique-specific caches are not part of this module's public API
    # Optional disk cache (shared) for expensive results
    disk_cache: Optional[DiskCache] = field(default=None, init=True, repr=False)

    @classmethod
    def from_props(cls, props: dict, grid_spec: GridSpec) -> "RockPhysicsModel":
        # Wrap numeric arrays in Quantity when available. Default unit guesses
        # are conservative; Quantity.to() can be used by callers to normalize.
        vp = props.get("vp")
        vs = props.get("vs")
        rho = props.get("rho")
        facies = props.get("facies")

        vp_q = Quantity(vp.copy(), "m/s") if vp is not None else None
        vs_q = Quantity(vs.copy(), "m/s") if vs is not None else None
        rho_q = Quantity(rho.copy(), "kg/m3") if rho is not None else None

        return cls(
            vp=vp_q,
            vs=vs_q,
            rho=rho_q,
            facies=(facies.copy() if facies is not None else None),
            grid_spec=grid_spec,
        )

    def ensure_units(self) -> None:
        """Ensure vp/vs/rho are in SI units (m/s, kg/m^3).

        This delegates to the small helper wrappers so the heuristics live in one
        place and callers remain concise.
        """
        if self.vp is not None:
            # ensure Quantity is in m/s
            if not isinstance(self.vp, Quantity):
                self.vp = Quantity(self.vp, "m/s")
            self.vp = self.vp.to("m/s", copy=True)
            # validate numeric array via VelocityModel
            vm = VelocityModel(vp=self.vp.array, grid_spec=self.grid_spec)
            vm.validate()

        if self.vs is not None:
            if not isinstance(self.vs, Quantity):
                self.vs = Quantity(self.vs, "m/s")
            self.vs = self.vs.to("m/s", copy=True)
            vsm = VsModel(self.vs.array)
            vsm.validate()
            # keep as Quantity
            self.vs = Quantity(vsm.vs, "m/s")

        if self.rho is not None:
            if not isinstance(self.rho, Quantity):
                self.rho = Quantity(self.rho, "kg/m3")
            self.rho = self.rho.to("kg/m3", copy=True)
            drm = DensityModel(self.rho.array)
            drm.validate()
            self.rho = Quantity(drm.rho, "kg/m3")

        # Any change to underlying properties invalidates derived caches
        self.invalidate_cache()

    def compute_ai(self) -> np.ndarray:
        raise AttributeError("compute_ai is not available")

    def reflectivity_from_props(self) -> np.ndarray:
        """Compute reflectivity from current rock properties.

        Use `src.signal.reflectivity` helpers for reflectivity calculations.
        """

    def invalidate_cache(self) -> None:
        """Invalidate internal caches for derived attributes."""
        self._derived_cache = None
        self._refl_cache = None
        # keep only caches used by current codepaths

    def to_props_dict(self) -> dict:
        out = {}
        if self.vp is not None:
            out["vp"] = self.vp.array if isinstance(self.vp, Quantity) else self.vp
        if self.vs is not None:
            out["vs"] = self.vs.array if isinstance(self.vs, Quantity) else self.vs
        if self.rho is not None:
            out["rho"] = self.rho.array if isinstance(self.rho, Quantity) else self.rho
        if self.facies is not None:
            out["facies"] = self.facies
        return out


# Create a convenient module-level default model lazily (preserve assignability)
def _create_placeholder_rock_physics_model() -> RockPhysicsModel:
    # Build a minimal placeholder RockPhysicsModel similar to the previous
    # proxy's behavior. Keep the GridSpec construction local to avoid
    # import-time side-effects.
    _placeholder_grid = GridSpec((0, 0, 0), dz=1.0, dt=0.001)
    return RockPhysicsModel(
        vp=None, vs=None, rho=None, facies=None, grid_spec=_placeholder_grid
    )


# Module-level singleton for the RockPhysicsModel
_rock_physics_factory: SingletonFactory[RockPhysicsModel] = SingletonFactory(
    _create_placeholder_rock_physics_model
)


def get_rock_physics_model(
    instance: RockPhysicsModel | None = None,
) -> RockPhysicsModel:
    """Return provided RockPhysicsModel or the module-level lazy singleton.

    Args:
        instance: Optional RockPhysicsModel to use instead of the singleton

    Returns:
        The provided instance or the module-level lazy singleton
    """
    return _rock_physics_factory.get(instance)
