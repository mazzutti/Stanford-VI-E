"""Rock physics data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from src.io.disk_cache import DiskCache
from src.io.grid import GridSpec
from src.processing.rock_physics.cache import ModelCache
from src.utils.quantity import Quantity

# Some imports are intentionally ordered to avoid import-time side-effects
# and circular dependencies (e.g., DiskCache and ModelCache). Relax import
# ordering checks in this module so lint focuses on functional issues.


__all__ = ["RockPhysicsModel"]


import logging

logger = logging.getLogger(__name__)

# Some helper imports (materials converters and models) are deferred to
# method scope to prevent import-time side effects and circular imports.
# These deferred imports are intentional; disable the import-order
# warnings at module level with a short justification.


@dataclass
class RockPhysicsModel:
    """Core data model for rock physics properties.

    Holds vp, vs, rho, and facies along with grid spec and manages unit
    conversion. Caching is delegated to ModelCache.

    Attributes:
        vp: P-wave velocity (optional)
        vs: S-wave velocity (optional)
        rho: Density (optional)
        facies: Facies classification (optional)
        grid_spec: Grid specification
        disk_cache: Optional shared disk cache
    """

    vp: Quantity | NDArray[Any] | None
    vs: Quantity | NDArray[Any] | None
    rho: Quantity | NDArray[Any] | None
    facies: NDArray[Any] | None
    grid_spec: GridSpec
    disk_cache: DiskCache | None = field(default=None, init=True, repr=False)
    _cache: ModelCache = field(default_factory=ModelCache, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize cache after dataclass construction."""
        if self.disk_cache is not None:
            self._cache = ModelCache(disk_cache=self.disk_cache)
        else:
            self._cache = ModelCache()

    @classmethod
    def from_props(cls, props: dict[str, Any], grid_spec: GridSpec) -> RockPhysicsModel:
        """Create model from properties dictionary.

        Wraps numeric arrays in Quantity with conservative unit guesses.
        Callers can use Quantity.to() to normalize units.

        Args:
            props: Dictionary with keys 'vp', 'vs', 'rho', 'facies' (optional)
            grid_spec: Grid specification

        Returns:
            New RockPhysicsModel instance
        """
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

        Delegates to small helper wrappers so heuristics live in one place.
        Invalidates cache after any changes.
        """
        from src.processing.materials.properties import (
            DensityModel,
            VsModel,
        )
        from src.processing.materials.velocity import (
            VelocityModel,
        )

        if self.vp is not None:
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
            self.vs = Quantity(vsm.vs, "m/s")

        if self.rho is not None:
            if not isinstance(self.rho, Quantity):
                self.rho = Quantity(self.rho, "kg/m3")
            self.rho = self.rho.to("kg/m3", copy=True)
            drm = DensityModel(self.rho.array)
            drm.validate()
            self.rho = Quantity(drm.rho, "kg/m3")

        # Invalidate derived caches after unit changes
        self._cache.invalidate()

    def invalidate_cache(self) -> None:
        """Invalidate internal caches for derived attributes."""
        self._cache.invalidate()

    def to_props_dict(self) -> dict[str, NDArray[Any]]:
        """Export properties as a dictionary of numpy arrays."""
        out: dict[str, NDArray[Any]] = {}
        if self.vp is not None:
            out["vp"] = self.vp.array if isinstance(self.vp, Quantity) else self.vp
        if self.vs is not None:
            out["vs"] = self.vs.array if isinstance(self.vs, Quantity) else self.vs
        if self.rho is not None:
            out["rho"] = self.rho.array if isinstance(self.rho, Quantity) else self.rho
        if self.facies is not None:
            out["facies"] = self.facies
        return out
