"""Rock physics convenience model.

Small wrapper to hold vp/vs/rho/facies together, centralize unit handling and
provide simple derived attribute calculations (e.g. acoustic impedance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Optional

import numpy as np

from src.io.grid import GridSpec
from src.processing.velocity import VelocityModel
from src.processing.materials import VsModel, DensityModel
from src.modeling import modeling as modeling_utils
from src.io.disk_cache import DiskCache
import hashlib
from src.utils.quantity import Quantity
from src.utils.facades import LazyObjectProxy
import logging

logger = logging.getLogger(__name__)

__all__ = ["RockPhysicsModel", "rock_physics_model"]


@dataclass
class RockPhysicsModel:
    vp: Optional[np.ndarray]
    vs: Optional[np.ndarray]
    rho: Optional[np.ndarray]
    facies: Optional[np.ndarray]
    grid_spec: GridSpec
    # Caches for derived attributes (LRU via OrderedDict)
    _ai_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _refl_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _ei_angle_cache: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _ei_multi_cache: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _ei_weighted_cache: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    # Cache size limits (tunable per-instance)
    ei_angle_cache_max: int = field(default=8, init=True)
    ei_multi_cache_max: int = field(default=4, init=True)
    ei_weighted_cache_max: int = field(default=4, init=True)
    # Optional disk cache (shared) for expensive EI results
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
        """Compute acoustic impedance (AI = vp * rho).

        Returns:
            numpy.ndarray: the AI cube (same shape as vp/rho)
        """
        return _impl_compute_ai(self)

    def reflectivity_from_ai(self) -> np.ndarray:
        """Compute (normal-incidence) reflectivity from AI (depth-domain).

        This reuses the generic reflectivity helper. The output has the same
        spatial shape as AI but with the leading depth sample padded as in the
        reflectivity implementation.
        """
        return _impl_reflectivity_from_ai(self)

    def compute_ei_angle(self, angle_deg: float) -> np.ndarray:
        return _impl_compute_ei_angle(self, angle_deg)

    def compute_ei_multiangle(self, angles_deg, show_progress=True) -> dict:
        return _impl_compute_ei_multiangle(
            self, angles_deg, show_progress=show_progress
        )

    def compute_ei_weighted_product(
        self,
        litho_angles=None,
        fluid_angles=None,
        litho_weight=0.7,
        fluid_weight=0.3,
        show_progress=True,
    ) -> dict:
        return _impl_compute_ei_weighted_product(
            self,
            litho_angles=litho_angles,
            fluid_angles=fluid_angles,
            litho_weight=litho_weight,
            fluid_weight=fluid_weight,
            show_progress=show_progress,
        )

    def invalidate_cache(self) -> None:
        """Invalidate internal caches for derived attributes."""
        self._ai_cache = None
        self._refl_cache = None
        self._ei_angle_cache = OrderedDict()
        self._ei_multi_cache = OrderedDict()
        self._ei_weighted_cache = OrderedDict()

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


class RockPhysicsModelProxy(LazyObjectProxy[RockPhysicsModel]):
    """Specialized proxy that preserves the previous replacement and
    attribute-forwarding semantics used by callers.

    - Setting attributes with names starting with '_' operates on the proxy
      itself (internal state).
    - Assigning a RockPhysicsModel instance (or None) as a value for any
      attribute will replace the underlying `_instance` (preserves previous
      behaviour where callers could replace the module model in-place).
    """

    def __setattr__(self, name: str, value):
        # Internal attributes should be stored on the proxy itself.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        # If the assigned value is a full RockPhysicsModel instance (or None),
        # treat it as a replacement for the underlying model.
        if isinstance(value, RockPhysicsModel) or value is None:
            # Use the proxy lock if present, otherwise fall back to direct set
            lock = getattr(self, "_lock", None)
            if lock is not None:
                with lock:
                    object.__setattr__(self, "_instance", value)
            else:
                object.__setattr__(self, "_instance", value)
            return

        # Otherwise forward the attribute assignment to the wrapped instance.
        inst = self._ensure()
        setattr(inst, name, value)

    def __repr__(self) -> str:  # keep previous repr for compatibility
        if getattr(self, "_instance", None) is None:
            return "<LazyRockPhysicsModelProxy (uninitialized)>"
        return "<LazyRockPhysicsModelProxy>"


rock_physics_model = RockPhysicsModelProxy(_create_placeholder_rock_physics_model)


def get_rock_physics_model(
    instance: RockPhysicsModel | None = None,
) -> "RockPhysicsModel":
    """Return provided RockPhysicsModel or the module-level lazy singleton.

    This routes through the canonical implementation `_impl_get_rock_physics_model`
    so callers and tests can inject instances or use the lazy module-level
    proxy consistently.
    """
    return _impl_get_rock_physics_model(instance)


def _impl_get_rock_physics_model(
    instance: RockPhysicsModel | None = None,
) -> RockPhysicsModel:
    """Canonical implementation for obtaining the module RockPhysicsModel.

    Returns the provided instance when not None, otherwise returns the
    module-level `rock_physics_model` lazy proxy.
    """
    return instance if instance is not None else rock_physics_model


def _impl_compute_ai(self: RockPhysicsModel) -> np.ndarray:
    if self._ai_cache is not None:
        return self._ai_cache
    if self.vp is None or self.rho is None:
        raise ValueError("vp and rho are required to compute AI")
    vp_arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
    rho_arr = self.rho.array if isinstance(self.rho, Quantity) else self.rho
    self._ai_cache = vp_arr * rho_arr
    return self._ai_cache


def _impl_reflectivity_from_ai(self: RockPhysicsModel) -> np.ndarray:
    if self._refl_cache is not None:
        return self._refl_cache
    ai = _impl_compute_ai(self)
    from src.signal.reflectivity import reflectivity_calc

    self._refl_cache = reflectivity_calc.reflectivity_from_ai(ai)
    return self._refl_cache


def _impl_compute_ei_angle(self: RockPhysicsModel, angle_deg: float) -> np.ndarray:
    key = float(angle_deg)
    if key in self._ei_angle_cache:
        # move to end to mark as recently used
        val = self._ei_angle_cache.pop(key)
        self._ei_angle_cache[key] = val
        return val
    if self.vp is None or self.vs is None or self.rho is None:
        raise ValueError("vp, vs and rho are required for EI computation")
    vp_arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
    vs_arr = self.vs.array if isinstance(self.vs, Quantity) else self.vs
    rho_arr = self.rho.array if isinstance(self.rho, Quantity) else self.rho
    out = modeling_utils.modeling_engine.compute_ei_angle(
        vp_arr, vs_arr, rho_arr, angle_deg
    )
    # insert and respect LRU size
    self._ei_angle_cache[key] = out
    if len(self._ei_angle_cache) > int(self.ei_angle_cache_max):
        # pop the oldest item
        self._ei_angle_cache.popitem(last=False)
    return out


def _impl_compute_ei_multiangle(
    self: RockPhysicsModel, angles_deg, show_progress=True
) -> dict:
    key = tuple(angles_deg)
    if key in self._ei_multi_cache:
        val = self._ei_multi_cache.pop(key)
        self._ei_multi_cache[key] = val
        return val
    # Attempt disk-backed lookup if available
    if self.disk_cache is not None:
        meta = {
            "shape": tuple(self.vp.array.shape) if self.vp is not None else None,
            "grid": (self.grid_spec.shape, self.grid_spec.dz, self.grid_spec.dt),
            "angles": list(angles_deg),
        }
        # include small hashes of input arrays to detect content changes
        try:
            meta["vp_hash"] = hashlib.sha1(self.vp.array.tobytes()).hexdigest()
            meta["vs_hash"] = hashlib.sha1(self.vs.array.tobytes()).hexdigest()
            meta["rho_hash"] = hashlib.sha1(self.rho.array.tobytes()).hexdigest()
        except Exception:
            pass
        disk_key = self.disk_cache.make_key("ei_multi", meta)
        loaded = self.disk_cache.load_npz(disk_key)
        if loaded is not None:
            # loaded is a dict; cache and return
            self._ei_multi_cache[key] = loaded
            if len(self._ei_multi_cache) > int(self.ei_multi_cache_max):
                self._ei_multi_cache.popitem(last=False)
            return loaded
    if self.vp is None or self.vs is None or self.rho is None:
        raise ValueError("vp, vs and rho are required for EI computation")
    vp_arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
    vs_arr = self.vs.array if isinstance(self.vs, Quantity) else self.vs
    rho_arr = self.rho.array if isinstance(self.rho, Quantity) else self.rho
    out = modeling_utils.modeling_engine.compute_ei_multiangle(
        vp_arr, vs_arr, rho_arr, angles_deg, show_progress=show_progress
    )
    self._ei_multi_cache[key] = out
    if len(self._ei_multi_cache) > int(self.ei_multi_cache_max):
        self._ei_multi_cache.popitem(last=False)
    # Save to disk cache asynchronously (best-effort)
    if self.disk_cache is not None:
        try:
            meta = {
                "shape": (tuple(self.vp.array.shape) if self.vp is not None else None),
                "grid": (
                    self.grid_spec.shape,
                    self.grid_spec.dz,
                    self.grid_spec.dt,
                ),
                "angles": list(angles_deg),
            }
            meta["vp_hash"] = hashlib.sha1(self.vp.array.tobytes()).hexdigest()
            meta["vs_hash"] = hashlib.sha1(self.vs.array.tobytes()).hexdigest()
            meta["rho_hash"] = hashlib.sha1(self.rho.array.tobytes()).hexdigest()
            disk_key = self.disk_cache.make_key("ei_multi", meta)
            self.disk_cache.save_npz(disk_key, out)
        except Exception:
            pass
    return out


def _impl_compute_ei_weighted_product(
    self: RockPhysicsModel,
    litho_angles=None,
    fluid_angles=None,
    litho_weight=0.7,
    fluid_weight=0.3,
    show_progress=True,
) -> dict:
    key = (
        tuple(litho_angles) if litho_angles is not None else None,
        tuple(fluid_angles) if fluid_angles is not None else None,
        float(litho_weight),
        float(fluid_weight),
    )
    if key in self._ei_weighted_cache:
        val = self._ei_weighted_cache.pop(key)
        self._ei_weighted_cache[key] = val
        return val
    # Try disk cache
    if self.disk_cache is not None:
        try:
            meta = {
                "shape": (tuple(self.vp.array.shape) if self.vp is not None else None),
                "grid": (
                    self.grid_spec.shape,
                    self.grid_spec.dz,
                    self.grid_spec.dt,
                ),
                "litho_angles": (
                    tuple(litho_angles) if litho_angles is not None else None
                ),
                "fluid_angles": (
                    tuple(fluid_angles) if fluid_angles is not None else None
                ),
                "litho_weight": float(litho_weight),
                "fluid_weight": float(fluid_weight),
            }
            meta["vp_hash"] = hashlib.sha1(self.vp.tobytes()).hexdigest()
            meta["vs_hash"] = hashlib.sha1(self.vs.tobytes()).hexdigest()
            meta["rho_hash"] = hashlib.sha1(self.rho.tobytes()).hexdigest()
            disk_key = self.disk_cache.make_key("ei_weighted", meta)
            loaded = self.disk_cache.load_npz(disk_key)
            if loaded is not None:
                self._ei_weighted_cache[key] = loaded
                if len(self._ei_weighted_cache) > int(self.ei_weighted_cache_max):
                    self._ei_weighted_cache.popitem(last=False)
                return loaded
        except Exception:
            pass
    if self.vp is None or self.vs is None or self.rho is None:
        raise ValueError("vp, vs and rho are required for EI computation")
    out = modeling_utils.compute_ei_weighted_product(
        self.vp,
        self.vs,
        self.rho,
        litho_angles=litho_angles,
        fluid_angles=fluid_angles,
        litho_weight=litho_weight,
        fluid_weight=fluid_weight,
        show_progress=show_progress,
    )
    self._ei_weighted_cache[key] = out
    if len(self._ei_weighted_cache) > int(self.ei_weighted_cache_max):
        self._ei_weighted_cache.popitem(last=False)
    if self.disk_cache is not None:
        try:
            meta = {
                "shape": tuple(self.vp.shape) if self.vp is not None else None,
                "grid": (
                    self.grid_spec.shape,
                    self.grid_spec.dz,
                    self.grid_spec.dt,
                ),
                "litho_angles": (
                    tuple(litho_angles) if litho_angles is not None else None
                ),
                "fluid_angles": (
                    tuple(fluid_angles) if fluid_angles is not None else None
                ),
                "litho_weight": float(litho_weight),
                "fluid_weight": float(fluid_weight),
            }
            meta["vp_hash"] = hashlib.sha1(self.vp.array.tobytes()).hexdigest()
            meta["vs_hash"] = hashlib.sha1(self.vs.array.tobytes()).hexdigest()
            meta["rho_hash"] = hashlib.sha1(self.rho.array.tobytes()).hexdigest()
            disk_key = self.disk_cache.make_key("ei_weighted", meta)
            self.disk_cache.save_npz(disk_key, out)
        except Exception:
            pass
    return out
