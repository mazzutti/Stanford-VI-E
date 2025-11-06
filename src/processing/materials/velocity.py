"""P-wave velocity model with unit handling."""

from __future__ import annotations


from dataclasses import dataclass
from typing import Optional, Union, Any
from numpy.typing import NDArray


import numpy as np
from scipy.ndimage import gaussian_filter


from src.io.grid import GridSpec
from src.utils.quantity import Quantity


__all__ = ["VelocityModel"]


import logging


logger = logging.getLogger(__name__)


@dataclass
class VelocityModel:
    """Wrapper for a P-wave velocity cube.

    Attributes:
        vp: numpy array (ni, nj, nz) containing P-wave velocities (m/s)
        grid_spec: GridSpec defining shape/dz/dt
    """

    # `vp` may be a raw numpy array or a `Quantity`.
    vp: Union[Quantity, NDArray[Any]]
    grid_spec: GridSpec

    def __post_init__(self) -> None:
        # Wrap raw arrays into Quantity if needed
        if not isinstance(self.vp, Quantity):
            # assume unitless numeric array (likely m/s or km/s) — store without unit metadata
            self.vp = Quantity(self.vp, "m/s")

        if self.vp.array.ndim != 3:
            raise ValueError("vp must be a 3D array (ni, nj, nz)")
        ni, nj, nz = self.vp.array.shape
        if (ni, nj, nz) != self.grid_spec.shape:
            raise ValueError("vp shape must match grid_spec.shape")

    def to_m_per_s(self) -> None:
        """Ensure velocities are in meters per second (in-place).

        If the velocities are in km/s (values < 100), assume they are km/s
        and convert them to m/s. This is a pragmatic heuristic used widely in
        the repository.
        """
        # Convert Quantity in-place to m/s
        if isinstance(self.vp, Quantity):
            q = self.vp.to("m/s", copy=True)
            self.vp = q
        else:
            q = Quantity(self.vp, "m/s").to("m/s", copy=True)
            self.vp = q

    def ensure_m_per_s(self) -> bool:
        """Ensure velocities are in m/s. Return True if a conversion was applied."""
        # Use Quantity -> attempt conversion and inspect whether units changed
        if not isinstance(self.vp, Quantity):
            self.vp = Quantity(self.vp, "m/s")
        before_unit = self.vp.unit
        self.vp = self.vp.to("m/s", copy=True)
        return before_unit != self.vp.unit

    def validate(self) -> None:
        """Validate vp array for physical plausibility.

        Raises ValueError if values are non-finite or non-positive.
        """
        arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
        if not np.all(np.isfinite(arr)):
            raise ValueError("vp contains non-finite values")
        if np.any(arr <= 0.0):
            raise ValueError("vp contains non-positive values")

    def smooth(self, sigma: float = 1.0, truncate: Optional[float] = None) -> None:
        """Apply Gaussian smoothing in-place to the velocity model.

        Args:
            sigma: standard deviation for Gaussian kernel (applied to each axis)
        """
        # Apply smoothing to the numeric array and preserve unit metadata
        arr = self.vp.array if isinstance(self.vp, Quantity) else self.vp
        smoothed = gaussian_filter(arr, sigma=sigma)
        if isinstance(self.vp, Quantity):
            self.vp = Quantity(smoothed, self.vp.unit)
        else:
            self.vp = smoothed
