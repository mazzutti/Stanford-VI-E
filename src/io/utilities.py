"""Utility functions for data loading and property management."""

from typing import Any

from numpy.typing import NDArray

from src.io.loader import DatasetManager
from src.utils.quantity import Quantity

def load_depth_properties(
    dm: DatasetManager,
) -> dict[str, NDArray[Any] | Quantity | None]:
    """Load depth-domain properties from a DatasetManager.

    Extracts the standard set of properties (vp, vs, rho, facies, full_stack)
    from a DatasetManager instance and returns them as a dictionary.

    Args:
        dm: Loaded DatasetManager instance

    Returns:
        Dictionary with keys: vp, vs, rho, facies, full_stack
        Values are numpy arrays, `Quantity` wrappers, or None if not available
    """
    return {
        "vp": dm.vp,
        "vs": dm.vs,
        "rho": dm.rho,
        "facies": dm.facies,
        "full_stack": dm.full_stack,
    }
