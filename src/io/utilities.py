"""Utility functions for data loading and property management."""

from typing import Any, Dict, Optional
from numpy.typing import NDArray
import numpy as np

from src.io.loader import DatasetManager


def load_depth_properties(
    dm: DatasetManager,
) -> Dict[str, Optional[NDArray[np.floating[Any]]]]:
    """Load depth-domain properties from a DatasetManager.

    Extracts the standard set of properties (vp, vs, rho, facies, full_stack)
    from a DatasetManager instance and returns them as a dictionary.

    Args:
        dm: Loaded DatasetManager instance

    Returns:
        Dictionary with keys: vp, vs, rho, facies, full_stack
        Values are numpy arrays or None if not available
    """
    return {
        "vp": dm.vp,
        "vs": dm.vs,
        "rho": dm.rho,
        "facies": dm.facies,
        "full_stack": dm.full_stack,
    }
