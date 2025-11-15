from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ValidationResult:
    """Result of array validation operations.

    Kept minimal and independent so it can be used across the package
    without introducing import cycles.
    """

    is_valid: bool
    arr1: Optional[NDArray[np.float64]] = None
    arr2: Optional[NDArray[np.float64]] = None
    n_removed: int = 0
    error_message: str = ""
