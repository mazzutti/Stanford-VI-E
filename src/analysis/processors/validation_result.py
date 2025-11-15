from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ValidationResult:
    """Result of array validation operations.

    Kept minimal and independent so it can be used across the package
    without introducing import cycles.
    """

    is_valid: bool
    arr1: NDArray[np.float64] | None = None
    arr2: NDArray[np.float64] | None = None
    n_removed: int = 0
    error_message: str = ""
