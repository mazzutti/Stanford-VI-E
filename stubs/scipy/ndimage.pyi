# Minimal stubs for the scipy.ndimage functions used in this project
from typing import Any, Optional, Sequence
import numpy as np

def sobel(
    input: np.ndarray, axis: int = -1, output: Optional[np.ndarray] = ...
) -> np.ndarray: ...
def gaussian_filter(
    input: np.ndarray,
    sigma: float | Sequence[float],
    output: Optional[np.ndarray] = ...,
) -> np.ndarray: ...
def binary_dilation(
    input: np.ndarray,
    iterations: int = 1,
    structure: Any = ...,
    output: Optional[np.ndarray] = ...,
) -> np.ndarray: ...
