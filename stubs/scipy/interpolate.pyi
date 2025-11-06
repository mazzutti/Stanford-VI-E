# Stub file for scipy.interpolate
from typing import Callable, Literal, overload
from numpy.typing import NDArray, ArrayLike
from typing import Any

class interp1d:
    """1-D interpolation function.

    Interpolates a 1-D function using fixed data points.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        kind: str | int = "linear",
        bounds_error: bool = True,
        fill_value: float | tuple[float, float] | str = "extrapolate",
        assume_sorted: bool = False,
        axis: int = -1,
    ) -> None: ...
    def __call__(self, x: ArrayLike) -> NDArray[Any]: ...
