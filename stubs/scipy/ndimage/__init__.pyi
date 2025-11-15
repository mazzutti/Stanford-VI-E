from typing import Any
from collections.abc import Sequence
from numpy.typing import NDArray, ArrayLike

def sobel(
    input: ArrayLike,
    axis: int = ...,
    output: NDArray[Any] | None = ...,
    mode: str = ...,
    cval: float = ...,
) -> NDArray[Any]: ...
def gaussian_filter(
    input: ArrayLike,
    sigma: float | Sequence[float],
    order: int = 0,
    output: NDArray[Any] | None = ...,
    mode: str = ...,
    cval: float = ...,
    truncate: float = ...,
) -> NDArray[Any]: ...
def binary_dilation(
    input: ArrayLike,
    structure: ArrayLike | None = ...,
    iterations: int = ...,
    output: NDArray[Any] | None = ...,
    border_value: int | float = ...,
    origin: int = ...,
) -> NDArray[Any]: ...
