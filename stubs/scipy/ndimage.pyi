from typing import Any
from collections.abc import Sequence
from numpy.typing import NDArray, ArrayLike

# Typed stubs for commonly used ndimage functions. Return types use
# NDArray[Any] conservatively to avoid propagating Unknown, while input
# types accept ArrayLike for flexibility.

def binary_dilation(
    input: ArrayLike,
    structure: ArrayLike | None = ...,
    iterations: int = ...,
    output: NDArray[Any] | None = ...,
    border_value: int | float = ...,
    origin: int = ...,
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
def sobel(
    input: ArrayLike,
    axis: int = ...,
    output: NDArray[Any] | None = ...,
    mode: str = ...,
) -> NDArray[Any]: ...
