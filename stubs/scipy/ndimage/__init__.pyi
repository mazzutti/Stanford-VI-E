from typing import Any, Optional, Sequence, Union
from numpy.typing import NDArray, ArrayLike

def sobel(
    input: ArrayLike,
    axis: int = ...,
    output: Optional[NDArray[Any]] = ...,
    mode: str = ...,
    cval: float = ...,
) -> NDArray[Any]: ...
def gaussian_filter(
    input: ArrayLike,
    sigma: Union[float, Sequence[float]],
    order: int = 0,
    output: Optional[NDArray[Any]] = ...,
    mode: str = ...,
    cval: float = ...,
    truncate: float = ...,
) -> NDArray[Any]: ...
def binary_dilation(
    input: ArrayLike,
    structure: Optional[ArrayLike] = ...,
    iterations: int = ...,
    output: Optional[NDArray[Any]] = ...,
    border_value: int | float = ...,
    origin: int = ...,
) -> NDArray[Any]: ...
