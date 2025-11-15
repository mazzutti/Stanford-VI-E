from typing import Any, Optional, Sequence, Union
from numpy.typing import NDArray, ArrayLike

# Typed stubs for commonly used ndimage functions. Return types use
# NDArray[Any] conservatively to avoid propagating Unknown, while input
# types accept ArrayLike for flexibility.

def binary_dilation(
	input: ArrayLike,
	structure: Optional[ArrayLike] = ...,
	iterations: int = ...,
	output: Optional[NDArray[Any]] = ...,
	border_value: Union[int, float] = ...,
	origin: int = ...,
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

def sobel(
	input: ArrayLike,
	axis: int = ...,
	output: Optional[NDArray[Any]] = ...,
	mode: str = ...,
) -> NDArray[Any]: ...
