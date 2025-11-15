from typing import Any, Sequence
from numpy.typing import NDArray

# Conservative stubs: return NDArray[Any] to avoid Unknown propagation

def binary_dilation(input: Any, structure: Any = ..., iterations: int = ..., mask: Any = ..., output: Any = ..., border_value: int = 0, origin: int = 0) -> NDArray[Any]: ...

def gaussian_filter(input: Any, sigma: float | Sequence[float] = ..., order: int | Sequence[int] = 0, output: Any = ..., mode: str = ..., cval: float = 0.0, truncate: float = 4.0) -> NDArray[Any]: ...

def sobel(input: Any, axis: int = ..., output: Any = ..., mode: str = ...) -> NDArray[Any]: ...
