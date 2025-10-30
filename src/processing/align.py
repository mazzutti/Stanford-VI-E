import numpy as np
import logging
from typing import Tuple
from numpy.typing import ArrayLike

__all__ = ["align_cubes"]

# lightweight module logger
logger = logging.getLogger(__name__)


def align_cubes(cube_a: ArrayLike, cube_b: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Trim two cubes to the same minimum shape along each axis.

    Returns the trimmed (cube_a_trimmed, cube_b_trimmed).
    """
    a = np.asarray(cube_a)
    b = np.asarray(cube_b)
    shape_a = a.shape
    shape_b = b.shape
    shape = tuple(min(a, b) for a, b in zip(shape_a, shape_b))
    slices = tuple(slice(0, s) for s in shape)
    return a[slices], b[slices]


# Simple OOP facade
class Aligner:
    def align_cubes(
        self, cube_a: ArrayLike, cube_b: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        return align_cubes(cube_a, cube_b)


from src.utils.facades import LazyObjectProxy


# Module-level lazy proxy using shared LazyObjectProxy
aligner = LazyObjectProxy(lambda: Aligner())
__all__.extend(["Aligner", "aligner"])


def get_aligner(config: dict | None = None):
    """Return the module-level `aligner` proxy when `config` is None,
    otherwise return a new `Aligner` instance.
    """
    if config is None:
        return aligner
    return Aligner()


__all__.append("get_aligner")
