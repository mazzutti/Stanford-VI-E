import numpy as np
import logging
from typing import Tuple
from numpy.typing import ArrayLike
from src.processing._singleton import SingletonFactory

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


# Module-level lazy factory for aligner
_aligner_factory: SingletonFactory[Aligner] = SingletonFactory(lambda: Aligner())


def get_aligner(aligner_inst: Aligner | None = None) -> Aligner:
    """Return the module-level aligner singleton or a custom instance."""
    return _aligner_factory.get(aligner_inst)


__all__.extend(["Aligner", "get_aligner"])
