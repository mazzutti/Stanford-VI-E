"""Utility functions for neural network modeling.

This module provides common utility functions used across modeling components,
including device resolution, coordinate grid generation, and seed setting.
"""

import random

import numpy as np
import torch


def get_mgrid(height: int, width: int) -> torch.Tensor:
    """Create a flattened meshgrid of normalized coordinates in [-1, 1].

    This is the module-level variant extracted from `NeuralSmoother.get_mgrid`.
    Keeping a top-level function makes it easier for other modules to import
    and reuse the grid logic without constructing a `NeuralSmoother`.
    """
    tensors = (
        torch.linspace(-1, 1, steps=height),
        torch.linspace(-1, 1, steps=width),
    )
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, 2)


def set_seed(seed: int = 42) -> None:
    """Set seeds for torch, numpy and python.random at module level.

    Keeping a module-level `set_seed` makes it easy for other modules to
    call it without constructing a `NeuralSmoother` instance. The
    `NeuralSmoother.set_seed` method remains as a thin wrapper that
    delegates to this function to preserve backward compatibility.
    """
    seed_int = int(seed)
    torch.manual_seed(seed_int)  # pyright: ignore
    np.random.seed(seed_int)
    random.seed(seed_int)


def resolve_device() -> torch.device:
    """Return the preferred device: MPS (Apple), CUDA, or CPU.

    Exposed at module level so other modules can determine the best device
    without constructing a `NeuralSmoother` instance.
    """
    if (
        getattr(torch.backends, "mps", None) is not None
        and getattr(torch.backends.mps, "is_available", lambda: False)()
        and getattr(torch.backends.mps, "is_built", lambda: True)()
    ):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
