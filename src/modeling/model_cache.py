"""Model cache wrappers.

Wrappers that focus on caching compute-heavy modeling outputs. They keep
call-time imports for heavy compute helpers to avoid import cycles and
mirror the public caching API used by modeling callers.
"""

from pathlib import Path
import os
import numpy as np
from typing import Any, Dict, Tuple
import logging

from src.io.cache import cache_for_dir
from src.utils.facades import LazyObjectProxy

__all__ = [
    "cached_avo",
    "cached_avo_from_vm",
    # Public API focuses on AVO cache helpers
]

logger = logging.getLogger(__name__)


def _impl_cached_avo(
    props_time: Dict[str, Any],
    angles,
    wavelet,
    cache_dir: str = ".cache",
    use_quality_weighting: bool = False,
    add_noise: bool = False,
    snr_db: int = 20,
    noise_seed: int | None = None,
) -> Tuple[list, Any]:
    """Canonical implementation: compute or load cached AVO synthetics.

    Kept as an _impl_* function so facades and module-level proxies can
    safely delegate without creating recursion.
    """

    # Local imports to avoid import cycles
    from src.modeling.modeling import _hash_for_cache, create_avo_synthetics

    force = os.environ.get("FORCE_RECOMPUTE", "0") == "1"

    vp = props_time["vp"]
    vs = props_time["vs"]
    rho = props_time["rho"]

    extra_params = [
        angles,
        wavelet,
        use_quality_weighting,
        add_noise,
        snr_db,
        noise_seed,
    ]

    key = _hash_for_cache([vp, vs, rho], extras=extra_params)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fn = Path(cache_dir) / f"avo_time_{key}.npz"

    if (not force) and fn.exists():
        data = np.load(fn)
        full_stack = data["full_stack"]
        angle_stacks = None
        if "angle_0" in data:
            angle_stacks = [data[f"angle_{i}"] for i in range(len(angles))]
        return angle_stacks, full_stack

    angle_stacks, full_stack = create_avo_synthetics(
        props_time,
        angles,
        wavelet,
        use_quality_weighting=use_quality_weighting,
        add_noise=add_noise,
        snr_db=snr_db,
        noise_seed=noise_seed,
    )

    save_dict: Dict[str, Any] = {"full_stack": full_stack}
    for i, angle_stack in enumerate(angle_stacks):
        save_dict[f"angle_{i}"] = angle_stack

    cache_for_dir(cache_dir).save_npz(fn, save_dict)
    return angle_stacks, full_stack


def _impl_cached_avo_from_vm(
    vm, vs, rho, angles, wavelet, cache_dir: str = ".cache", **kwargs
):
    """Convenience implementation: build props_time from a VelocityModel and
    delegate to the canonical cached_avo implementation.
    """
    props_time = {"vp": vm.vp, "vs": vs, "rho": rho}
    return _impl_cached_avo(props_time, angles, wavelet, cache_dir=cache_dir, **kwargs)


# Object-oriented facade for modeling cache helpers
class ModelingCache:
    def __init__(self):
        # placeholder for cache configuration in future
        pass

    def cached_avo(self, *args, **kwargs):
        return _impl_cached_avo(*args, **kwargs)

    def cached_avo_from_vm(self, *args, **kwargs):
        return _impl_cached_avo_from_vm(*args, **kwargs)


# Module-level lazy proxy singleton for ModelingCache
modeling_cache: ModelingCache = LazyObjectProxy(lambda: ModelingCache())


def cached_avo_from_vm(
    vm, vs, rho, angles, wavelet, cache_dir: str = ".cache", **kwargs
):
    """Convenience wrapper accepting a VelocityModel and forwarding to cached_avo.

    vm: VelocityModel (contains vp and grid_spec)
    vs, rho: numpy arrays matching vm.vp shape
    """
    props_time = {"vp": vm.vp, "vs": vs, "rho": rho}
    return _impl_cached_avo(props_time, angles, wavelet, cache_dir=cache_dir, **kwargs)


__all__.extend(
    [
        "ModelingCache",
        "modeling_cache",
        "cached_avo",
        "cached_avo_from_vm",
    ]
)


def get_modeling_cache(cache: ModelingCache | None = None) -> "ModelingCache":
    """Return the provided ModelingCache or the module-level lazy singleton.

    This makes it easy to inject a test double or a configured cache instance.
    """
    return _impl_get_modeling_cache(cache)


def _impl_get_modeling_cache(cache: ModelingCache | None = None) -> "ModelingCache":
    return cache if cache is not None else modeling_cache


__all__.append("get_modeling_cache")


# Depth cache helpers for other per-technique caches are not part of the current API


def cached_avo(
    props_time: Dict[str, Any],
    angles,
    wavelet,
    cache_dir: str = ".cache",
    use_quality_weighting: bool = False,
    add_noise: bool = False,
    snr_db: int = 20,
    noise_seed: int | None = None,
) -> Tuple[list, Any]:
    return modeling_cache.cached_avo(
        props_time,
        angles,
        wavelet,
        cache_dir=cache_dir,
        use_quality_weighting=use_quality_weighting,
        add_noise=add_noise,
        snr_db=snr_db,
        noise_seed=noise_seed,
    )
