"""Multi-angle EI helpers.

Provides helpers to compute statistics, create weighted stacks and package results.
"""

# typing not required at runtime in this helper module
import numpy as np
from src.utils.facades import LazyObjectProxy
import time
import os
import hashlib
import logging
from typing import Any, Dict, Optional, Sequence
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


__all__ = []


class MultiAngleEI:
    """Object-oriented facade for multi-angle EI helpers.

    This thin facade delegates to module-level implementations so callers
    may use an OOP-style API while preserving behaviour.
    """

    def compute_angle_statistics_and_correlation(
        self, ei_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        return _impl_compute_angle_statistics_and_correlation(ei_results)

    def assemble_weighted_stack(
        self, ei_volumes: Sequence[ArrayLike], weights, verbose: bool = True
    ) -> np.ndarray:
        return _impl_assemble_weighted_stack(ei_volumes, weights, verbose=verbose)

    def create_weighted_stack_and_report(self, ei_volumes, weights):
        return _impl_create_weighted_stack_and_report(ei_volumes, weights)

    def compute_ei_weights(self, ei_volumes, angles, method="variance"):
        return _impl_compute_ei_weights(ei_volumes, angles, method=method)

    def finalize_weighted_stack(self, ei_volumes, weights):
        return _impl_finalize_weighted_stack(ei_volumes, weights)

    def package_ei_results(self, *args, **kwargs):
        return _impl_package_ei_results(*args, **kwargs)

    def compute_ei_gradient_and_summary(
        self, ei_volumes, angles_array, weights=None, verbose=True
    ):
        return _impl_compute_ei_gradient_and_summary(
            ei_volumes, angles_array, weights=weights, verbose=verbose
        )

    def compare_ei_angles(self, ei_results, facies_depth, slice_inline=75):
        return _impl_compare_ei_angles(
            ei_results, facies_depth, slice_inline=slice_inline
        )

    def analyze_facies_correlation_depth(self, ei_volume, facies):
        return _impl_analyze_facies_correlation_depth(ei_volume, facies)

    def compute_and_save_multiangle_ei(self, *args, **kwargs):
        return _impl_compute_and_save_multiangle_ei(*args, **kwargs)

    def compute_and_save_multiangle_ei_from_vm(self, *args, **kwargs):
        return _impl_compute_and_save_multiangle_ei_from_vm(*args, **kwargs)


# Module-level singleton (lazy proxy)
ei_processor = LazyObjectProxy(lambda: MultiAngleEI())


def get_ei_processor(instance: MultiAngleEI | None = None) -> "MultiAngleEI":
    """Return provided MultiAngleEI or the module-level lazy singleton."""
    return _impl_get_ei_processor(instance)


__all__.append("get_ei_processor")


def _impl_get_ei_processor(instance: MultiAngleEI | None = None) -> MultiAngleEI:
    """Canonical implementation for obtaining a MultiAngleEI instance.

    Returns the provided instance when not None, otherwise returns the
    module-level `ei_processor` lazy proxy. Kept as a single `_impl_*`
    entrypoint to make testing and dependency-injection easier.
    """
    return instance if instance is not None else ei_processor


def _impl_compute_angle_statistics_and_correlation(
    ei_results: Dict[str, Any],
) -> Dict[str, Any]:
    ei_volumes = ei_results["ei_volumes"]
    angles = ei_results["angles"]

    angle_stats = []
    for angle, ei_vol in zip(angles, ei_volumes):
        # Accept Quantity-wrapped EI volumes
        vol_arr = ei_vol.array if hasattr(ei_vol, "array") else np.asarray(ei_vol)
        stats = {
            "angle": angle,
            "mean": float(vol_arr.mean()),
            "std": float(vol_arr.std()),
            "min": float(vol_arr.min()),
            "max": float(vol_arr.max()),
            "dynamic_range": float(vol_arr.max() - vol_arr.min()),
        }
        angle_stats.append(stats)

    n_angles = len(ei_volumes)
    corr_matrix = np.zeros((n_angles, n_angles))
    for i in range(n_angles):
        for j in range(n_angles):
            flat_i = (
                ei_volumes[i].array.flatten()
                if hasattr(ei_volumes[i], "array")
                else ei_volumes[i].flatten()
            )
            flat_j = (
                ei_volumes[j].array.flatten()
                if hasattr(ei_volumes[j], "array")
                else ei_volumes[j].flatten()
            )
            corr_matrix[i, j] = np.corrcoef(flat_i, flat_j)[0, 1]

    best_idx = (
        int(np.argmax([s["dynamic_range"] for s in angle_stats]))
        if angle_stats
        else None
    )
    best_angle = float(angles[best_idx]) if best_idx is not None else None

    return {
        "angle_statistics": angle_stats,
        "best_angle": best_angle,
        "best_angle_idx": best_idx,
        "correlation_matrix": corr_matrix.tolist(),
    }


def _impl_assemble_weighted_stack(
    ei_volumes: Sequence[ArrayLike], weights, verbose: bool = True
) -> np.ndarray:
    first = ei_volumes[0]
    first_arr = first.array if hasattr(first, "array") else np.asarray(first)
    ei_stack = np.zeros_like(first_arr)
    for ei_vol, weight in zip(ei_volumes, weights):
        vol_arr = ei_vol.array if hasattr(ei_vol, "array") else np.asarray(ei_vol)
        ei_stack += weight * vol_arr
    if verbose:
        try:
            logger.info(
                "✓ Optimal stack range: %.2e - %.2e", ei_stack.min(), ei_stack.max()
            )
        except Exception:
            pass
    return ei_stack


def _impl_create_weighted_stack_and_report(ei_volumes, weights):
    ei_stack = _impl_assemble_weighted_stack(ei_volumes, weights)
    logger.info("  Created weighted EI stack with shape: %s", ei_stack.shape)
    return ei_stack, weights


def _impl_compute_ei_weights(ei_volumes, angles, method="variance"):
    import numpy as _np

    if method == "equal":
        weights = _np.ones(len(angles)) / len(angles)
    elif method == "cosine":
        w = _np.cos(_np.deg2rad(angles))
        weights = w / w.sum()
    elif method == "variance":
        variances = _np.array([vol.var() for vol in ei_volumes])
        total = float(variances.sum())
        # If all variances are zero or total is not finite, fall back to equal weights
        if total == 0.0 or not _np.isfinite(total):
            weights = _np.ones(len(angles)) / len(angles)
        else:
            weights = variances / total
    elif method == "gradient":
        grads = []
        for vol in ei_volumes:
            grads.append(_np.abs(_np.gradient(vol, axis=2)).mean())
        grads = _np.array(grads)
        totalg = float(grads.sum())
        if totalg == 0.0 or not _np.isfinite(totalg):
            weights = _np.ones(len(angles)) / len(angles)
        else:
            weights = grads / totalg
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return weights


def _impl_finalize_weighted_stack(ei_volumes, weights):
    return _impl_create_weighted_stack_and_report(ei_volumes, weights)


def _impl_package_ei_results(
    ei_volumes,
    angles_array,
    ei_stack,
    ei_gradient,
    weights=None,
    formula: str = "Connolly (1999)",
    extra_config: Optional[Dict[str, Any]] = None,
):
    if weights is None:
        import numpy as _np

        weights = _np.ones(len(angles_array)) / len(angles_array)

    cfg = {
        "angles_deg": list(angles_array),
        "n_angles": len(angles_array),
        "weights": (weights.tolist() if hasattr(weights, "tolist") else list(weights)),
        "formula": formula,
    }
    if extra_config:
        cfg.update(extra_config)

    # Preserve Quantity wrappers where present; otherwise use raw arrays
    results = {
        "ei_volumes": ei_volumes,
        "angles": angles_array,
        "ei_stack": ei_stack,
        "ei_gradient": ei_gradient,
        "config": cfg,
    }

    return results


def _impl_compute_ei_gradient_and_summary(
    ei_volumes, angles_array, weights=None, verbose=True
):
    if weights is None:
        weights = np.ones(len(angles_array)) / len(angles_array)

    ei_stack = _impl_assemble_weighted_stack(ei_volumes, weights, verbose=verbose)
    ei_near = ei_volumes[0]
    ei_far = ei_volumes[-1]
    ei_gradient = ei_far - ei_near

    if verbose:
        try:
            logger.info(
                "✓ EI gradient range: %.2e - %.2e", ei_gradient.min(), ei_gradient.max()
            )
        except Exception:
            pass

    results = _impl_package_ei_results(
        ei_volumes, angles_array, ei_stack, ei_gradient, weights=weights
    )
    return {"ei_stack": ei_stack, "ei_gradient": ei_gradient, "results": results}


def _impl_compare_ei_angles(ei_results, facies_depth, slice_inline=75):
    stats = _impl_compute_angle_statistics_and_correlation(ei_results)
    return stats


def _impl_analyze_facies_correlation_depth(ei_volume, facies):
    from scipy import stats as scipy_stats

    grad_ei = np.abs(np.gradient(ei_volume, axis=2))

    grad_facies = np.abs(np.gradient(facies.astype(float), axis=2))
    facies_boundaries = grad_facies > 0.1

    grad_at_boundaries = grad_ei[facies_boundaries]
    grad_away = grad_ei[~facies_boundaries]

    mean_boundary = grad_at_boundaries.mean()
    mean_away = grad_away.mean()
    std_pooled = np.sqrt((grad_at_boundaries.var() + grad_away.var()) / 2)
    cohens_d = (mean_boundary - mean_away) / (std_pooled + 1e-10)

    flat_grad = grad_ei.flatten()
    flat_boundaries = facies_boundaries.flatten().astype(float)
    r_pearson = np.corrcoef(flat_grad, flat_boundaries)[0, 1]

    r_spearman = scipy_stats.spearmanr(flat_grad, flat_boundaries)[0]

    boundary_amp_mean = grad_at_boundaries.mean()
    away_amp_mean = grad_away.mean()

    snr = boundary_amp_mean / (away_amp_mean + 1e-10)

    return {
        "cohens_d": cohens_d,
        "pearson_r": r_pearson,
        "spearman_r": r_spearman,
        "snr": snr,
    }


def _impl_compute_and_save_multiangle_ei(
    props_depth,
    angles_deg: Optional[Sequence[int]] = None,
    ei_pc3_fluid=None,
    optimization="cohens_d",
    top_n_angles=4,
    cache_dir=".cache",
    show_progress=True,
    compute_ei_multiangle=None,
    create_optimal_ei_stack=None,
    analyze_facies_correlation_depth=None,
):
    if (
        compute_ei_multiangle is None
        or create_optimal_ei_stack is None
        or analyze_facies_correlation_depth is None
    ):
        raise ValueError(
            "Please pass compute_ei_multiangle, create_optimal_ei_stack "
            "and analyze_facies_correlation_depth functions"
        )

    from src.utils import formatting as formatting_utils

    # switched to CacheManager usage

    formatting_utils.print_header("MULTI-ANGLE EI ANALYSIS (DEPTH DOMAIN)")
    if angles_deg is None:
        angles_deg = [0, 5, 10, 15, 20, 25]
    logger.info("Computing EI at %d angles: %s°", len(angles_deg), angles_deg)

    t0 = time.time()
    ei_results = compute_ei_multiangle(
        props_depth["vp"],
        props_depth["vs"],
        props_depth["rho"],
        angles_deg,
        show_progress=show_progress,
    )
    t1 = time.time()
    logger.info("✓ Computed %d EI volumes in %.2fs", len(angles_deg), t1 - t0)

    logger.info("%s", "\nCreating angle stacks...")
    ei_volumes_list = ei_results["ei_volumes"]
    angles_array = ei_results["angles"]
    ei_dict = {
        int(angle): ei_vol for angle, ei_vol in zip(angles_array, ei_volumes_list)
    }

    near_angles = [a for a in angles_deg if a <= 10]
    mid_angles = [a for a in angles_deg if 10 < a <= 20]
    far_angles = [a for a in angles_deg if a > 20]

    ei_near = (
        np.mean([ei_dict[a] for a in near_angles], axis=0) if near_angles else None
    )
    _ = np.mean([ei_dict[a] for a in mid_angles], axis=0) if mid_angles else None
    ei_far = np.mean([ei_dict[a] for a in far_angles], axis=0) if far_angles else None

    facies = props_depth.get("facies")  # Define facies from props_depth
    ei_optimal, weights, selected_angles = create_optimal_ei_stack(
        ei_results, optimization=optimization, facies=facies, top_n_angles=top_n_angles
    )
    formatting_utils.print_selected_angles(selected_angles, weights)

    if ei_far is not None and ei_near is not None:
        ei_gradient = ei_far - ei_near
    else:
        ei_gradient = None

    logger.info("✓ Created angle stacks")

    logger.info("\nAnalyzing facies correlation for each angle...")

    angle_stats_result = _impl_compute_angle_statistics_and_correlation(ei_results)
    angle_statistics = angle_stats_result.get("angle_statistics", [])
    _ = angle_stats_result.get("correlation_matrix")

    correlations = {}
    for s in angle_statistics:
        angle = s["angle"]
        ei_vol = ei_dict.get(angle)
        stats_dict = analyze_facies_correlation_depth(ei_vol, facies)
        correlations[angle] = stats_dict
        try:
            logger.info(
                "  %3d°: Cohen's d = %.4f, Pearson r = %.4f, SNR = %.2f",
                int(angle),
                stats_dict["cohens_d"],
                stats_dict["pearson_r"],
                stats_dict["snr"],
            )
        except Exception:
            logger.info("  %s°: stats available", angle)

    results = _impl_package_ei_results(
        ei_volumes=ei_volumes_list,
        angles_array=angles_array,
        ei_stack=ei_optimal,
        ei_gradient=ei_gradient,
        weights=weights,
        extra_config={
            "vp_range": [
                float(props_depth["vp"].min()),
                float(props_depth["vp"].max()),
            ],
            "vs_range": [
                float(props_depth["vs"].min()),
                float(props_depth["vs"].max()),
            ],
            "rho_range": [
                float(props_depth["rho"].min()),
                float(props_depth["rho"].max()),
            ],
        },
    )

    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    fn = os.path.join(cache_dir, f"multiangle_ei_{key}.npz")
    from src.io.cache import cache_for_dir

    cache_for_dir(cache_dir).save_npz(fn, results)
    logger.info("  Saved multi-angle EI results to: %s", fn)

    return {
        "ei_dict": ei_dict,
        "ei_optimal": ei_optimal,
        "correlations": correlations,
        "best_angle": angle_stats_result.get("best_angle"),
        "cache_file": fn,
    }


def _impl_compute_and_save_multiangle_ei_from_vm(
    vm, vs, rho, angles_deg: Optional[Sequence[int]] = None, **kwargs
) -> Dict[str, Any]:
    """Convenience wrapper: accept a VelocityModel + vs/rho arrays and
    delegate to compute_and_save_multiangle_ei using a props_depth dict.

    This keeps the core logic unchanged while allowing callers to pass a
    GridSpec-aware VelocityModel.
    """
    if angles_deg is None:
        angles_deg = [0, 5, 10, 15, 20, 25]

    props_depth = {
        "vp": vm.vp,
        "vs": vs,
        "rho": rho,
        "facies": kwargs.pop("facies", None),
    }
    return _impl_compute_and_save_multiangle_ei(
        props_depth, angles_deg=angles_deg, **kwargs
    )


# Prefer the OO facade and the module-level lazy proxy. Callers should use
# `ei_processor` or obtain a configured instance via `get_ei_processor()`.
__all__ = ["MultiAngleEI", "ei_processor", "get_ei_processor"]
