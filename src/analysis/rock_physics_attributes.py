"""AVO-focused rock physics attribute helpers.

This module provides compact, well-tested computational utilities used by
analysis and plotting code. It intentionally contains only numerical
functions (no plotting or cache handling) to keep the surface area
small and avoid cross-module import/undefined-name issues.

Exports:
    - compute_avo_attributes: returns intercept/gradient and derived volumes
    - compute_lambda_mu_rho: returns lambda_rho, mu_rho and ratio
    - compute_fluid_factor: simple fluid factor from lambda/mu
    - analyze_attribute_discrimination: basic discrimination statistics
    - compare_all_attributes: run discrimination over a dict of arrays
"""

from __future__ import annotations

import logging
from typing import Sequence, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


__all__ = [
    "compute_avo_attributes",
    "compute_lambda_mu_rho",
    "compute_fluid_factor",
    "analyze_attribute_discrimination",
    "compare_all_attributes",
]


def compute_avo_attributes(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    angles_deg: Sequence[float] = (0, 5, 10, 15, 20, 25),
) -> Dict[str, np.ndarray]:
    """Compute AVO attributes (intercept, gradient) from rock property cubes.

    Args:
        vp, vs, rho: 3D numpy arrays with identical shapes (ni, nj, nk)
        angles_deg: sequence of incidence angles in degrees used for fitting

    Returns:
        dict with keys: 'intercept', 'gradient', 'product', 'scaled_gradient'
    """
    logger.info("Computing AVO attributes from rock physics...")

    # Local import to avoid heavy dependencies at module import time
    from src.signal.reflectivity import zoeppritz_solver as solver

    angles_rad = np.deg2rad(angles_deg)
    sin2_theta = np.sin(angles_rad) ** 2

    ni, nj, nk = vp.shape
    intercept = np.zeros((ni, nj, nk - 1), dtype=np.float32)
    gradient = np.zeros((ni, nj, nk - 1), dtype=np.float32)

    # Compute reflectivity at each angle and fit R(θ) = A + B*sin²θ per trace
    for k in range(nk - 1):
        vp1, vp2 = vp[:, :, k], vp[:, :, k + 1]
        vs1, vs2 = vs[:, :, k], vs[:, :, k + 1]
        rho1, rho2 = rho[:, :, k], rho[:, :, k + 1]

        # reflectivities: shape (n_angles, ni, nj)
        refl_list = []
        for angle in angles_rad:
            r = solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, angle)
            refl_list.append(r)

        reflectivities = np.array(refl_list)

        # Fit per trace using least squares
        A = np.vstack([np.ones(len(sin2_theta)), sin2_theta]).T
        for i in range(ni):
            for j in range(nj):
                r_trace = reflectivities[:, i, j]
                # guard against constant / invalid traces
                if not np.all(np.isfinite(r_trace)):
                    intercept[i, j, k] = np.nan
                    gradient[i, j, k] = np.nan
                    continue
                coeffs, *_ = np.linalg.lstsq(A, r_trace, rcond=None)
                intercept[i, j, k] = coeffs[0]
                gradient[i, j, k] = coeffs[1]

    product = intercept * gradient
    scaled_gradient = gradient / (np.abs(intercept) + 1e-10)

    return {
        "intercept": intercept,
        "gradient": gradient,
        "product": product,
        "scaled_gradient": scaled_gradient,
    }


def compute_lambda_mu_rho(
    vp: np.ndarray, vs: np.ndarray, rho: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute Lambda-Rho and Mu-Rho attributes.

    Returns a dict with 'lambda_rho', 'mu_rho' and 'lambda_mu_ratio'.
    """
    logger.info("Computing Lambda-Mu-Rho attributes...")

    mu = rho * vs**2
    lambda_mod = rho * vp**2 - 2 * mu

    # convention: lambda_rho = lambda * rho, mu_rho = mu * rho
    lambda_rho = lambda_mod * rho
    mu_rho = mu * rho

    lambda_mu_ratio = lambda_mod / (mu + 1e-10)

    return {
        "lambda_rho": lambda_rho,
        "mu_rho": mu_rho,
        "lambda_mu_ratio": lambda_mu_ratio,
    }


def compute_fluid_factor(
    lambda_rho: np.ndarray, mu_rho: np.ndarray, k: float = 1.0
) -> np.ndarray:
    """Compute a simple fluid factor = lambda_rho - k * mu_rho.

    k can be tuned; default is 1.0 which works for many clastic datasets.
    """
    return lambda_rho - k * mu_rho


def analyze_attribute_discrimination(
    attribute: np.ndarray, facies: np.ndarray, name: str = "Attribute"
) -> Dict[str, Any]:
    """Compute simple discrimination statistics of an attribute vs facies.

    Returns a dict with Cohen's d, Pearson r (absolute), p-value, SNR and basic stats.
    The function is robust to NaNs and works for multi-class facies, but the
    Cohen's d is computed between class 0 and class 1 if present; otherwise
    it's computed between the two most frequent classes.
    """
    from scipy.stats import pearsonr

    attr = np.asarray(attribute).flatten()
    fac = np.asarray(facies).flatten()

    mask = np.isfinite(attr) & np.isfinite(fac)
    if mask.sum() == 0:
        return {
            "name": name,
            "cohens_d": 0.0,
            "pearson_r": 0.0,
            "p_value": 1.0,
            "snr": 0.0,
            "mean_class0": 0.0,
            "mean_class1": 0.0,
            "std_class0": 0.0,
            "std_class1": 0.0,
        }

    attr_valid = attr[mask]
    fac_valid = fac[mask]

    # Decide which two classes to compare for Cohen's d
    unique, counts = np.unique(fac_valid, return_counts=True)
    if unique.size >= 2:
        # pick the two most common classes
        idx_sorted = np.argsort(counts)[::-1]
        class0 = unique[idx_sorted[0]]
        class1 = unique[idx_sorted[1]]
    elif unique.size == 1:
        class0 = unique[0]
        class1 = unique[0]
    else:
        class0, class1 = 0, 1

    a0 = attr_valid[fac_valid == class0]
    a1 = attr_valid[fac_valid == class1]

    if a0.size == 0:
        a0 = np.array([0.0])
    if a1.size == 0:
        a1 = np.array([0.0])

    mean0 = float(a0.mean())
    mean1 = float(a1.mean())
    std0 = float(a0.std())
    std1 = float(a1.std())

    pooled_std = np.sqrt((std0**2 + std1**2) / 2.0) + 1e-10
    cohens_d = abs(mean1 - mean0) / pooled_std if pooled_std > 0 else 0.0

    # Pearson correlation (attribute vs facies numeric encoding)
    try:
        pearson_r, p_value = (
            pearsonr(attr_valid, fac_valid) if attr_valid.size > 1 else (0.0, 1.0)
        )
    except Exception:
        pearson_r, p_value = 0.0, 1.0

    signal = abs(mean1 - mean0)
    noise = (std0 + std1) / 2.0
    snr = float(signal / noise) if noise > 0 else 0.0

    return {
        "name": name,
        "cohens_d": float(cohens_d),
        "pearson_r": float(pearson_r),
        "p_value": float(p_value),
        "snr": float(snr),
        "mean_class0": mean0,
        "mean_class1": mean1,
        "std_class0": std0,
        "std_class1": std1,
    }


def compare_all_attributes(
    attribute_results: Dict[str, np.ndarray], facies: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """Run discrimination analysis for every attribute in attribute_results.

    Returns a mapping attribute_name -> statistics dict (as produced by
    analyze_attribute_discrimination).
    """
    summary = {}
    for name, arr in attribute_results.items():
        try:
            stats = analyze_attribute_discrimination(arr, facies, name=name)
        except Exception:
            logger.exception("Error analyzing attribute %s", name)
            stats = {
                "name": name,
                "cohens_d": 0.0,
                "pearson_r": 0.0,
                "p_value": 1.0,
                "snr": 0.0,
                "mean_class0": 0.0,
                "mean_class1": 0.0,
                "std_class0": 0.0,
                "std_class1": 0.0,
            }
        summary[name] = stats
    return summary
