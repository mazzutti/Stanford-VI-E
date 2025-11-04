"""AVO processing helpers.

Contains AVO linearization checks and reporting helpers.
"""

from typing import Dict, Any
import numpy as np
from numpy.typing import ArrayLike
import logging

__all__ = ["check_linearization_validity", "print_validity_report"]

# module logger
logger = logging.getLogger(__name__)


def check_linearization_validity(
    vp: ArrayLike, vs: ArrayLike, rho: ArrayLike, *, max_angle: float = 30.0
) -> Dict[str, Any]:
    vp = np.asarray(vp)
    vs = np.asarray(vs)
    rho = np.asarray(rho)

    def frac_contrast(arr: np.ndarray) -> float:
        amax = np.nanmax(arr)
        amin = np.nanmin(arr)
        if amax == 0:
            return 0.0
        return float((amax - amin) / max(amax, 1e-12))

    contrast_vp = frac_contrast(vp)
    contrast_vs = frac_contrast(vs)
    contrast_rho = frac_contrast(rho)

    contrast_flag = any(c > 0.20 for c in (contrast_vp, contrast_vs, contrast_rho))
    angle_flag = float(max_angle) > 30.0

    suggested_angles = None
    if contrast_flag and angle_flag:
        suggested_angles = [0, 10, 20]
    elif contrast_flag:
        suggested_angles = [0, 10, 15]
    elif angle_flag:
        suggested_angles = [0, 10, 20]

    return {
        "max_angle": float(max_angle),
        "contrast_vp": contrast_vp,
        "contrast_vs": contrast_vs,
        "contrast_rho": contrast_rho,
        "contrast_flag": bool(contrast_flag),
        "angle_flag": bool(angle_flag),
        "suggested_angles": suggested_angles,
    }


def print_validity_report(report: Dict[str, Any]) -> None:
    if not isinstance(report, dict):
        logger.error("Validity report: <invalid format>")
        return
    logger.info("Aki-Richards Linearization Validity Summary:")
    logger.info("  Max angle checked: %s deg", report.get("max_angle", "N/A"))
    cvp = report.get("contrast_vp")
    cvs = report.get("contrast_vs")
    crho = report.get("contrast_rho")
    if cvp is not None:
        logger.info("  Vp fractional contrast: %.3f", cvp)
    if cvs is not None:
        logger.info("  Vs fractional contrast: %.3f", cvs)
    if crho is not None:
        logger.info("  Rho fractional contrast: %.3f", crho)

    if report.get("contrast_flag"):
        logger.warning(
            "  ⚠️  Large property contrasts detected; linear approximation may be poor."
        )
    if report.get("angle_flag"):
        logger.warning(
            "  ⚠️  Large maximum angle requested; AVO linearization accuracy decreases with angle."
        )

    suggested = report.get("suggested_angles")
    if suggested:
        logger.info("  Suggested safer angles: %s", suggested)
    else:
        logger.info("  Linearization checks: OK (no immediate issues detected)")
