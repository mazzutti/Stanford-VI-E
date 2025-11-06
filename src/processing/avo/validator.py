"""AVO validation and analysis utilities."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, cast
import numpy as np
from numpy.typing import ArrayLike
import logging

from src.processing.core.abstracts import Validator
from src.processing.core.constants import (
    DEFAULT_MAX_AVO_ANGLE,
    DEFAULT_CONTRAST_THRESHOLD,
)

__all__ = ["AVOValidator", "AVOValidityReport"]

logger = logging.getLogger(__name__)


@dataclass
class AVOValidityReport:
    """Report on AVO linearization validity.

    Attributes:
        max_angle: Maximum angle checked (degrees)
        contrast_vp: Fractional contrast in Vp
        contrast_vs: Fractional contrast in Vs
        contrast_rho: Fractional contrast in density
        contrast_flag: True if contrasts exceed threshold
        angle_flag: True if max_angle exceeds safe limits
        suggested_angles: List of safer angles to use
    """

    max_angle: float
    contrast_vp: float
    contrast_vs: float
    contrast_rho: float
    contrast_flag: bool
    angle_flag: bool
    suggested_angles: Optional[List[float]] = None

    def is_valid(self, contrast_threshold: float = DEFAULT_CONTRAST_THRESHOLD) -> bool:
        """Check if linearization conditions are acceptable.

        Args:
            contrast_threshold: Maximum acceptable fractional contrast

        Returns:
            True if valid within constraints
        """
        return (
            not self.contrast_flag
            and not self.angle_flag
            and max(self.contrast_vp, self.contrast_vs, self.contrast_rho)
            <= contrast_threshold
        )

    def print_summary(self) -> None:
        """Print a formatted summary of the validity report."""
        logger.info("Aki-Richards Linearization Validity Summary:")
        logger.info("  Max angle checked: %s deg", self.max_angle)
        logger.info("  Vp fractional contrast: %.3f", self.contrast_vp)
        logger.info("  Vs fractional contrast: %.3f", self.contrast_vs)
        logger.info("  Rho fractional contrast: %.3f", self.contrast_rho)

        if self.contrast_flag:
            logger.warning(
                "  ⚠️  Large property contrasts detected; linear approximation may be poor."
            )
        if self.angle_flag:
            logger.warning(
                "  ⚠️  Large maximum angle; AVO linearization accuracy decreases with angle."
            )

        if self.suggested_angles:
            logger.info("  Suggested safer angles: %s", self.suggested_angles)
        else:
            logger.info("  Linearization checks: OK (no immediate issues detected)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "max_angle": self.max_angle,
            "contrast_vp": self.contrast_vp,
            "contrast_vs": self.contrast_vs,
            "contrast_rho": self.contrast_rho,
            "contrast_flag": self.contrast_flag,
            "angle_flag": self.angle_flag,
            "suggested_angles": self.suggested_angles,
        }


class AVOValidator(Validator):
    """Validates AVO linearization assumptions and provides recommendations.

    Checks whether data satisfies the conditions for linear AVO approximation
    (Aki-Richards) and suggests safer angle ranges if needed.
    """

    def __init__(
        self,
        max_angle: float = DEFAULT_MAX_AVO_ANGLE,
        contrast_threshold: float = DEFAULT_CONTRAST_THRESHOLD,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize validator.

        Args:
            max_angle: Maximum angle to check (degrees)
            contrast_threshold: Fractional contrast threshold
            logger: Optional logger instance
        """
        self.max_angle = max_angle
        self.contrast_threshold = contrast_threshold
        self.logger = logger or logging.getLogger(__name__)

    def validate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Validate AVO linearization conditions.

        Args:
            vp: P-wave velocity array
            vs: S-wave velocity array
            rho: Density array

        Returns:
            AVOValidityReport as dictionary with results
        """
        # Extract positional or keyword arguments
        if args:
            if len(args) < 3:
                raise ValueError("validate() requires vp, vs, rho as positional arguments")
            vp = args[0]
            vs = args[1]
            rho = args[2]
        else:
            # Try to get from kwargs
            vp = kwargs.get('vp')
            vs = kwargs.get('vs')
            rho = kwargs.get('rho')
            if vp is None or vs is None or rho is None:
                raise ValueError("validate() requires vp, vs, rho arguments")
        
        vp = np.asarray(vp)
        vs = np.asarray(vs)
        rho = np.asarray(rho)

        # Compute fractional contrasts
        contrast_vp = self._compute_fractional_contrast(vp)
        contrast_vs = self._compute_fractional_contrast(vs)
        contrast_rho = self._compute_fractional_contrast(rho)

        # Check flags
        contrast_flag = any(
            c > self.contrast_threshold
            for c in (contrast_vp, contrast_vs, contrast_rho)
        )
        angle_flag = self.max_angle > 30.0

        # Suggest angles
        suggested_angles = self._suggest_angles(contrast_flag, angle_flag)

        report = AVOValidityReport(
            max_angle=float(self.max_angle),
            contrast_vp=contrast_vp,
            contrast_vs=contrast_vs,
            contrast_rho=contrast_rho,
            contrast_flag=contrast_flag,
            angle_flag=angle_flag,
            suggested_angles=suggested_angles,
        )
        # Return as Dict[str, Any] per interface, but actually return the report object
        # Mypy sees it as a dict due to the cast, but it's actually the report object
        return cast(Dict[str, Any], report)

    def is_valid(self, *args: Any, **kwargs: Any) -> bool:
        """Quick check if conditions are acceptable.

        Args:
            vp: P-wave velocity array
            vs: S-wave velocity array
            rho: Density array

        Returns:
            True if valid
        """
        # Support both positional and keyword arguments
        if args and len(args) >= 3:
            vp, vs, rho = args[0], args[1], args[2]
        else:
            vp = kwargs.get('vp')
            vs = kwargs.get('vs')
            rho = kwargs.get('rho')
            if vp is None or vs is None or rho is None:
                raise ValueError("is_valid() requires vp, vs, rho arguments")
        
        # The validate method returns a cast Dict but is actually an AVOValidityReport
        report = cast(AVOValidityReport, self.validate(vp, vs, rho))
        return report.is_valid(self.contrast_threshold)

    @staticmethod
    def _compute_fractional_contrast(arr: np.ndarray) -> float:
        """Compute fractional contrast of array.

        Args:
            arr: Input array

        Returns:
            Fractional contrast (0 to 1)
        """
        amax = np.nanmax(arr)
        amin = np.nanmin(arr)
        if amax == 0:
            return 0.0
        return float((amax - amin) / max(amax, 1e-12))

    @staticmethod
    def _suggest_angles(contrast_flag: bool, angle_flag: bool) -> Optional[List[float]]:
        """Suggest safer angle ranges based on flags.

        Args:
            contrast_flag: True if contrasts exceed threshold
            angle_flag: True if max_angle exceeds 30°

        Returns:
            List of suggested angles or None
        """
        if contrast_flag and angle_flag:
            return [0, 10, 20]
        elif contrast_flag:
            return [0, 10, 15]
        elif angle_flag:
            return [0, 10, 20]
        return None
