"""Gradient correlation calculator processor."""

import logging

import numpy as np
from scipy.stats import pearsonr, spearmanr
from numpy.typing import NDArray

from src.analysis.models import GradientCorrelationResult

from .base import BaseProcessor
from .boundary import BoundaryDetector
from .config import ProcessorConfig
from .decorators import ProcessorDecorators
from .types import CorrelationFunction, CorrelationResult
from .utils import ProcessorUtils

logger = logging.getLogger(__name__)

__all__ = ["GradientCorrelationCalculator"]


class GradientCorrelationCalculator(BaseProcessor):
    """Calculates correlation between vertical gradient and facies boundaries.

    This processor computes how well seismic amplitude gradients align with
    facies boundaries, using both Pearson (parametric) and Spearman (rank-based)
    correlation methods. High correlations suggest that seismic reflectivity
    patterns are driven by facies changes in the subsurface.

    Notes
    -----
    The calculation pipeline:
    1. Aligns seismic and facies cubes to a common shape
    2. Computes vertical gradients (dseismic/dz) along the k-axis
    3. Detects 4-connected facies boundaries
    4. Correlates absolute gradient with boundary mask (1 on boundary, 0 otherwise)

    Both correlation methods are computed for robustness:
    - Pearson: Assumes linear relationship between gradient and boundaries
    - Spearman: Rank-based, resistant to outliers and non-linear associations
    """

    def __init__(self) -> None:
        """Initialize the calculator."""
        super().__init__()
        self._detector = BoundaryDetector()

    def __repr__(self) -> str:
        """Return string representation of GradientCorrelationCalculator instance.

        Returns
        -------
        str
            Representation including detector and aligner references.
        """
        return f"{self.__class__.__name__}(detector={self._detector!r}, aligner={self._aligner!r})"

    @ProcessorDecorators.time_operation(
        "gradient correlation calculation",
        threshold_ms=ProcessorConfig().gradient_correlation_threshold_ms,
    )
    @ProcessorDecorators.log_debug("Calculating gradient correlation...")
    def calculate(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> GradientCorrelationResult:
        """Compute correlation between absolute vertical gradient and boundaries.

        Quantifies how well seismic amplitude gradients align with facies
        boundaries using both parametric (Pearson) and rank-based (Spearman)
        correlation methods. High correlations suggest that seismic reflectivity
        patterns are driven by subsurface facies changes.

        Follows a multi-step pipeline: align cubes → compute vertical gradients →
        detect boundaries → correlate using both methods.

        Parameters
        ----------
        seismic_cube : numpy.ndarray(dtype=float64)
            3D seismic amplitude cube with shape (ni, nj, nk).
        facies_cube : numpy.ndarray(dtype=int64)
            3D facies label cube with shape (ni, nj, nk).

        Returns
        -------
        GradientCorrelationResult
            Named tuple containing:
            - pearson_correlation: float, linear correlation coefficient
            - pearson_pvalue: float, p-value for Pearson test
            - spearman_correlation: float, rank correlation coefficient
            - spearman_pvalue: float, p-value for Spearman test
            - seismic_gradient: NDArray, absolute vertical gradient
            - boundaries: NDArray(bool), facies boundary mask

        Notes
        -----
        **Correlation Methods**:
        - **Pearson**: Assumes linear relationship; sensitive to outliers
        - **Spearman**: Rank-based; resistant to outliers and non-linear patterns

        **Performance**: Typically ~50-150ms for a 100x100x100 cube, including
        gradient computation and boundary detection.

        Examples
        --------
        >>> import numpy as np
        >>> calc = GradientCorrelationCalculator()
        >>> seismic = np.random.randn(50, 50, 50)
        >>> facies = np.random.randint(0, 3, (50, 50, 50))
        >>> result = calc.calculate(seismic, facies)
        >>> if result.pearson_pvalue < 0.05:
        ...     print(
        ...         f"Significant gradient-boundary correlation: "
        ...         f"r={result.pearson_correlation:.3f}"
        ...     )
        """
        # Align/crop cubes to the same shape
        seismic_aligned, facies_aligned = self._aligner.align(seismic_cube, facies_cube)

        # Compute derivative along vertical (k) axis using consolidated helper
        seismic_grad_abs = ProcessorUtils.compute_vertical_gradient(seismic_aligned)

        # Detect facies boundaries
        boundaries = self._detector.detect(facies_aligned)

        # Calculate correlation
        pearson_corr, pearson_pval = self._compute_correlation(
            seismic_grad_abs, boundaries, pearsonr
        )
        spearman_corr, spearman_pval = self._compute_correlation(
            seismic_grad_abs, boundaries, spearmanr
        )

        return GradientCorrelationResult(
            pearson_correlation=pearson_corr,
            pearson_pvalue=pearson_pval,
            spearman_correlation=spearman_corr,
            spearman_pvalue=spearman_pval,
            seismic_gradient=seismic_grad_abs,
            boundaries=boundaries,
        )

    @staticmethod
    def _compute_correlation(
        seismic_grad: NDArray[np.float64],
        boundaries: NDArray[np.bool_],
        correlation_fn: CorrelationFunction,
    ) -> CorrelationResult:
        """Safely compute correlation between gradient and boundaries.

        Uses validated and filtered arrays to compute correlation coefficients
        and p-values. Returns (NaN, NaN) on any validation failure.

        Parameters
        ----------
        seismic_grad
            Absolute seismic gradient array.
        boundaries
            Boolean boundary mask.
        correlation_fn
            Correlation function (pearsonr or spearmanr).

        Returns
        -------
        CorrelationResult
            (correlation_coefficient, p_value) or (nan, nan) on failure.
        """
        # Flatten and filter in composite operation
        seismic_grad_valid, boundaries_valid = (
            ProcessorUtils._flatten_and_filter_finite_static(seismic_grad, boundaries)
        )

        if seismic_grad_valid is None or boundaries_valid is None:
            return np.nan, np.nan

        # Check minimum sample size after filtering
        if seismic_grad_valid.size < 2 or boundaries_valid.size < 2:
            logger.warning("Insufficient valid samples for correlation computation")
            return np.nan, np.nan

        try:
            seismic_std = np.std(seismic_grad_valid)
            boundaries_std = np.std(boundaries_valid)

            if np.allclose(seismic_std, 0) or np.allclose(boundaries_std, 0):
                logger.warning("Zero variance detected in correlation computation")
                return np.nan, np.nan

            corr, pval = correlation_fn(seismic_grad_valid, boundaries_valid)
            logger.debug(
                "Computed correlation: %.4f (p=%.4e) using %s",
                corr,
                pval,
                correlation_fn.__name__,
            )
            return float(corr), float(pval)
        except ValueError as e:
            logger.error("Error computing correlation: %s", str(e))
            return np.nan, np.nan
