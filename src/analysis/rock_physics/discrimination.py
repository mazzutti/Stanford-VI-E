"""Attribute discrimination analysis for facies classification.

This module provides statistical analysis to quantify how well rock physics
attributes separate different facies classes using measures like Cohen's d,
Pearson correlation, and signal-to-noise ratio.
"""

from __future__ import annotations

import logging
import warnings
from typing import TypedDict, Dict

import numpy as np

from src.analysis.processors.types import FloatingArray, IntegerArray

logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10
SNR_EPSILON = 1e-10


class DiscriminationResult(TypedDict):
    """Typed result for attribute discrimination analysis.

    Contains statistical measures quantifying how well an attribute
    separates different facies classes.
    """

    name: str
    cohens_d: float
    pearson_r: float
    p_value: float
    snr: float
    mean_class0: float
    mean_class1: float
    std_class0: float
    std_class1: float


class AttributeDiscriminationAnalyzer:
    """Analyzes discrimination power of attributes versus facies.

    Computes statistical measures to quantify how well an attribute
    separates different facies classes.
    """

    def analyze_single(
        self,
        attribute: FloatingArray,
        facies: IntegerArray,
        name: str = "Attribute",
    ) -> DiscriminationResult:
        """Compute discrimination statistics of an attribute versus facies.

        Computes Cohen's d, Pearson correlation, p-value, SNR, and basic
        statistics to evaluate how well an attribute separates facies classes.
        Results are robust to NaN values and work with multi-class facies.

        Args:
            attribute: Array of attribute values (any shape, will be flattened)
            facies: Array of facies labels (must match attribute shape)
            name: Descriptive name for the attribute

        Returns:
            DiscriminationResult containing:
                - name: attribute name
                - cohens_d: Cohen's d effect size between two most common classes
                - pearson_r: Pearson correlation coefficient
                - p_value: Statistical significance (p-value)
                - snr: Signal-to-noise ratio
                - mean_class0/1, std_class0/1: Statistics for each class

        Note:
            If multiple classes exist, Cohen's d is computed between the two
            most frequent classes. Returns all zeros if inputs contain no valid data.
        """
        from scipy.stats import pearsonr

        attr = np.asarray(attribute).flatten()
        fac = np.asarray(facies).flatten()

        # Validate input compatibility
        if attr.size != fac.size:
            logger.warning(
                f"Attribute size ({attr.size}) != facies size ({fac.size}). "
                f"Will use only valid paired data."
            )
            # Truncate to shorter length to allow pairing
            min_size = min(attr.size, fac.size)
            attr = attr[:min_size]
            fac = fac[:min_size]

        mask = np.isfinite(attr) & np.isfinite(fac)
        if not mask.any():  # More Pythonic than mask.sum() == 0
            return self._empty_result(name)

        attr_valid = attr[mask]
        fac_valid = fac[mask]

        # Select two classes for Cohen's d computation
        class0, class1 = self._select_comparison_classes(fac_valid)

        a0 = attr_valid[fac_valid == class0]
        a1 = attr_valid[fac_valid == class1]

        # Ensure arrays are not empty using atleast_1d instead of conditionals
        a0 = self._ensure_valid_array(a0)
        a1 = self._ensure_valid_array(a1)

        # Extract statistics for both classes
        mean0, std0, mean1, std1 = self._compute_class_stats(a0, a1)

        # Compute Cohen's d effect size
        pooled_std = np.sqrt((std0**2 + std1**2) / 2.0) + EPSILON
        cohens_d = abs(mean1 - mean0) / pooled_std if pooled_std > 0 else 0.0

        # Pearson correlation (attribute vs facies numeric encoding)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message=".*constant.*"
                )
                if attr_valid.size > 1:
                    pr, pv = pearsonr(attr_valid, fac_valid)
                    pearson_r: float = float(pr)
                    p_value: float = float(pv)
                else:
                    pearson_r = 0.0
                    p_value = 1.0
        except Exception as e:
            logger.debug(f"Pearson correlation failed for {name}: {e}")
            pearson_r = 0.0
            p_value = 1.0

        # Compute signal-to-noise ratio
        signal = abs(mean1 - mean0)
        noise = (std0 + std1) / 2.0
        snr = float(signal / noise) if noise > SNR_EPSILON else 0.0

        return {
            "name": name,
            "cohens_d": float(cohens_d),
            "pearson_r": float(pearson_r),
            "p_value": float(p_value),
            "snr": snr,
            "mean_class0": mean0,
            "mean_class1": mean1,
            "std_class0": std0,
            "std_class1": std1,
        }

    @staticmethod
    def _ensure_valid_array(
        arr: FloatingArray,
    ) -> FloatingArray:
        """Ensure array is non-empty, returning [0.0] if empty.

        Args:
            arr: Input array that may be empty

        Returns:
            np.atleast_1d(arr) if arr.size > 0, otherwise np.array([0.0])
        """
        return np.atleast_1d(arr) if arr.size > 0 else np.array([0.0])

    @staticmethod
    def _compute_class_stats(
        class0_data: FloatingArray, class1_data: FloatingArray
    ) -> tuple[float, float, float, float]:
        """Compute mean and standard deviation for two classes.

        Args:
            class0_data: Attribute values for class 0
            class1_data: Attribute values for class 1

        Returns:
            Tuple of (mean0, std0, mean1, std1) as floats
        """
        return (
            float(class0_data.mean()),
            float(class0_data.std()),
            float(class1_data.mean()),
            float(class1_data.std()),
        )

    def analyze_multiple(
        self,
        attribute_results: Dict[str, FloatingArray],
        facies: IntegerArray,
    ) -> Dict[str, DiscriminationResult]:
        """Analyze discrimination power for multiple attributes.

        Runs analyze_single for each attribute in the dictionary, collecting
        statistics in a single mapping. Gracefully handles errors for individual
        attributes by returning empty results.

        Args:
            attribute_results: Dict mapping attribute names to arrays
            facies: Facies labels (must be compatible with attribute shapes)

        Returns:
            Dict mapping attribute names to their discrimination statistics.
            Failed attributes receive empty/default statistics.

        Example:
            attrs = {'intercept': arr1, 'gradient': arr2}
            stats = analyzer.analyze_multiple(attrs, facies)
            # stats['intercept'] = {'cohens_d': 0.8, 'snr': 2.1, ...}
        """
        summary: Dict[str, DiscriminationResult] = {}
        for name, arr in attribute_results.items():
            try:
                stats = self.analyze_single(arr, facies, name=name)
            except Exception:
                logger.exception("Error analyzing attribute %s", name)
                stats = self._empty_result(name)
            summary[name] = stats
        return summary

    @staticmethod
    def _empty_result(name: str) -> DiscriminationResult:
        """Return empty/default result dictionary when analysis fails.

        Used when attribute analysis cannot be performed due to invalid or
        insufficient data. All statistical measures default to 0.0 or neutral values.

        Args:
            name: Attribute name for the result entry

        Returns:
            Dictionary with all statistical fields set to default/zero values
        """
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

    @staticmethod
    def _select_comparison_classes(fac_valid: IntegerArray) -> tuple[int, int]:
        """Select two classes for Cohen's d comparison.

        Returns the two most common classes if available, otherwise defaults
        to the first class or (0, 1) if no classes are present.

        Args:
            fac_valid: 1D array of valid facies values

        Returns:
            Tuple of (class0, class1) for comparison
        """
        unique, counts = np.unique(fac_valid, return_counts=True)
        if unique.size >= 2:
            # pick the two most common classes
            idx_sorted = np.argsort(counts)[::-1]
            return int(unique[idx_sorted[0]]), int(unique[idx_sorted[1]])
        elif unique.size == 1:
            return int(unique[0]), int(unique[0])
        else:
            return 0, 1
