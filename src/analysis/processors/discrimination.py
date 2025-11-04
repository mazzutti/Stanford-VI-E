"""Facies discrimination calculator processor."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from src.analysis.models import (
    FaciesDiscriminationResult,
    FaciesStats,
)

from .base import BaseProcessor
from .config import ProcessorConfig
from .decorators import ProcessorDecorators
from .utils import ProcessorUtils

logger = logging.getLogger(__name__)

__all__ = ["FaciesDiscriminationCalculator"]


class FaciesDiscriminationCalculator(BaseProcessor):
    """Calculates facies discrimination capability using amplitude statistics."""

    def __init__(self) -> None:
        """Initialize the calculator."""
        super().__init__()

    def __repr__(self) -> str:
        """Return string representation of FaciesDiscriminationCalculator instance.

        Returns
        -------
        str
            Representation including aligner reference.
        """
        return f"{self.__class__.__name__}(aligner={self._aligner!r})"

    @ProcessorDecorators.log_debug("Calculating facies discrimination capability...")
    def calculate(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> FaciesDiscriminationResult:
        """Measure how well seismic amplitudes discriminate between facies.

        Computes per-facies amplitude statistics and a separation matrix showing
        standardized differences between facies. High separation values indicate
        good discrimination capability.

        Returns
        -------
        FaciesDiscriminationResult
            Facies statistics, separation matrix, and amplitude distributions.

        Examples
        --------
        >>> seismic = np.random.randn(10, 10, 20)
        >>> facies = np.random.randint(0, 3, (10, 10, 20))
        >>> calc = FaciesDiscriminationCalculator()
        >>> result = calc.calculate(seismic, facies)
        >>> print(f"Analyzed {len(result.facies_stats)} facies")
        >>> print(f"Separation matrix shape: {result.separation_matrix.shape}")
        """
        seismic_aligned, facies_aligned = self._aligner.align(seismic_cube, facies_cube)

        # Extract amplitudes per facies
        facies_amplitudes, label_order = self._extract_facies_amplitudes(
            seismic_aligned, facies_aligned
        )

        # Calculate statistics per facies
        facies_stats = self._calculate_facies_stats(facies_amplitudes)

        # Calculate separation matrix
        separation_matrix = self._calculate_separation_matrix(facies_stats, label_order)

        return FaciesDiscriminationResult(
            facies_stats=facies_stats,
            separation_matrix=separation_matrix,
            facies_amplitudes=facies_amplitudes,
            label_order=label_order,
        )

    @staticmethod
    def _extract_facies_amplitudes(
        seismic_aligned: NDArray[np.float64], facies_aligned: NDArray[np.int64]
    ) -> Tuple[Dict[int, NDArray[np.float64]], List[int]]:
        """Extract amplitude arrays for each observed facies.

        Groups seismic amplitudes by their corresponding facies labels,
        creating an efficiency-optimized dictionary for downstream statistics.

        Parameters
        ----------
        seismic_aligned
            Aligned seismic amplitude cube.
        facies_aligned
            Aligned facies label cube.

        Returns
        -------
        tuple
            (facies_amplitudes dict, sorted label_order list)

        Notes
        -----
        Returns only facies that have at least one sample in the seismic cube.
        Label order is sorted in ascending numerical order for consistency.
        """
        facies_amplitudes: Dict[int, NDArray[np.float64]] = {}
        # Unique returns already sorted unique values; cast once at creation
        label_order: List[int] = np.unique(facies_aligned).astype(int).tolist()

        logger.debug("Extracting amplitudes for %d facies labels", len(label_order))

        for facies_val in label_order:
            mask = facies_aligned == facies_val
            if np.any(mask):
                amp_count = np.count_nonzero(mask)
                facies_amplitudes[facies_val] = seismic_aligned[mask]
                logger.debug(
                    "Facies %d: extracted %d amplitude samples",
                    facies_val,
                    amp_count,
                )

        return facies_amplitudes, label_order

    @staticmethod
    def _calculate_facies_stats(
        facies_amplitudes: Dict[int, NDArray[np.float64]],
    ) -> Dict[int, FaciesStats]:
        """Calculate statistics for each facies.

        Parameters
        ----------
        facies_amplitudes
            Dictionary mapping facies labels to amplitude arrays.

        Returns
        -------
        dict
            Dictionary mapping facies labels to FaciesStats instances.
        """
        facies_stats: Dict[int, FaciesStats] = {}

        for facies_val, amps in facies_amplitudes.items():
            assert isinstance(
                facies_val, (int, np.integer)
            ), f"Expected int facies value, got {type(facies_val)}"
            stats = ProcessorUtils.compute_amplitude_stats(amps)
            if stats is not None:
                facies_stats[facies_val] = stats

        return facies_stats

    @staticmethod
    def _calculate_separation_matrix(
        facies_stats: Dict[int, FaciesStats], label_order: List[int]
    ) -> NDArray[np.float64]:
        """Calculate separation matrix between facies.

        The separation between two facies is computed as the standardized
        difference in means:

            separation = |mean_i - mean_j| / pooled_std

        where pooled_std = sqrt((std_i^2 + std_j^2) / 2)

        This method uses NumPy broadcasting for O(n²) vectorized computation
        instead of nested loops, providing significant speedup for large
        facies counts.

        Parameters
        ----------
        facies_stats
            Dictionary mapping facies labels to statistics.
        label_order
            Ordered list of facies labels.

        Returns
        -------
        numpy.ndarray
            Separation matrix indexed by label_order. Diagonal is zero,
            off-diagonal [i,j] is the standardized separation from facies i to j.
        """
        n_fac = len(label_order)
        separation_matrix = np.zeros((n_fac, n_fac), dtype=float)

        # Filter facies_stats to only include labels in label_order
        available_facies = [lab for lab in label_order if lab in facies_stats]
        if len(available_facies) < 2:
            logger.debug("Insufficient facies with stats for separation matrix")
            return separation_matrix

        # Pre-compute index mapping for efficient lookup (O(n) vs O(n²) for repeated lookups)
        label_to_idx = {lab: idx for idx, lab in enumerate(label_order)}

        # Extract means and stds as arrays for vectorization
        means = np.array(
            [facies_stats[lab].mean for lab in available_facies], dtype=float
        )
        stds = np.array(
            [facies_stats[lab].std for lab in available_facies], dtype=float
        )

        # Vectorized computation using broadcasting
        # Shape: (n, 1) - (1, n) -> (n, n)
        mean_diffs = np.abs(means[:, np.newaxis] - means[np.newaxis, :])
        pooled_stds = np.sqrt((stds[:, np.newaxis] ** 2 + stds[np.newaxis, :] ** 2) / 2)

        # Compute separation with epsilon for numerical stability
        separations = mean_diffs / (pooled_stds + 1e-10)

        # Map results back to the full matrix using pre-computed index mapping
        row_indices = np.array([label_to_idx[lab] for lab in available_facies])
        col_indices = row_indices  # Same indexing for both axes
        separation_matrix[np.ix_(row_indices, col_indices)] = separations

        logger.debug(
            "Computed separation matrix for %d facies using vectorized operations",
            len(available_facies),
        )

        return separation_matrix
