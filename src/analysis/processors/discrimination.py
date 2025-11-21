"""Facies discrimination calculator processor."""

import logging

import numpy as np
from numpy.typing import NDArray

from src.analysis.models import FaciesDiscriminationResult, FaciesStats
from src.core.processors import BaseProcessor

from .decorators import ProcessorDecorators
from .operations import AlignmentOps, ExtractionOps, StatsOps

logger = logging.getLogger(__name__)

__all__ = ["FaciesDiscriminationCalculator"]

class FaciesDiscriminationCalculator(BaseProcessor):
    """Calculates facies discrimination capability using amplitude statistics."""

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
        seismic_aligned, facies_aligned = AlignmentOps.align_cubes(
            self._aligner, seismic_cube, facies_cube
        )

        # Extract amplitudes per facies
        facies_amplitudes, label_order = ExtractionOps.extract_by_labels(
            seismic_aligned.flatten(), facies_aligned.flatten()
        )

        # Calculate statistics per facies
        facies_stats = StatsOps.aggregate_stats(facies_amplitudes)

        # Calculate separation matrix
        separation_matrix = StatsOps.compute_separation_matrix(
            facies_stats, label_order
        )

        return FaciesDiscriminationResult(
            facies_stats=facies_stats,
            separation_matrix=separation_matrix,
            facies_amplitudes=facies_amplitudes,
            label_order=label_order,
        )

    @staticmethod
    def _extract_facies_amplitudes(
        seismic_aligned: NDArray[np.float64],
        facies_aligned: NDArray[np.int64],
    ) -> tuple[dict[int, NDArray[np.float64]], list[int]]:
        """Extract amplitudes grouped by facies label.

        Delegates to consolidated extraction operation for cleaner,
        more maintainable code.

        Parameters
        ----------
        seismic_aligned
            3D seismic amplitude cube.
        facies_aligned
            3D facies label cube.

        Returns
        -------
        tuple
            (facies_amplitudes_dict, label_order) where label_order
            is sorted unique labels.
        """
        return ExtractionOps.extract_by_labels(
            seismic_aligned.flatten(), facies_aligned.flatten()
        )

    @staticmethod
    def _calculate_facies_stats(
        facies_amplitudes: dict[int, NDArray[np.float64]],
    ) -> dict[int, FaciesStats]:
        """Calculate statistics for each facies.

        Delegates to consolidated statistics operation for cleaner,
        more maintainable code.

        Parameters
        ----------
        facies_amplitudes
            Dictionary mapping labels to amplitude arrays.

        Returns
        -------
        dict
            Dictionary mapping labels to FaciesStats objects.
        """
        return StatsOps.aggregate_stats(facies_amplitudes)

    @staticmethod
    def _calculate_separation_matrix(
        facies_stats: dict[int, FaciesStats],
        label_order: list[int],
    ) -> NDArray[np.float64]:
        """Calculate pairwise facies separation matrix.

        Delegates to consolidated statistics operation for cleaner,
        more maintainable code.

        Parameters
        ----------
        facies_stats
            Dictionary mapping labels to FaciesStats objects.
        label_order
            Sorted list of unique labels.

        Returns
        -------
        numpy.ndarray
            Pairwise separation matrix (symmetric, diagonal=0).
        """
        return StatsOps.compute_separation_matrix(facies_stats, label_order)
