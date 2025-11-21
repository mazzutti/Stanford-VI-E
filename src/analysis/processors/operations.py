"""Common operations for processor implementations.

This module consolidates frequently-used patterns shared across multiple
processor implementations:
- Cube alignment and reshaping
- Data extraction (by mask, by label, at transitions)
- Statistics aggregation
- Separation matrix computation

By factoring out shared logic, we reduce code duplication and improve
maintainability across amplitude, gradient, interface, and discrimination
processors.

Pattern: Operations/Utilities Pattern
- Stateless operation functions (can be called directly or via inheritance)
- Focused, single-responsibility methods
- Used across multiple processors
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.analysis.models import FaciesStats
from src.analysis.processors.management import compute_amplitude_stats

logger = logging.getLogger(__name__)

__all__ = [
    "AlignmentOps",
    "ReshapeOps",
    "ExtractionOps",
    "StatsOps",
]

# pyright: reportUnnecessaryCast=false


class AlignmentOps:
    """Stateless operations for cube alignment.

    Provides common alignment patterns used across processors.
    Designed to work with CubeAligner instances from BaseProcessor.
    """

    # Stateless helper class used as a namespace for related functions.
    # It intentionally exposes only a small public surface.

    @staticmethod
    def align_cubes(
        aligner: Any,
        seismic_cube: NDArray[Any],
        facies_cube: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Align seismic and facies cubes to common shape.

        Parameters
        ----------
        aligner
            CubeAligner instance (from BaseProcessor._aligner).
        seismic_cube
            3D seismic amplitude cube.
        facies_cube
            3D facies label cube.

        Returns
        -------
        tuple
            (seismic_aligned, facies_aligned) with matching shapes.
        """
        # aligner.align is dynamically-typed; cast to the declared return
        # type so static checkers don't treat this as `Any`.
        return cast(
            tuple[NDArray[Any], NDArray[Any]], aligner.align(seismic_cube, facies_cube)
        )


class ReshapeOps:
    """Stateless operations for array reshaping."""

    # Stateless helper class used as a namespace for related functions.
    # It intentionally exposes only a small public surface.

    @staticmethod
    def reshape_to_traces(
        seismic_aligned: NDArray[Any],
        facies_aligned: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Reshape 3D cubes to 2D trace-sample format (n_traces, nk).

        Consolidates the repeated (ni, nj, nk) → (ni*nj, nk) reshape
        pattern used in trace-based analysis.

        Parameters
        ----------
        seismic_aligned
            3D seismic array with shape (ni, nj, nk).
        facies_aligned
            3D facies array with shape (ni, nj, nk).

        Returns
        -------
        tuple
            (seismic_2d, facies_2d) both with shape (n_traces, nk).

        Raises
        ------
        ValueError
            If shapes don't match or reshape fails.
        """
        if seismic_aligned.shape != facies_aligned.shape:
            raise ValueError(
                f"Shape mismatch: seismic {seismic_aligned.shape} "
                f"vs facies {facies_aligned.shape}"
            )

        ni, nj, nk = facies_aligned.shape
        n_traces = ni * nj

        try:
            seismic_2d = seismic_aligned.reshape(n_traces, nk)
            facies_2d = facies_aligned.reshape(n_traces, nk).astype(int, copy=False)
        except ValueError as exc:
            raise ValueError(
                f"Failed to reshape to (n_traces={n_traces}, nk={nk}): {exc}"
            ) from exc

        logger.debug("Reshaped cubes from (ni=%s, nj=%s, nk=%s) to traces", ni, nj, nk)
        # Ensure the return type matches the annotation (NDArray[Any])
        return cast(tuple[NDArray[Any], NDArray[Any]], (seismic_2d, facies_2d))


class ExtractionOps:
    """Stateless operations for extracting data subsets."""

    @staticmethod
    def extract_by_mask(
        data: NDArray[Any],
        mask: NDArray[Any],
        mask_value: bool = True,
    ) -> NDArray[Any]:
        """Extract data where mask matches specified value.

        Parameters
        ----------
        data
            Data array to extract from.
        mask
            Boolean mask array (same shape as data).
        mask_value
            Value to match in mask (True or False).

        Returns
        -------
        numpy.ndarray(dtype=float64)
            Extracted data where mask == mask_value.
        """
        if mask_value:
            return cast(NDArray[Any], data[mask])
        return cast(NDArray[Any], data[~mask])

    @staticmethod
    def extract_by_labels(
        seismic_flat: NDArray[Any],
        labels_flat: NDArray[Any],
    ) -> tuple[dict[int, NDArray[Any]], list[int]]:
        """Group flattened seismic data by facies labels.

        Creates a dictionary mapping each observed label to its amplitudes,
        maintaining label order for reproducibility.

        Parameters
        ----------
        seismic_flat
            Flattened seismic amplitude array.
        labels_flat
            Flattened label array (same shape as seismic).

        Returns
        -------
        tuple
            (label_amplitudes_dict, label_order) where label_order is sorted
            unique labels for consistent matrix construction.
        """
        label_amplitudes: dict[int, list[float]] = {}

        for label in np.unique(labels_flat):
            mask = labels_flat == label
            label_amplitudes[label] = seismic_flat[mask].tolist()

        # Convert lists to arrays for consistency
        label_amps_dict = {
            label: np.array(amps, dtype=np.float64)
            for label, amps in label_amplitudes.items()
        }

        label_order = sorted(label_amps_dict.keys())

        logger.debug(
            "Extracted amplitudes for %d unique labels: %s",
            len(label_order),
            label_order,
        )

        return label_amps_dict, label_order

    @staticmethod
    def extract_at_transitions(
        seismic_2d: NDArray[Any],
        facies_2d: NDArray[Any],
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Extract amplitudes and transition information at facies boundaries.

        Finds vertical transitions (where facies changes between adjacent samples)
        and extracts amplitudes at each transition.

        Parameters
        ----------
        seismic_2d
            2D seismic array (n_traces, nk).
        facies_2d
            2D facies array (n_traces, nk).

        Returns
        -------
        tuple
            (from_facies, to_facies, amplitudes) at each transition.
            All same length, indexed by transition location.
        """
        # Find vertical transitions
        diffs = facies_2d[:, 1:] != facies_2d[:, :-1]
        rows, ks = np.nonzero(diffs)

        if rows.size == 0:
            logger.debug("No transitions found")
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

        # Transition is at k (second sample)
        ks = ks + 1

        # Get from/to facies
        from_facies = facies_2d[rows, ks - 1]
        to_facies = facies_2d[rows, ks]

        # Extract amplitudes
        amplitudes = seismic_2d[rows, ks]

        logger.debug("Found %d transitions", len(rows))

        return from_facies, to_facies, amplitudes


class StatsOps:
    """Stateless operations for statistics aggregation."""

    @staticmethod
    def aggregate_stats(
        label_amplitudes: dict[int, NDArray[np.float64]],
    ) -> dict[int, FaciesStats]:
        """Compute statistics for each facies group.

        Consolidates the repeated pattern of computing statistics per facies
        from the aggregated amplitudes dictionary.

        Parameters
        ----------
        label_amplitudes
            Dictionary mapping facies labels to amplitude arrays.

        Returns
        -------
        dict
            Dictionary mapping facies labels to FaciesStats objects.
        """
        facies_stats: dict[int, FaciesStats] = {}

        for label, amplitudes in label_amplitudes.items():
            stats = compute_amplitude_stats(amplitudes)
            facies_stats[label] = stats
            logger.debug(
                "Facies %s: mean=%.4f, std=%.4f, count=%d",
                label,
                stats.mean,
                stats.std,
                stats.count,
            )

        return facies_stats

    @staticmethod
    def compute_separation_matrix(
        facies_stats: dict[int, FaciesStats],
        label_order: list[int],
        epsilon: float = 1e-10,
    ) -> NDArray[np.float64]:
        """Compute pairwise separation matrix between facies.

        Creates a square matrix where element (i,j) is the standardized
        difference in mean amplitudes between facies i and j.

        The separation metric is: |mean_i - mean_j| / (std_i + std_j + epsilon)

        Parameters
        ----------
        facies_stats
            Dictionary of FaciesStats by facies label.
        label_order
            Sorted list of facies labels for consistent indexing.
        epsilon
            Small epsilon for numerical stability (default 1e-10).

        Returns
        -------
        numpy.ndarray(dtype=float64)
            Square matrix of shape (n_facies, n_facies) with separation values.
            Diagonal is zero (facies separated from itself).
            Symmetric for mean-based separation.
        """
        n = len(label_order)
        matrix = np.zeros((n, n), dtype=np.float64)

        # Filter facies_stats to only include labels in label_order
        available_facies = [lab for lab in label_order if lab in facies_stats]
        if len(available_facies) < 2:
            logger.debug("Insufficient facies with stats for separation matrix")
            return matrix

        for i, label_i in enumerate(label_order):
            for j, label_j in enumerate(label_order):
                if i == j:
                    matrix[i, j] = 0.0
                elif label_i in facies_stats and label_j in facies_stats:
                    stats_i = facies_stats[label_i]
                    stats_j = facies_stats[label_j]

                    mean_diff = abs(stats_i.mean - stats_j.mean)
                    std_sum = stats_i.std + stats_j.std + epsilon

                    matrix[i, j] = mean_diff / std_sum

        logger.debug(
            "Computed separation matrix (%dx%d): max_separation=%.4f",
            n,
            n,
            np.max(matrix),
        )

        return matrix
