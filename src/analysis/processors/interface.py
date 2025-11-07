"""Interface reflection analyzer processor."""

import logging
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from src.analysis.models import (
    FaciesStats,
    InterfaceReflectionResult,
    Transition,
)

from src.core import BaseProcessor
from .management import ProcessorConfig
from .decorators import ProcessorDecorators
from .operations import AlignmentOps, ReshapeOps, ExtractionOps, StatsOps
from .management import convert_numpy_scalars_to_float, compute_amplitude_stats

logger = logging.getLogger(__name__)

__all__ = ["InterfaceReflectionAnalyzer"]


class InterfaceReflectionAnalyzer(BaseProcessor):
    """Analyzes reflection amplitudes at facies interfaces."""

    def __init__(self) -> None:
        """Initialize the analyzer."""
        super().__init__()

    def __repr__(self) -> str:
        """Return string representation of InterfaceReflectionAnalyzer instance.

        Returns
        -------
        str
            Representation including aligner reference.
        """
        return f"{self.__class__.__name__}(aligner={self._aligner!r})"

    @ProcessorDecorators.time_operation(
        "interface reflection analysis",
        threshold_ms=ProcessorConfig().interface_reflection_analysis_threshold_ms,
    )
    @ProcessorDecorators.log_debug("Analyzing reflection strength at interfaces...")
    def analyze(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> InterfaceReflectionResult:
        """Aggregate reflection amplitudes at facies interfaces.

        Detects facies transitions (vertical changes in facies labels) and
        computes windowed amplitude statistics around each transition.

        Returns
        -------
        InterfaceReflectionResult
            Summary and raw statistics grouped by transition pairs.
        """
        # Align cubes and reshape to (n_traces, nk) for trace-wise analysis
        seismic_aligned, facies_aligned = AlignmentOps.align_cubes(
            self._aligner, seismic_cube, facies_cube
        )

        if seismic_aligned.shape != facies_aligned.shape:
            raise ValueError(
                f"Aligned cubes have mismatched shapes: "
                f"seismic {seismic_aligned.shape} vs facies {facies_aligned.shape}"
            )

        seismic_2d, facies_2d = ReshapeOps.reshape_to_traces(
            seismic_aligned, facies_aligned
        )

        # Extract amplitudes at transitions
        fac_from, fac_to, amps = ExtractionOps.extract_at_transitions(
            seismic_2d, facies_2d
        )

        if fac_from.size == 0:
            return InterfaceReflectionResult(transitions_summary={}, interface_stats={})

        # Aggregate by transition type
        return InterfaceReflectionAnalyzer._aggregate_by_transition(
            fac_from, fac_to, amps
        )

    @staticmethod
    def _aggregate_by_transition(
        fac_from: NDArray[np.int64],
        fac_to: NDArray[np.int64],
        amps: NDArray[np.float64],
    ) -> InterfaceReflectionResult:
        """Aggregate amplitudes by transition type.

        Uses efficient NumPy aggregation by encoding (from, to) facies pairs
        as unique codes: code = from * base + to, where base = max_label + 1.
        This allows sorting and grouping in linear time.

        Parameters
        ----------
        fac_from
            Source facies labels.
        fac_to
            Target facies labels.
        amps
            Amplitudes at transitions.

        Returns
        -------
        InterfaceReflectionResult
            Aggregated summary and raw data by transition.
        """
        # Encode transition pairs to enable efficient NumPy aggregation
        max_label = max(int(fac_from.max()), int(fac_to.max()))
        base = int(max_label) + 1
        codes = fac_from.astype(np.int64) * base + fac_to.astype(np.int64)

        # Sort by encoded transition codes and corresponding amplitudes
        order = np.argsort(codes)
        sorted_codes = codes[order]
        sorted_amps = np.asarray(amps, dtype=np.float64)[order]

        # Find unique transitions and their grouping indices
        unique_codes, idx_start, counts = np.unique(
            sorted_codes, return_index=True, return_counts=True
        )

        interface_stats: dict[Transition, NDArray[np.float64]] = {}
        summary: dict[Transition, Optional[FaciesStats]] = {}

        for ucode, start, cnt in zip(unique_codes, idx_start, counts):
            end = start + cnt
            group_amps = sorted_amps[start:end]
            # Decode transition pair from code (already integral types)
            f_from = ucode // base
            f_to = ucode % base
            key = Transition(int(f_from), int(f_to))

            interface_stats[key] = group_amps
            summary[key] = compute_amplitude_stats(group_amps)

        return InterfaceReflectionResult(
            transitions_summary=summary,
            interface_stats=cast(
                dict[Transition, Optional[NDArray[np.float64]]], interface_stats
            ),
        )

    @staticmethod
    def _reshape_to_traces(
        seismic_aligned: NDArray[np.float64], facies_aligned: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Reshape 3D cubes to 2D trace-sample format.

        Delegates to consolidated reshaping helper for cleaner, more maintainable code.

        Parameters
        ----------
        seismic_aligned
            3D seismic cube (ni, nj, nk).
        facies_aligned
            3D facies cube (ni, nj, nk).

        Returns
        -------
        tuple
            (seismic_2d, facies_2d) with shapes (n_traces, nk).

        Raises
        ------
        ValueError
            If reshape fails due to shape mismatch.
        """
        return ReshapeOps.reshape_to_traces(seismic_aligned, facies_aligned)

    @staticmethod
    def _extract_amplitudes(
        seismic_2d: NDArray[np.float64],
        rows: NDArray[np.intp],
        ks: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Extract windowed amplitudes around transition points.

        Extracts mean absolute amplitudes in windows around
        specified transition sample indices using NumPy vectorization.

        Parameters
        ----------
        seismic_2d
            2D seismic array (n_traces, nk).
        rows
            Row indices of transition points.
        ks
            Column (sample) indices of transition points.

        Returns
        -------
        numpy.ndarray(dtype=float64)
            Mean absolute amplitudes at transition windows (same length as rows).

        Raises
        ------
        ValueError
            If row/column index arrays have different lengths or are empty.
        """
        if rows.size == 0 or ks.size == 0:
            raise ValueError("No transition points provided")

        if rows.size != ks.size:
            raise ValueError(
                f"rows and ks arrays must have same size, got {rows.size} and {ks.size}"
            )

        # Determine window size (minimum of 2 samples around transition)
        window_size = 2

        # Pad seismic data along the sample axis to handle edge cases
        padded = np.pad(
            seismic_2d, ((0, 0), (window_size - 1, window_size - 1)), mode="edge"
        )

        # Adjust indices for padding
        ks_padded = ks + (window_size - 1)

        # Extract windows using NumPy advanced indexing
        windows = np.array(
            [
                padded[row, k - (window_size - 1) : k + 1]
                for row, k in zip(rows, ks_padded)
            ],
            dtype=np.float64,
        )

        # Compute mean absolute amplitude in each window
        amps = np.abs(windows).mean(axis=1)
        logger.debug("Extracted %d windowed amplitudes", len(amps))
        return cast(NDArray[np.float64], amps)
