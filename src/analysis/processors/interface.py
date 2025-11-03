"""Interface reflection analyzer processor."""

import logging
from typing import Literal, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.analysis.models import (
    FaciesStats,
    InterfaceReflectionResult,
    Transition,
)

from .base import BaseProcessor
from .config import ProcessorConfig
from .decorators import ProcessorDecorators
from .utils import ProcessorUtils

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
        seismic_aligned, facies_aligned = self._aligner.align(seismic_cube, facies_cube)

        if seismic_aligned.shape != facies_aligned.shape:
            raise ValueError(
                f"Aligned cubes have mismatched shapes: "
                f"seismic {seismic_aligned.shape} vs facies {facies_aligned.shape}"
            )

        seismic_2d, facies_2d = self._reshape_to_traces(seismic_aligned, facies_aligned)

        # Find transitions (vertical facies changes) across all traces
        diffs = facies_2d[:, 1:] != facies_2d[:, :-1]
        rows, ks = np.nonzero(diffs)

        if rows.size == 0:
            return InterfaceReflectionResult(transitions_summary={}, interface_stats={})

        ks = ks + 1  # transition index is k (second sample)

        # Extract amplitudes at transitions
        amps = self._extract_amplitudes(seismic_2d, rows, ks)

        # Determine transition pairs (from -> to facies)
        fac_from = facies_2d[rows, ks - 1]
        fac_to = facies_2d[rows, ks]

        # Aggregate by transition type
        return self._aggregate_by_transition(fac_from, fac_to, amps)

    @staticmethod
    def _reshape_to_traces(
        seismic_aligned: NDArray[np.float64], facies_aligned: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
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
            (seismic_2d, facies_2d) both with shape (n_traces, nk).

        Raises
        ------
        ValueError
            If reshape fails due to shape mismatch.
        """
        return ProcessorUtils.reshape_3d_to_2d(seismic_aligned, facies_aligned)

    @staticmethod
    def _extract_amplitudes(
        seismic_2d: NDArray[np.float64],
        rows: NDArray[np.intp],
        ks: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Extract windowed amplitudes around transition points.

        Efficiently extracts mean absolute amplitudes in windows around
        specified transition sample indices. Uses sliding_window_view when
        available for optimal memory efficiency, with a faster NumPy-based
        fallback for compatibility.

        Parameters
        ----------
        seismic_2d
            2D seismic array (n_traces, nk).
        rows
            Trace indices.
        ks
            Sample indices (transition locations).

        Returns
        -------
        numpy.ndarray
            Mean absolute amplitudes in windows around transitions.

        Raises
        ------
        ValueError
            If no valid transition points are provided.
        """
        if rows.size == 0 or ks.size == 0:
            raise ValueError("No transition points provided")

        pad_width = (
            (0, 0),
            (
                ProcessorConfig.AMPLITUDE_WINDOW_RADIUS,
                ProcessorConfig.AMPLITUDE_WINDOW_RADIUS,
            ),
        )
        seismic_padded = np.pad(
            seismic_2d,
            pad_width=pad_width,
            mode=cast(
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                ProcessorConfig.PAD_MODE,
            ),
        )

        try:
            from numpy.lib.stride_tricks import sliding_window_view

            windows = sliding_window_view(
                seismic_padded,
                window_shape=ProcessorConfig.AMPLITUDE_WINDOW_SIZE,
                axis=1,
            )
            logger.debug("Using optimized sliding_window_view for amplitude extraction")
        except (ImportError, ValueError) as e:
            logger.debug(
                "Falling back to efficient manual window construction: %s", str(e)
            )
            # Fallback: construct windows using strided-view manual implementation
            # More memory efficient than explicit loop construction
            n_traces, padded_nk = seismic_padded.shape
            n_windows = padded_nk - ProcessorConfig.AMPLITUDE_WINDOW_SIZE + 1
            windows = np.lib.stride_tricks.as_strided(
                seismic_padded,
                shape=(n_traces, n_windows, ProcessorConfig.AMPLITUDE_WINDOW_SIZE),
                strides=(
                    seismic_padded.strides[0],
                    seismic_padded.strides[1],
                    seismic_padded.strides[1],
                ),
                writeable=False,
            )

        amps = np.abs(windows[rows, ks, :]).mean(axis=1)
        logger.debug("Extracted %d windowed amplitudes", len(amps))
        return cast(NDArray[np.float64], amps)

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

            interface_stats[key] = np.asarray(group_amps)
            summary[key] = ProcessorUtils.compute_amplitude_stats(group_amps)

        return InterfaceReflectionResult(
            transitions_summary=summary,
            interface_stats=cast(
                dict[Transition, Optional[NDArray[np.float64]]], interface_stats
            ),
        )
