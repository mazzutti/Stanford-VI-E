"""Utility functions for processor operations."""

import logging
from typing import Any, cast, Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.analysis.models import FaciesStats
from src.analysis.strategies import (
    ArrayStatisticsStrategy,
    StandardArrayStatistics,
)


if TYPE_CHECKING:
    from .boundary import CubeAligner

logger = logging.getLogger(__name__)

__all__ = ["ProcessorUtils"]


class ProcessorUtils:
    """Utility class providing common operations for processors.

    Groups frequently-used helper methods for numerical computation
    data filtering, and statistics calculation.

    This is a stateless utility class designed to be used via static methods:
    ProcessorUtils.convert_numpy_scalars_to_float(...)

    Can be configured with different statistics strategies for flexible
    computation approaches (standard vs robust, for example).
    """

    # Default statistics strategy - can be overridden per instance or globally
    _default_strategy: ArrayStatisticsStrategy = StandardArrayStatistics()

    def __init__(self, strategy: Optional[ArrayStatisticsStrategy] = None):
        """Initialize processor utils with optional strategy.

        Parameters
        ----------
        strategy : ArrayStatisticsStrategy, optional
            Statistics strategy to use. Defaults to StandardArrayStatistics.
        """
        self.strategy = strategy or self._default_strategy

    @classmethod
    def set_default_strategy(cls, strategy: ArrayStatisticsStrategy) -> None:
        """Set default statistics strategy for all new instances.

        Parameters
        ----------
        strategy
            New default strategy to use.
        """
        cls._default_strategy = strategy

    @classmethod
    def get_default_strategy(cls) -> ArrayStatisticsStrategy:
        """Get current default statistics strategy.

        Returns
        -------
        ArrayStatisticsStrategy
            Current default strategy.
        """
        return cls._default_strategy

    @staticmethod
    def _convert_numpy_scalars_to_float_static(
        *values: NDArray[np.floating[Any]] | np.floating[Any],
    ) -> Tuple[float, ...] | float:
        """Efficiently convert one or more NumPy scalars/arrays to Python floats.

        Consolidates the common pattern of `float(np.operation())` into a
        single function call, improving code readability and reducing overhead.

        Parameters
        ----------
        *values : numpy.ndarray | numpy.floating
            One or more NumPy scalars or arrays to convert.

        Returns
        -------
        float | tuple of float
            Converted Python float values. Returns single float if one argument
            tuple of floats if multiple arguments.

        Examples
        --------
        >>> mean_val, std_val = ProcessorUtils.convert_numpy_scalars_to_float(
        ...     np.mean(arr), np.std(arr)
        ... )
        >>> single = ProcessorUtils.convert_numpy_scalars_to_float(np.median(arr))
        """
        if len(values) == 1:
            # Extract scalar value if it's a NumPy array/scalar
            val = values[0]
            return float(val.item() if hasattr(val, "item") else val)
        return tuple(float(v.item() if hasattr(v, "item") else v) for v in values)

    @staticmethod
    def _compute_quartiles_static(amps: NDArray[np.float64]) -> Tuple[float, float]:
        """Efficiently compute Q1 and Q3 percentiles from amplitude array.

        Consolidates the repeated pattern of computing first and third quartiles
        and converting them to floats. Uses standard percentiles (Q1=25, Q3=75).

        Parameters
        ----------
        amps : numpy.ndarray(dtype=float64)
            Array of amplitude values.

        Returns
        -------
        tuple of float
            (q1, q3) quartile values as Python floats.

        Examples
        --------
        >>> q1, q3 = ProcessorUtils.compute_quartiles(amplitude_array)
        >>> iqr = q3 - q1
        """
        percentiles = np.percentile(amps, [25, 75])
        result = ProcessorUtils._convert_numpy_scalars_to_float_static(*percentiles)
        q1, q3 = cast(Tuple[float, float], result)
        return q1, q3

    @staticmethod
    def _filter_finite_values_static(
        arr1: NDArray[np.float64], arr2: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Filter out NaN and Inf values from paired arrays.

        Removes elements from both arrays where either value is NaN or Inf.
        Returns filtered arrays and count of removed values.

        Parameters
        ----------
        arr1 : numpy.ndarray(dtype=float64)
            First input array to filter.
        arr2 : numpy.ndarray(dtype=float64)
            Second input array to filter.

        Returns
        -------
        tuple
            (filtered_arr1, filtered_arr2, n_removed) where n_removed is the count
            of elements removed due to non-finite values.

        Examples
        --------
        >>> arr1_filtered, arr2_filtered, n_removed = ProcessorUtils.filter_finite_values(
        ...     gradient, boundaries
        ... )
        """
        valid_mask = np.isfinite(arr1) & np.isfinite(arr2)
        n_removed = (~valid_mask).sum()

        if n_removed > 0:
            logger.debug(
                "Filtered %d non-finite values from arrays (original size: %d)",
                n_removed,
                len(arr1),
            )

        return arr1[valid_mask], arr2[valid_mask], int(n_removed)

    @staticmethod
    def _flatten_and_filter_finite_static(
        arr: NDArray[np.float64], bool_mask: NDArray[np.bool_]
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """Flatten array and boolean mask, then filter finite values.

        Combines flattening and finite-value filtering in a single operation
        for computational efficiency.

        Parameters
        ----------
        arr : numpy.ndarray(dtype=float64)
            Array to flatten and filter.
        bool_mask : numpy.ndarray(dtype=bool)
            Boolean mask to flatten and convert to float.

        Returns
        -------
        tuple
            (flattened_arr, flattened_mask) if sufficient valid samples exist
            else (None, None).

        Examples
        --------
        >>> arr_filtered, mask_filtered = ProcessorUtils.flatten_and_filter_finite(
        ...     seismic_gradient, boundary_mask
        ... )
        """
        arr_flat = arr.flatten()
        mask_flat = bool_mask.flatten().astype(float)

        arr_filtered, mask_filtered, _ = ProcessorUtils._filter_finite_values_static(
            arr_flat, mask_flat
        )
        return arr_filtered, mask_filtered

    @staticmethod
    def reshape_3d_to_2d(
        seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Reshape 3D cubes to 2D trace-sample format (n_traces, nk).

        Consolidates repeated reshape pattern used in trace-based analysis.
        Reshapes (ni, nj, nk) cubes to (ni*nj, nk) for efficient trace processing.

        Parameters
        ----------
        seismic_cube : numpy.ndarray(dtype=float64)
            3D seismic cube with shape (ni, nj, nk).
        facies_cube : numpy.ndarray(dtype=int64)
            3D facies cube with shape (ni, nj, nk).

        Returns
        -------
        tuple
            (seismic_2d, facies_2d) both with shape (n_traces, nk).

        Raises
        ------
        ValueError
            If reshape fails due to shape mismatch or incompatible dimensions.

        Examples
        --------
        >>> seismic_2d, facies_2d = ProcessorUtils.reshape_3d_to_2d(seismic_3d, facies_3d)
        >>> n_traces, nk = seismic_2d.shape
        """
        ni, nj, nk = facies_cube.shape
        n_traces = ni * nj

        try:
            seismic_2d = seismic_cube.reshape(n_traces, nk)
            facies_2d = facies_cube.reshape(n_traces, nk).astype(int, copy=False)
        except ValueError as e:
            raise ValueError(
                f"Failed to reshape cubes to (n_traces={n_traces}, nk={nk}): {e}"
            )

        return seismic_2d, facies_2d

    @staticmethod
    def align_and_reshape(
        aligner: "CubeAligner",
        seismic_cube: NDArray[np.float64],
        facies_cube: NDArray[np.int64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Align 3D cubes and reshape to 2D trace-sample format.

        Consolidates repeated "align then reshape" pattern used across multiple
        processors. This is a common workflow: first align cubes to common shape
        then reshape (ni, nj, nk) → (ni*nj, nk) for trace-based analysis.

        Parameters
        ----------
        aligner : CubeAligner
            Alignment operator initialized with target dimensions.
        seismic_cube : numpy.ndarray(dtype=float64)
            3D seismic amplitude cube.
        facies_cube : numpy.ndarray(dtype=int64)
            3D facies classification cube.

        Returns
        -------
        tuple
            (seismic_2d, facies_2d) with shape (n_traces, nk).

        Raises
        ------
        ValueError
            If alignment or reshape fails.

        Examples
        --------
        >>> aligner = CubeAligner.create_from_target(ni=100, nj=100, nk=500)
        >>> seismic_2d, facies_2d = ProcessorUtils.align_and_reshape(
        ...     aligner, seismic_cube, facies_cube
        ... )
        """
        seismic_aligned = aligner.align(seismic_cube)
        facies_aligned = aligner.align(facies_cube)
        return ProcessorUtils.reshape_3d_to_2d(seismic_aligned, facies_aligned)

    @staticmethod
    def compute_vertical_gradient(
        seismic_cube: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute vertical (k-axis) gradient of seismic cube with absolute values.

        Efficiently computes the vertical derivative along the k-axis and returns
        absolute values for correlation/amplitude analysis. This is a common expensive
        operation that benefits from clear documentation and reuse.

        Parameters
        ----------
        seismic_cube : numpy.ndarray(dtype=float64)
            3D seismic amplitude cube with shape (i, j, k).

        Returns
        -------
        numpy.ndarray(dtype=float64)
            Absolute vertical gradient array with same shape as input.

        Notes
        -----
        Uses numpy.gradient() with axis=2 (vertical k-axis) for stable
        finite-difference approximation. Returns absolute values automatically.

        Examples
        --------
        >>> seismic_grad_abs = ProcessorUtils.compute_vertical_gradient(seismic_cube)
        >>> print(f"Gradient range: [{seismic_grad_abs.min():.4e}, {seismic_grad_abs.max():.4e}]")
        """
        seismic_grad = np.gradient(seismic_cube, axis=2)
        seismic_grad_abs = np.abs(seismic_grad)
        logger.debug(
            "Computed vertical gradient: range=[%.4e, %.4e]",
            np.min(seismic_grad_abs),
            np.max(seismic_grad_abs),
        )
        return cast(NDArray[np.float64], seismic_grad_abs)

    @staticmethod
    def extract_amplitude_subset(
        data: NDArray[np.float64], mask: NDArray[np.bool_], mask_value: bool = True
    ) -> NDArray[np.float64]:
        """Extract amplitude values where mask has specified value.

        Consolidates the repeated pattern of using boolean masks to extract
        amplitude subsets (e.g., amplitudes at boundaries vs away from boundaries).

        Parameters
        ----------
        data : numpy.ndarray(dtype=float64)
            Amplitude data array.
        mask : numpy.ndarray(dtype=bool)
            Boolean mask array (same shape as data).
        mask_value : bool, optional
            Value to match in mask (True or False). Default is True.

        Returns
        -------
        numpy.ndarray(dtype=float64)
            Extracted amplitude values where mask matches mask_value.

        Examples
        --------
        >>> amps_at_boundaries = ProcessorUtils.extract_amplitude_subset(
        ...     seismic, boundaries, mask_value=True
        ... )
        >>> amps_away = ProcessorUtils.extract_amplitude_subset(
        ...     seismic, boundaries, mask_value=False
        ... )
        """
        if mask_value:
            return data[mask]
        else:
            return data[~mask]

    @staticmethod
    def compute_amplitude_stats(amps: NDArray[np.float64]) -> FaciesStats:
        """Compute statistical summary of amplitude array.

        Efficiently computes all standard statistical measures (mean, std, median
        percentiles, min, max) from an amplitude array in a single pass where possible.

        Parameters
        ----------
        amps : numpy.ndarray(dtype=float64)
            Array of amplitude values.

        Returns
        -------
        FaciesStats
            Statistical summary. Empty FaciesStats (all fields zero) if array is empty.

        Notes
        -----
        Converts all NumPy scalars to Python floats efficiently in a single call
        to minimize overhead. Uses consolidated quartile computation helper.

        Examples
        --------
        >>> amps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> stats = ProcessorUtils.compute_amplitude_stats(amps)
        >>> stats.mean
        3.0
        """
        if amps.size == 0:
            logger.debug("Computing stats for empty amplitude array")
            return FaciesStats()

        # Compute all statistics with single conversion call
        result = ProcessorUtils._convert_numpy_scalars_to_float_static(
            np.mean(amps), np.std(amps), np.median(amps), np.min(amps), np.max(amps)
        )
        mean_val, std_val, median_val, min_val, max_val = cast(
            Tuple[float, float, float, float, float], result
        )

        # Compute quartiles using consolidated helper
        q1, q3 = ProcessorUtils._compute_quartiles_static(amps)

        stats = FaciesStats(
            count=len(amps),
            mean=mean_val,
            std=std_val,
            median=median_val,
            q25=q1,
            q75=q3,
            min=min_val,
            max=max_val,
        )

        logger.debug(
            "Computed stats from %d samples: mean=%.4f, std=%.4f",
            stats.count,
            stats.mean,
            stats.std,
        )

        return stats
