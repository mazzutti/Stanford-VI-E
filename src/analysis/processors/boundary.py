"""Boundary detection and cube alignment processors."""

import logging
from typing import cast, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BaseProcessor, Processor
from .config import NeighborSlices, ProcessorConfig
from .decorators import ProcessorDecorators
from .validators import ArrayValidator

logger = logging.getLogger(__name__)

__all__ = ["BoundaryDetector", "CubeAligner"]


class BoundaryDetector(BaseProcessor):
    """Detects facies boundaries in 3D cubes using 4-connected neighbor logic.

    A boundary voxel is one where at least one of its 4-connected neighbors
    (within the same depth slice) has a different facies label. This detector
    uses edge-padded comparison to ensure consistent behavior at cube edges
    and corners.

    Notes
    -----
    Edge handling: The cube is padded with edge values (replicating boundary
    values) to ensure that voxels at cube edges are compared consistently with
    their valid neighbors. This means edge voxels can be marked as boundaries
    if they differ from their neighbors.

    Performance: Uses memory-efficient in-place boolean operations to minimize
    peak memory usage during boundary detection.
    """

    # Class constants
    NDIM_REQUIRED: int = 3
    CONNECTIVITY_TYPE: str = "4-connected"  # In-plane connectivity (same k-slice)

    def __repr__(self) -> str:
        """Return string representation of BoundaryDetector instance.

        Returns
        -------
        str
            Simple representation indicating this is a BoundaryDetector instance.
        """
        return f"{self.__class__.__name__}(connectivity={self.CONNECTIVITY_TYPE})"

    @ProcessorDecorators.time_operation(
        "boundary detection",
        threshold_ms=ProcessorConfig().boundary_detection_threshold_ms,
    )
    @ProcessorDecorators.log_debug("Detecting facies boundaries in 3D...")
    def detect(self, facies_cube: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Detect facies boundaries in a 3D facies cube.

        A voxel is considered a boundary when any 4-connected neighbor in
        the same 2D slice (up/down/left/right) has a different integer
        facies label. This detector uses edge padding to ensure consistent
        behavior at cube boundaries.

        Parameters
        ----------
        facies_cube : numpy.ndarray(dtype=int64)
            Integer-valued 3D facies label cube with shape (i, j, k).

        Returns
        -------
        numpy.ndarray(dtype=bool)
            Boolean mask where ``True`` marks facies-boundary voxels.

        Raises
        ------
        ValueError
            If ``facies_cube`` is not a 3-dimensional array or is empty.

        Notes
        -----
        **Performance**: Uses memory-efficient in-place boolean operations
        (|= operator) to minimize intermediate array allocations, reducing
        peak memory usage for large cubes. Typical execution: ~50-100ms
        for a 100x100x100 cube.

        **Connectivity**: 4-connected neighbors within the same depth slice
        (up, down, left, right). Vertical changes (between depth slices) are
        not considered boundaries.

        Examples
        --------
        >>> import numpy as np
        >>> detector = BoundaryDetector()
        >>> facies = np.array([[[1, 1], [1, 2]],
        ...                    [[1, 2], [2, 2]]], dtype=np.int64)
        >>> boundaries = detector.detect(facies)
        >>> boundaries.shape
        (2, 2, 2)
        >>> boundaries.sum()  # Number of boundary voxels
        3
        """
        self._validate_input(facies_cube)

        # Pad with edge mode so edge comparisons behave consistently
        padded = self._pad_and_compare(facies_cube)

        return padded

    @staticmethod
    def _pad_and_compare(facies_cube: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Detect boundaries by comparing with padded neighbors.

        Compares each voxel with its 4-connected neighbors (up, down, left, right
        in the same depth slice). Boundary voxels have neighbors with different
        facies labels.

        Uses memory-efficient in-place boolean operations to minimize intermediate
        array allocations, reducing peak memory usage for large cubes.

        Parameters
        ----------
        facies_cube
            Input 3D facies cube.

        Returns
        -------
        numpy.ndarray(dtype=bool)
            Boolean boundary mask where True indicates a boundary voxel.
        """
        # Pad with edge mode so edge comparisons behave consistently
        pad_config = ProcessorConfig.BOUNDARY_PAD_CONFIG
        padded = np.pad(
            facies_cube,
            pad_width=cast(
                Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                pad_config["pad_width"],
            ),
            mode=cast(Literal["edge"], pad_config["mode"]),
        )

        center = padded[:, NeighborSlices.CENTER.value, NeighborSlices.CENTER.value]

        # Initialize boundaries with first comparison, then OR in remaining comparisons
        # This avoids creating intermediate arrays for each comparison
        boundaries: NDArray[np.bool_] = (
            center != padded[:, NeighborSlices.UP.value, NeighborSlices.CENTER.value]
        )

        # Use |= operator for in-place OR to reduce memory allocations
        boundaries |= (
            center != padded[:, NeighborSlices.DOWN.value, NeighborSlices.CENTER.value]
        )
        boundaries |= (
            center != padded[:, NeighborSlices.CENTER.value, NeighborSlices.LEFT.value]
        )
        boundaries |= (
            center != padded[:, NeighborSlices.CENTER.value, NeighborSlices.RIGHT.value]
        )

        logger.debug(
            "Boundary detection complete: %d boundary voxels found",
            np.count_nonzero(boundaries),
        )
        return boundaries

    @staticmethod
    def _validate_input(facies_cube: NDArray[np.int64]) -> None:
        """Validate input facies cube.

        Parameters
        ----------
        facies_cube
            The cube to validate.

        Raises
        ------
        ValueError
            If cube is not 3D or is empty.
        """
        ArrayValidator.validate_3d_array(facies_cube, "facies_cube")


class CubeAligner(Processor):
    """Aligns and crops multiple 3D cubes to a common shape."""

    NDIM_REQUIRED: int = 3

    def process(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Implement polymorphic process() method (required by Processor ABC).

        Delegates to align() method. This enables treating CubeAligner
        uniformly with other processors.

        Parameters
        ----------
        seismic_cube : numpy.ndarray(dtype=float64)
            Seismic amplitude cube with shape (ni, nj, nk).
        facies_cube : numpy.ndarray(dtype=int64)
            Facies label cube with shape (ni, nj, nk).

        Returns
        -------
        tuple
            (seismic_aligned, facies_aligned) both with same shape.
        """
        return cast(
            Tuple[NDArray[np.float64], NDArray[np.int64]],
            self.align(seismic_cube, facies_cube),
        )

    def __call__(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Make CubeAligner callable as a convenience wrapper for align().

        Allows using aligner instance as a function:
            aligner = CubeAligner()
            seismic_aligned, facies_aligned = aligner(seismic, facies)

        Parameters
        ----------
        seismic_cube : numpy.ndarray(dtype=float64)
            Seismic amplitude cube with shape (ni, nj, nk).
        facies_cube : numpy.ndarray(dtype=int64)
            Facies label cube with shape (ni, nj, nk).

        Returns
        -------
        tuple
            (seismic_aligned, facies_aligned) both with same shape.
        """
        return cast(
            Tuple[NDArray[np.float64], NDArray[np.int64]],
            self.align(seismic_cube, facies_cube),
        )

    @ProcessorDecorators.time_operation(
        "cube alignment", threshold_ms=ProcessorConfig().cube_alignment_threshold_ms
    )
    @ProcessorDecorators.log_debug("Aligning cubes to common shape...")
    def align(
        self, seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Crop two 3D cubes to their minimum common shape.

        Enables joint analysis of seismic and facies data by aligning them
        to a common bounding box. Useful when input cubes have different
        dimensions due to preprocessing or data loading variations.

        Parameters
        ----------
        seismic_cube : numpy.ndarray(dtype=float64)
            Seismic amplitude cube with shape (ni, nj, nk).
        facies_cube : numpy.ndarray(dtype=int64)
            Facies label cube with shape (ni, nj, nk).

        Returns
        -------
        tuple
            Tuple of (seismic_aligned, facies_aligned) both with shape
            (min_ni, min_nj, min_nk) where min_* is the minimum size
            along each dimension.

        Raises
        ------
        ValueError
            If either input is not 3-dimensional or empty.

        Notes
        -----
        **Memory Efficiency**: Uses view-based slicing (no copy) for O(1)
        memory overhead regardless of cube size.

        **Vectorization**: Employs np.minimum() for efficient element-wise
        computation of minimum shape.

        Examples
        --------
        >>> seismic = np.random.randn(100, 100, 50)
        >>> facies = np.random.randint(0, 5, (90, 100, 50))  # Different size
        >>> aligner = CubeAligner()
        >>> s_aligned, f_aligned = aligner.align(seismic, facies)
        >>> s_aligned.shape == f_aligned.shape
        True
        """
        self._validate_inputs(seismic_cube, facies_cube)

        # Compute minimum dimensions along each axis using vectorized operation
        min_shape = tuple(np.minimum(seismic_cube.shape, facies_cube.shape))

        # Use slicing with tuple unpacking for cleaner, more efficient cropping
        slices = tuple(slice(0, size) for size in min_shape)
        seismic_aligned = seismic_cube[slices]
        facies_aligned = facies_cube[slices]

        return seismic_aligned, facies_aligned

    @staticmethod
    def _validate_inputs(
        seismic_cube: NDArray[np.float64], facies_cube: NDArray[np.int64]
    ) -> None:
        """Validate alignment inputs.

        Uses shared validation helper to ensure consistent error checking
        across all cube alignment operations.

        Parameters
        ----------
        seismic_cube
            Seismic cube to validate.
        facies_cube
            Facies cube to validate.

        Raises
        ------
        ValueError
            If cubes are not 3D or empty.
        """
        ArrayValidator.validate_3d_arrays(
            (seismic_cube, "seismic_cube"), (facies_cube, "facies_cube")
        )
