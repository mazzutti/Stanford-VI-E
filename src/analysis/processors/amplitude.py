"""Boundary amplitude extraction processor."""

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation
from numpy.typing import NDArray

from src.analysis.models import BoundaryAmpsResult

from src.core import BaseProcessor
from .management import ProcessorConfig
from .decorators import ProcessorDecorators
from .operations import AlignmentOps, ExtractionOps
from .validators import ArrayValidator

logger = logging.getLogger(__name__)

__all__ = ["BoundaryAmplitudeExtractor"]


class BoundaryAmplitudeExtractor(BaseProcessor):
    """Extracts amplitudes at and away from facies boundaries."""

    def __init__(self, dilation_window: int = 2):
        """Initialize the extractor.

        Parameters
        ----------
        dilation_window
            Default dilation radius for boundary zones. Must be positive.

        Raises
        ------
        ValueError
            If dilation_window is not positive.
        """
        super().__init__()
        ArrayValidator.validate_positive_parameter(dilation_window, "dilation_window")

        self.dilation_window = dilation_window
        logger.debug(
            "Initialized BoundaryAmplitudeExtractor with window=%d", dilation_window
        )

    @ProcessorDecorators.time_operation(
        "boundary amplitude extraction",
        threshold_ms=ProcessorConfig().boundary_amplitude_extraction_threshold_ms,
    )
    @ProcessorDecorators.log_debug("Extracting amplitudes at facies boundaries...")
    def extract(
        self,
        seismic_cube: NDArray[np.float64],
        boundaries: NDArray[np.bool_],
        window: Optional[int] = None,
    ) -> BoundaryAmpsResult:
        """Extract amplitudes at and away from facies boundaries.

        Separates seismic amplitudes into two groups: those within a dilated
        boundary zone and those outside. The dilation expands boundaries by a
        configurable radius to capture transition-zone effects.

        Parameters
        ----------
        seismic_cube
            3D seismic amplitude cube with shape (i, j, k).
        boundaries
            Boolean mask indicating facies-boundary voxels.
        window
            Optional dilation radius. When ``None`` uses default. Must be positive.

        Returns
        -------
        BoundaryAmpsResult
            Contains amplitudes at/away from boundaries and boundary mask.

        Raises
        ------
        ValueError
            If window is non-positive or inputs have mismatched dimensions.

        Notes
        -----
        The boundary zone is created by dilating the input boundaries mask.
        A larger window captures more transition-zone amplitudes. The dilated
        zone may overlap, so samples are partitioned into exactly two groups:
        inside the dilated zone and outside.

        Examples
        --------
        >>> seismic = np.random.randn(10, 10, 20)
        >>> boundaries = np.zeros((10, 10, 20), dtype=bool)
        >>> boundaries[5, 5, :] = True  # Boundary line
        >>> extractor = BoundaryAmplitudeExtractor(dilation_window=2)
        >>> result = extractor.extract(seismic, boundaries)
        >>> print(f"Amplitudes at boundaries: {len(result.at_boundaries)}")
        """
        if window is None:
            window = self.dilation_window
        else:
            ArrayValidator.validate_positive_parameter(window, "window")

        # Align to common shape using composed CubeAligner
        seismic_aligned, boundaries_aligned = AlignmentOps.align_cubes(
            self._aligner, seismic_cube, boundaries.astype(np.int64)
        )
        boundaries_aligned = boundaries_aligned.astype(np.bool_)

        # Dilate boundaries to create a window
        boundary_zone = binary_dilation(boundaries_aligned, iterations=window)

        # Extract amplitudes using boundary mask
        at_boundaries = ExtractionOps.extract_by_mask(
            seismic_aligned.flatten(), boundary_zone.flatten(), mask_value=True
        )
        away_from_boundaries = ExtractionOps.extract_by_mask(
            seismic_aligned.flatten(), boundary_zone.flatten(), mask_value=False
        )

        return BoundaryAmpsResult(
            at_boundaries=at_boundaries,
            away_from_boundaries=away_from_boundaries,
            boundary_mask=boundary_zone.flatten(),
        )
