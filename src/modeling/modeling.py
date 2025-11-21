"""Core modeling routines.

Object-oriented AVO modeling with angle handling, convolution, and caching.
Synthesizes angle-dependent seismograms using Zoeppritz approximations and
wavelet convolution.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from src.modeling.processors import ReflectivityComputer, WaveletConvolver
from src.utils.quantity import Quantity

logger = logging.getLogger(__name__)

# Type aliases
PropsDict: TypeAlias = dict[str, NDArray[np.floating[Any]] | Quantity]
PropsUnwrapped: TypeAlias = dict[str, NDArray[np.floating[Any]]]

# Configuration constants
CONVOLUTION_BLOCK_SIZE: int = 10
"""Number of depth samples to process per convolution block"""

__all__ = [
    "AngleModel",
    "AVOSynthesizer",
    "SynthesisConfig",
]


def unwrap_quantity(
    value: Quantity | NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Extract array from Quantity or return as ndarray."""
    if isinstance(value, Quantity):
        return value.array
    return np.asarray(value)


# Backwards-compatible alias: some tests and callers expect a private
# helper named `_unwrap_quantity`. Keep an alias to avoid breaking imports.
_unwrap_quantity = unwrap_quantity


@dataclass
class SynthesisConfig:
    """Configuration for AVO synthesis parameters."""

    use_quality_weighting: bool = False
    add_noise: bool = False
    snr_db: float = 20
    noise_seed: int | None = None


class AngleModel:
    """Manages angle-dependent quality weights and noise characteristics.

    Provides interpolated weights and noise levels for arbitrary angles
    based on calibrated reference values from the inversion study.
    """

    # Calibrated parameters from AVO modeling improvements (Oct 2025)
    QUALITY_WEIGHTS: dict[int, float] = {
        0: 0.90,
        5: 0.95,
        10: 0.98,
        15: 1.00,
        30: 0.70,
        45: 0.40,
    }

    NOISE_SIGMA: dict[int, float] = {
        0: 0.011,
        5: 0.007,
        10: 0.004,
        15: 0.002,
        30: 0.033,
        45: 0.023,
    }

    def quality_weight(self, angle_deg: float) -> float:
        """Get interpolated quality weight for given angle."""
        return self._interpolate(angle_deg, self.QUALITY_WEIGHTS)

    def noise_level(self, angle_deg: float) -> float:
        """Get interpolated noise sigma for given angle."""
        return self._interpolate(angle_deg, self.NOISE_SIGMA)

    def _interpolate(self, angle_deg: float, lookup: dict[int, float]) -> float:
        """Linear interpolation for arbitrary angles from lookup table.

        Uses numpy.interp for robust handling of boundary cases and
        linearly interpolates between calibrated angles.
        """
        angles_sorted = np.array(sorted(lookup.keys()), dtype=float)
        values = np.array([lookup[a] for a in angles_sorted.astype(int)])
        return float(np.interp(angle_deg, angles_sorted, values))

    def add_noise(
        self,
        seismic: NDArray[np.floating[Any]],
        angle: float,
        snr_db: float,
        seed: int | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Add realistic angle-dependent noise to seismic data.

        Combines systematic (angle-dependent) and random (SNR-dependent) noise.
        """
        if seed is not None:
            np.random.seed(seed)

        sigma_systematic = self.noise_level(angle)
        signal_power = float(np.var(seismic))
        target_snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / target_snr_linear

        random_data_1 = np.random.randn(*seismic.shape)
        random_data_2 = np.random.randn(*seismic.shape)
        noise_random: NDArray[np.floating[Any]] = np.asarray(
            random_data_1, dtype=np.float64
        ) * np.sqrt(noise_power)
        noise_systematic: NDArray[np.floating[Any]] = (
            np.asarray(random_data_2, dtype=np.float64) * sigma_systematic
        )
        total_noise: NDArray[np.floating[Any]] = noise_random + noise_systematic

        noisy_seismic: NDArray[np.floating[Any]] = seismic + total_noise.astype(
            seismic.dtype
        )
        return noisy_seismic.astype(np.float32)

    def weighted_stack(
        self,
        angle_stacks: list[NDArray[np.floating[Any]]],
        angles: list[float],
    ) -> NDArray[np.floating[Any]]:
        """Combine angle stacks using quality weights.

        Args:
            angle_stacks: List of angle-dependent seismic stacks
            angles: Corresponding angles in degrees

        Returns:
            Weighted combination of all angle stacks
        """
        if len(angle_stacks) != len(angles):
            raise ValueError("Number of angle stacks must match number of angles")

        weights = np.array([self.quality_weight(a) for a in angles])
        weights = weights / weights.sum()

        weighted_stack = np.zeros_like(angle_stacks[0])
        for stack, weight in zip(angle_stacks, weights):
            weighted_stack += stack * weight

        return weighted_stack


class AVOSynthesizer:
    """Synthesizes angle-dependent AVO seismograms from rock properties.

    Handles Zoeppritz reflectivity computation, wavelet convolution, and
    angle-dependent processing including weighting and noise.
    """

    def __init__(
        self,
        angle_model: AngleModel | None = None,
        reflectivity_computer: ReflectivityComputer | None = None,
        wavelet_convolver: WaveletConvolver | None = None,
    ):
        """Initialize with optional custom components.

        Args:
            angle_model: AngleModel instance for weights/noise; uses default if None
            reflectivity_computer: ReflectivityComputer instance; uses default if None
            wavelet_convolver: WaveletConvolver instance; uses default if None
        """
        self.angle_model = angle_model or AngleModel()
        self.reflectivity_computer = reflectivity_computer or ReflectivityComputer(
            block_size=CONVOLUTION_BLOCK_SIZE
        )
        self.wavelet_convolver = wavelet_convolver or WaveletConvolver()

    def create_synthetics(
        self,
        props_time: PropsDict,
        angles: list[float],
        wavelet: NDArray[np.floating[Any]],
        config: SynthesisConfig | None = None,
    ) -> tuple[list[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]:
        """Create angle-dependent AVO synthetics from time-domain properties.

        Args:
            props_time: Dict with 'vp', 'vs', 'rho' as arrays or Quantity
            angles: List of incidence angles in degrees
            wavelet: Source wavelet for convolution
            config: SynthesisConfig with weighting, noise, and SNR settings

        Returns:
            (angle_stacks, full_stack): List of angle-dependent stacks and combined stack
        """
        # The orchestration in this method necessarily introduces several
        # local variables for buffer management, progress reporting and
        # intermediate results. Suppress the too-many-locals warning here.

        config = config or SynthesisConfig()

        # Unwrap Quantity objects to numeric arrays
        vp = unwrap_quantity(props_time["vp"])
        vs = unwrap_quantity(props_time["vs"])
        rho = unwrap_quantity(props_time["rho"])

        angle_stacks: list[NDArray[np.floating[Any]]] = []
        full_stack: NDArray[np.floating[Any]] = np.zeros(vp.shape, dtype=np.float32)

        # Narrowly allow many locals in this orchestration method.
        # The method mainly wires processors together; keeping the
        # implementation explicit improves readability for reviewers.

        with tqdm(
            total=len(angles),
            desc="Processing Angles",
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
        ) as progress:
            for idx, angle in enumerate(angles):
                angle_stack_full = self._process_angle(vp, vs, rho, angle, wavelet)

                if config.add_noise:
                    angle_stack_full = self.angle_model.add_noise(
                        angle_stack_full,
                        angle,
                        snr_db=config.snr_db,
                        seed=config.noise_seed,
                    )

                angle_stacks.append(angle_stack_full)
                full_stack += angle_stack_full / float(len(angles))

                progress.update(1)

                if sys.gettrace() is not None:
                    logger.debug("[DEBUG] Angle %d/%d completed", idx + 1, len(angles))

        if config.use_quality_weighting:
            full_stack = self.angle_model.weighted_stack(angle_stacks, angles)

        return angle_stacks, full_stack

    def _process_angle(
        self,
        vp: NDArray[np.floating[Any]],
        vs: NDArray[np.floating[Any]],
        rho: NDArray[np.floating[Any]],
        angle: float,
        wavelet: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Process a single angle: compute reflectivity and convolve."""
        # The helper intentionally accepts multiple array-like inputs; silence
        # argument-count warnings for this low-level processor method.

        # Compute reflectivity using dedicated processor
        rc_pad = self.reflectivity_computer.compute_reflectivity(vp, vs, rho, angle)

        # Apply wavelet convolution using dedicated processor
        angle_stack_full = self.wavelet_convolver.convolve_3d(rc_pad, wavelet)

        return angle_stack_full
