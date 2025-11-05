"""Core modeling routines.

Object-oriented AVO modeling with angle handling, convolution, and caching.
Synthesizes angle-dependent seismograms using Zoeppritz approximations and
wavelet convolution.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.signal import convolve
from tqdm.auto import tqdm
import sys
import logging
from typing import TypeAlias
from src.utils.quantity import Quantity


logger = logging.getLogger(__name__)

# Type aliases
PropsDict: TypeAlias = dict[str, np.ndarray | Quantity]
PropsUnwrapped: TypeAlias = dict[str, np.ndarray]

# Configuration constants
CONVOLUTION_BLOCK_SIZE: int = 10
"""Number of depth samples to process per convolution block"""

CALIBRATION_ANGLES: list[int] = [0, 5, 10, 15, 30, 45]
"""Angles at which quality weights and noise levels are calibrated"""

__all__ = [
    "AngleModel",
    "AVOSynthesizer",
    "SynthesisConfig",
]


def _unwrap_quantity(value: Quantity | np.ndarray) -> np.ndarray:
    """Extract array from Quantity or return as ndarray."""
    if isinstance(value, Quantity):
        return value.array
    return np.asarray(value)


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
        seismic: np.ndarray,
        angle: float,
        snr_db: float,
        seed: int | None = None,
    ) -> np.ndarray:
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
        noise_random: np.ndarray = np.asarray(
            random_data_1, dtype=np.float64
        ) * np.sqrt(noise_power)
        noise_systematic: np.ndarray = (
            np.asarray(random_data_2, dtype=np.float64) * sigma_systematic
        )
        total_noise: np.ndarray = noise_random + noise_systematic

        noisy_seismic: np.ndarray = seismic + total_noise.astype(seismic.dtype)
        return noisy_seismic.astype(np.float32)

    def weighted_stack(
        self,
        angle_stacks: list[np.ndarray],
        angles: list[float],
    ) -> np.ndarray:
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

    def __init__(self, angle_model: AngleModel | None = None):
        """Initialize with optional custom angle model.

        Args:
            angle_model: AngleModel instance for weights/noise; uses default if None
        """
        self.angle_model = angle_model or AngleModel()

    def run_convolution_3d(
        self,
        rc_cube: np.ndarray,
        wavelet: np.ndarray,
    ) -> np.ndarray:
        """Apply 3D convolution on reflectivity cube with wavelet.

        Vectorized convolution across all traces (more efficient than
        apply_along_axis). Preserves trace length using 'same' mode.

        Args:
            rc_cube: Reflectivity cube (nz, nx, ny)
            wavelet: Source wavelet (1D)

        Returns:
            Convolved seismogram cube same shape as rc_cube
        """
        nz, nx, ny = rc_cube.shape
        result = np.zeros_like(rc_cube, dtype=np.float32)

        for ix in range(nx):
            for iy in range(ny):
                trace = rc_cube[:, ix, iy]
                result[:, ix, iy] = convolve(
                    trace, wavelet, mode="same", method="fft"
                ).astype(np.float32)

        return result

    def create_synthetics(
        self,
        props_time: PropsDict,
        angles: list[float],
        wavelet: np.ndarray,
        config: SynthesisConfig | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Create angle-dependent AVO synthetics from time-domain properties.

        Args:
            props_time: Dict with 'vp', 'vs', 'rho' as arrays or Quantity
            angles: List of incidence angles in degrees
            wavelet: Source wavelet for convolution
            config: SynthesisConfig with weighting, noise, and SNR settings

        Returns:
            (angle_stacks, full_stack): List of angle-dependent stacks and combined stack
        """
        config = config or SynthesisConfig()

        # Unwrap Quantity objects to numeric arrays
        vp = _unwrap_quantity(props_time["vp"])
        vs = _unwrap_quantity(props_time["vs"])
        rho = _unwrap_quantity(props_time["rho"])

        ni, nj, nk = vp.shape
        angle_stacks = []
        full_stack = np.zeros((ni, nj, nk), dtype=np.float32)
        n_angles = len(angles)

        debug_mode = sys.gettrace() is not None

        with tqdm(
            total=n_angles,
            desc="Processing Angles",
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
        ) as bar:
            for idx, angle in enumerate(angles):
                angle_stack_full = self._process_angle(
                    vp, vs, rho, angle, wavelet, ni, nj, nk, CONVOLUTION_BLOCK_SIZE
                )

                if config.add_noise:
                    angle_stack_full = self.angle_model.add_noise(
                        angle_stack_full,
                        angle,
                        snr_db=config.snr_db,
                        seed=config.noise_seed,
                    )

                angle_stacks.append(angle_stack_full)
                full_stack += angle_stack_full / float(n_angles)

                bar.update(1)

                if debug_mode:
                    logger.debug("[DEBUG] Angle %d/%d completed", idx + 1, n_angles)

        if config.use_quality_weighting:
            full_stack = self.angle_model.weighted_stack(angle_stacks, angles)

        return angle_stacks, full_stack

    def _process_angle(
        self,
        vp: np.ndarray,
        vs: np.ndarray,
        rho: np.ndarray,
        angle: float,
        wavelet: np.ndarray,
        ni: int,
        nj: int,
        nk: int,
        block_i: int,
    ) -> np.ndarray:
        """Process a single angle: compute reflectivity and convolve."""
        from src.signal.reflectivity import zoeppritz_solver

        angle_stack_full = np.zeros((ni, nj, nk), dtype=np.float32)

        for i0 in range(0, ni, block_i):
            i1 = min(ni, i0 + block_i)

            vp_block = vp[i0:i1]
            vs_block = vs[i0:i1]
            rho_block = rho[i0:i1]

            vp1b, vp2b = vp_block[..., :-1], vp_block[..., 1:]
            vs1b, vs2b = vs_block[..., :-1], vs_block[..., 1:]
            rho1b, rho2b = rho_block[..., :-1], rho_block[..., 1:]

            rc_values = zoeppritz_solver.solve(
                vp1b, vs1b, rho1b, vp2b, vs2b, rho2b, angle
            )
            rc_real = np.real(rc_values).astype(np.float32)
            rc_pad = np.zeros((i1 - i0, nj, nk), dtype=np.float32)
            rc_pad[..., 1:] = rc_real

            angle_block = self.run_convolution_3d(rc_pad, wavelet)
            angle_stack_full[i0:i1] = angle_block

        return angle_stack_full
