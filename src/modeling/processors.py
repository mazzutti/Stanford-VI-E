"""Specialized processors for AVO modeling computations.

Extracts concerns from AVOSynthesizer into focused, testable classes:
- ReflectivityComputer: Zoeppritz-based reflectivity computation
- WaveletConvolver: 3D seismic wavelet convolution
"""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve
import logging

logger = logging.getLogger(__name__)

__all__ = ["ReflectivityComputer", "WaveletConvolver"]


class ReflectivityComputer:
    """Computes angle-dependent reflectivity using Zoeppritz approximation.

    Handles block-wise reflectivity computation for memory efficiency
    and numerical stability.
    """

    def __init__(self, block_size: int = 10):
        """Initialize reflectivity computer.

        Args:
            block_size: Number of depth samples to process per block
        """
        self.block_size = block_size

    def compute_reflectivity(
        self,
        vp: np.ndarray,
        vs: np.ndarray,
        rho: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Compute reflectivity cube at given incidence angle.

        Uses block-wise processing to balance memory usage and performance.

        Args:
            vp: P-wave velocity cube (nz, nx, ny)
            vs: S-wave velocity cube (nz, nx, ny)
            rho: Density cube (nz, nx, ny)
            angle: Incidence angle in degrees

        Returns:
            Reflectivity cube (nz, nx, ny) with zeros at top level
        """
        from src.signal.reflectivity import ZoeppritzSolver

        solver = ZoeppritzSolver()
        ni, nj, nk = vp.shape
        rc_full = np.zeros((ni, nj, nk), dtype=np.float32)

        for i0 in range(0, ni, self.block_size):
            i1 = min(ni, i0 + self.block_size)

            # Extract blocks
            vp_block = vp[i0:i1]
            vs_block = vs[i0:i1]
            rho_block = rho[i0:i1]

            # Compute contrasts at interfaces (using [-1] and [1:])
            vp1b, vp2b = vp_block[..., :-1], vp_block[..., 1:]
            vs1b, vs2b = vs_block[..., :-1], vs_block[..., 1:]
            rho1b, rho2b = rho_block[..., :-1], rho_block[..., 1:]

            # Zoeppritz approximation
            rc_values = solver.solve(vp1b, vs1b, rho1b, vp2b, vs2b, rho2b, angle)
            rc_real = np.real(rc_values).astype(np.float32)

            # Pad to match input depth dimension (first level = 0)
            rc_pad = np.zeros((i1 - i0, nj, nk), dtype=np.float32)
            rc_pad[..., 1:] = rc_real

            rc_full[i0:i1] = rc_pad

        return rc_full


class WaveletConvolver:
    """Performs efficient 3D wavelet convolution on seismic data.

    Handles 1D wavelet convolution across all traces in a 3D cube
    using FFT-based methods for efficiency.
    """

    @staticmethod
    def convolve_3d(
        cube: np.ndarray,
        wavelet: np.ndarray,
    ) -> np.ndarray:
        """Apply 3D convolution: cube with 1D wavelet.

        Vectorized convolution across all traces using FFT.
        Preserves trace length using 'same' mode.

        Args:
            cube: 3D seismic cube (nz, nx, ny)
            wavelet: 1D source wavelet

        Returns:
            Convolved cube same shape as input
        """
        _, nx, ny = cube.shape
        result = np.zeros_like(cube, dtype=np.float32)

        for ix in range(nx):
            for iy in range(ny):
                trace = cube[:, ix, iy]
                result[:, ix, iy] = convolve(
                    trace, wavelet, mode="same", method="fft"
                ).astype(np.float32)

        return result
