"""Domain-specific computation classes for rock physics attributes.

This module provides focused classes for computing specific rock physics
attributes using composition pattern for clean separation of concerns.

Classes:
    - AVOAttributesComputer: AVO intercept/gradient computation
    - LambdaMuRhoComputer: Lamé parameter computation
    - FluidFactorComputer: Fluid factor derivation
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence, cast

import numpy as np

from src.analysis.processors.types import FloatingArray

logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10
DEFAULT_AVO_ANGLES_DEG = (0, 5, 10, 15, 20, 25)
DEFAULT_FLUID_FACTOR_K = 1.0


class AVOAttributesComputer:
    """Computes AVO (Amplitude Variation with Offset) attributes.

    Handles computation of intercept and gradient from rock property cubes
    using least-squares fitting of reflectivity values across angles.
    """

    def compute(
        self,
        vp: FloatingArray,
        vs: FloatingArray,
        rho: FloatingArray,
        angles_deg: Sequence[float] = DEFAULT_AVO_ANGLES_DEG,
    ) -> Dict[str, FloatingArray]:
        """Compute AVO attributes (intercept, gradient) from rock property cubes.

        Args:
            vp, vs, rho: 3D numpy arrays with identical shapes (ni, nj, nk)
            angles_deg: sequence of incidence angles in degrees used for fitting

        Returns:
            dict with keys: 'intercept', 'gradient', 'product', 'scaled_gradient'

        Raises:
            ValueError: if input arrays have mismatched shapes, invalid dimensions, or angles_deg is empty
        """
        self._validate_inputs(vp, vs, rho)

        # Validate that angles are provided
        if not angles_deg or len(angles_deg) == 0:
            raise ValueError("angles_deg must contain at least one angle value")

        logger.info("Computing AVO attributes from rock physics...")

        # Local import to avoid heavy dependencies at module import time
        from src.signal.reflectivity import zoeppritz_solver as solver

        angles_rad = np.deg2rad(angles_deg)

        ni, nj, nk = vp.shape
        intercept = np.zeros((ni, nj, nk - 1), dtype=np.float32)
        gradient = np.zeros((ni, nj, nk - 1), dtype=np.float32)

        # Pre-compute design matrix for AVO fitting
        design_matrix = self._build_design_matrix(angles_rad)

        # Compute reflectivity at each angle and fit R(θ) = A + B*sin²θ per trace
        for k in range(nk - 1):
            vp1, vp2 = vp[:, :, k], vp[:, :, k + 1]
            vs1, vs2 = vs[:, :, k], vs[:, :, k + 1]
            rho1, rho2 = rho[:, :, k], rho[:, :, k + 1]

            # reflectivities: shape (n_angles, ni, nj)
            reflectivities = np.array(
                [
                    solver.solve(vp1, vs1, rho1, vp2, vs2, rho2, angle)
                    for angle in angles_rad
                ]
            )

            # Vectorized fitting: reshape to (n_angles, ni*nj), fit all traces at once
            # reflectivities_2d shape: (n_angles, ni*nj)
            reflectivities_2d = reflectivities.reshape(len(angles_rad), -1)

            # Fit all traces using least squares (fast batch operation)
            # Result shape: (2, ni*nj) where result[0] = intercepts, result[1] = gradients
            solution, _, rank, _ = np.linalg.lstsq(
                design_matrix, reflectivities_2d, rcond=None
            )

            # Validate solution rank: design_matrix has shape (n_angles, 2)
            expected_rank = min(design_matrix.shape)
            if rank < expected_rank:
                logger.warning(
                    f"Rank-deficient design matrix at layer {k}: "
                    f"rank={rank}, expected={expected_rank}. "
                    f"Solution may be unreliable."
                )

            coeffs = solution

            # Reshape back to (ni, nj) and assign to output arrays
            # Take real part of coefficients (lstsq may return complex for some inputs)
            intercept[:, :, k] = np.real(coeffs[0]).reshape(ni, nj)
            gradient[:, :, k] = np.real(coeffs[1]).reshape(ni, nj)

            # Mark invalid traces (where coefficients are NaN)
            self._mark_invalid_traces(intercept, gradient, k, ni, nj)

        product = intercept * gradient
        scaled_gradient = gradient / (np.abs(intercept) + EPSILON)

        return {
            "intercept": intercept,
            "gradient": gradient,
            "product": product,
            "scaled_gradient": scaled_gradient,
        }

    @staticmethod
    def _mark_invalid_traces(
        intercept: FloatingArray,
        gradient: FloatingArray,
        k: int,
        ni: int,
        nj: int,
    ) -> None:
        """Mark invalid traces where coefficients are NaN or infinite.

        Sets intercept and gradient to NaN for any trace with non-finite coefficients.
        Marks entire spatial traces (across all layers) as NaN if invalid at any layer.

        Args:
            intercept: Intercept volume (modified in-place) with shape (ni, nj, nlayers)
            gradient: Gradient volume (modified in-place) with shape (ni, nj, nlayers)
            k: Layer index
            ni: Number of rows
            nj: Number of columns
        """
        # Check each spatial position (i, j) at layer k
        for i in range(ni):
            for j in range(nj):
                # If either intercept or gradient is not finite at layer k,
                # mark entire trace (all layers) as NaN
                if not np.isfinite(intercept[i, j, k]) or not np.isfinite(
                    gradient[i, j, k]
                ):
                    intercept[i, j, :] = np.nan
                    gradient[i, j, :] = np.nan

    @staticmethod
    def _build_design_matrix(
        angles_rad: FloatingArray,
    ) -> FloatingArray:
        """Construct design matrix for AVO least-squares fitting.

        Builds the matrix for fitting AVO equation: R(θ) ≈ A + B*sin²(θ)

        Args:
            angles_rad: Incidence angles in radians

        Returns:
            Design matrix of shape (n_angles, 2) with columns [1, sin²(θ)]
        """
        sin2_theta = np.sin(angles_rad) ** 2
        return np.vstack([np.ones(len(sin2_theta)), sin2_theta]).T

    @staticmethod
    def _validate_inputs(
        vp: FloatingArray,
        vs: FloatingArray,
        rho: FloatingArray,
    ) -> None:
        """Validate that input arrays have compatible shapes and dimensions.

        Raises:
            ValueError: if inputs have mismatched shapes or wrong number of dimensions
        """
        if vp.shape != vs.shape or vp.shape != rho.shape:
            raise ValueError(
                f"Input arrays must have matching shapes. Got vp: {vp.shape}, "
                f"vs: {vs.shape}, rho: {rho.shape}"
            )
        if vp.ndim != 3:
            raise ValueError(
                f"Input arrays must be 3D. Got shape {vp.shape} ({vp.ndim}D)"
            )


class LambdaMuRhoComputer:
    """Computes Lamé parameters and derived rock physics attributes.

    Handles computation of Lambda-Rho and Mu-Rho from seismic velocities
    and density.
    """

    def compute(
        self,
        vp: FloatingArray,
        vs: FloatingArray,
        rho: FloatingArray,
    ) -> Dict[str, FloatingArray]:
        """Compute Lambda-Rho and Mu-Rho attributes.

        Returns a dict with 'lambda_rho', 'mu_rho' and 'lambda_mu_ratio'.

        Args:
            vp: P-wave velocity array
            vs: S-wave velocity array
            rho: Density array

        Returns:
            Dict containing computed Lamé parameters and their ratio
        """
        logger.info("Computing Lambda-Mu-Rho attributes...")

        mu = rho * vs**2
        with np.errstate(invalid="ignore"):
            lambda_mod = rho * vp**2 - 2 * mu

        # convention: lambda_rho = lambda * rho, mu_rho = mu * rho
        lambda_rho = lambda_mod * rho
        mu_rho = mu

        lambda_mu_ratio = lambda_mod / (mu + EPSILON)

        return {
            "lambda_rho": lambda_rho,
            "mu_rho": mu_rho,
            "lambda_mu_ratio": lambda_mu_ratio,
        }


class FluidFactorComputer:
    """Computes fluid factor attribute.

    Derives fluid-sensitive attributes from Lambda-Rho and Mu-Rho.
    """

    def compute(
        self,
        lambda_rho: FloatingArray,
        mu_rho: FloatingArray,
        k: float = DEFAULT_FLUID_FACTOR_K,
    ) -> FloatingArray:
        """Compute a simple fluid factor = lambda_rho - k * mu_rho.

        Args:
            lambda_rho: Lambda-Rho attribute volume
            mu_rho: Mu-Rho attribute volume
            k: tuning parameter; default is 1.0 which works for many clastic datasets

        Returns:
            Fluid factor volume as numpy array
        """
        return cast(FloatingArray, lambda_rho - k * mu_rho)
