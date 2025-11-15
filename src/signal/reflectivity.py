"""Reflectivity helpers.

Canonical implementation for Zoeppritz equations used by modeling and analysis
pipelines. Provides optimized reflection coefficient calculations with Numba
JIT compilation.
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
import logging
from numba import njit, prange

__all__ = ["ZoeppritzSolver"]

# Module logger
logger = logging.getLogger(__name__)


class ZoeppritzSolver:
    """Batched Zoeppritz equation solver for P-P reflection coefficients.

    Provides optimized Zoeppritz computations using Numba JIT compilation.
    Numba is a required dependency for this solver.
    """

    def __init__(self, cpu_batch: int | None = None):
        if cpu_batch is None:
            try:
                import os

                self.cpu_batch = int(os.environ.get("ZOEPPRITZ_CPU_BATCH", "1024"))
            except Exception:
                self.cpu_batch = 1024
        else:
            self.cpu_batch = int(cpu_batch)

    def solve(
        self,
        vp1: NDArray[np.floating[Any]],
        vs1: NDArray[np.floating[Any]],
        rho1: NDArray[np.floating[Any]],
        vp2: NDArray[np.floating[Any]],
        vs2: NDArray[np.floating[Any]],
        rho2: NDArray[np.floating[Any]],
        theta1_deg: float,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Solve for P-P reflection coefficients.

        Args are the same as the prior function. Returns complex128 array
            shaped like the inputs.
        """
        spatial_shape = vp1.shape
        if spatial_shape:
            n = 1
            for s in spatial_shape:
                n *= int(s)
        else:
            n = 1
        vp1f = vp1.reshape(n)
        vs1f = vs1.reshape(n)
        rho1f = rho1.reshape(n)
        vp2f = vp2.reshape(n)
        vs2f = vs2.reshape(n)
        rho2f = rho2.reshape(n)

        theta1 = np.deg2rad(theta1_deg)
        p_flat = np.sin(theta1) / vp1f

        theta2_flat = np.lib.scimath.arcsin(p_flat * vp2f)
        phi1_flat = np.lib.scimath.arcsin(p_flat * vs1f)
        phi2_flat = np.lib.scimath.arcsin(p_flat * vs2f)

        rp_flat = _numba_solve_zoeppritz(
            vp1f,
            vs1f,
            rho1f,
            vp2f,
            vs2f,
            rho2f,
            theta1,
            theta2_flat,
            phi1_flat,
            phi2_flat,
        )
        return np.asarray(rp_flat.reshape(spatial_shape), dtype=np.complex128)


@njit
def _solve_4x4_numba(
    A: "NDArray[np.complex128]", b: "NDArray[np.complex128]"
) -> "NDArray[np.complex128]":
    # In-place Gaussian elimination with partial pivoting
    M = A.copy()
    rhs = b.copy()
    # Forward elimination
    for k in range(4):
        piv = k
        maxval = abs(M[k, k])
        for ii in range(k + 1, 4):
            aval = abs(M[ii, k])
            if aval > maxval:
                maxval = aval
                piv = ii
        if piv != k:
            for jj in range(k, 4):
                tmp = M[k, jj]
                M[k, jj] = M[piv, jj]
                M[piv, jj] = tmp
            tmp = rhs[k]
            rhs[k] = rhs[piv]
            rhs[piv] = tmp
        akk = M[k, k]
        if akk != 0:
            for ii in range(k + 1, 4):
                factor = M[ii, k] / akk
                rhs[ii] = rhs[ii] - factor * rhs[k]
                for jj in range(k, 4):
                    M[ii, jj] = M[ii, jj] - factor * M[k, jj]
    # Back substitution
    x = np.empty(4, dtype=np.complex128)
    for ii in range(3, -1, -1):
        s = rhs[ii]
        for jj in range(ii + 1, 4):
            s = s - M[ii, jj] * x[jj]
        if M[ii, ii] == 0:
            x[ii] = 0
        else:
            x[ii] = s / M[ii, ii]
    return x


@njit(parallel=True)
def _numba_solve_zoeppritz(
    vp1f: "NDArray[np.floating[Any]]",
    vs1f: "NDArray[np.floating[Any]]",
    rho1f: "NDArray[np.floating[Any]]",
    vp2f: "NDArray[np.floating[Any]]",
    vs2f: "NDArray[np.floating[Any]]",
    rho2f: "NDArray[np.floating[Any]]",
    theta1: float,
    theta2f: "NDArray[np.floating[Any]]",
    phi1f: "NDArray[np.floating[Any]]",
    phi2f: "NDArray[np.floating[Any]]",
) -> "NDArray[np.complex128]":
    N = vp1f.size
    cth1 = np.cos(theta1)
    sth1 = np.sin(theta1)

    out = np.empty(N, dtype=np.complex128)
    for i in prange(N):
        theta2 = theta2f[i]
        phi1 = phi1f[i]
        phi2 = phi2f[i]

        A = np.empty((4, 4), dtype=np.complex128)
        b = np.empty(4, dtype=np.complex128)
        A[0, 0] = cth1
        A[0, 1] = -np.sin(phi1)
        A[0, 2] = np.cos(theta2)
        A[0, 3] = np.sin(phi2)

        A[1, 0] = sth1
        A[1, 1] = np.cos(phi1)
        A[1, 2] = -np.sin(theta2)
        A[1, 3] = np.cos(phi2)

        A[2, 0] = rho1f[i] * vp1f[i] * np.cos(2 * phi1)
        A[2, 1] = -rho1f[i] * vs1f[i] * np.sin(2 * phi1)
        A[2, 2] = -rho2f[i] * vp2f[i] * np.cos(2 * phi2)
        A[2, 3] = -rho2f[i] * vs2f[i] * np.sin(2 * phi2)

        A[3, 0] = rho1f[i] * vs1f[i] * (vs1f[i] / vp1f[i]) * np.sin(2 * theta1)
        A[3, 1] = rho1f[i] * vs1f[i] * np.cos(2 * phi1)
        A[3, 2] = rho2f[i] * vs2f[i] * (vs2f[i] / vp2f[i]) * np.sin(2 * theta2)
        A[3, 3] = -rho2f[i] * vs2f[i] * np.cos(2 * phi2)

        b[0] = cth1
        b[1] = -sth1
        b[2] = -rho1f[i] * vp1f[i] * np.cos(2 * phi1)
        b[3] = rho1f[i] * vs1f[i] * (vs1f[i] / vp1f[i]) * np.sin(2 * theta1)

        x = _solve_4x4_numba(A, b)
        out[i] = x[0]

    return out.reshape((vp1f.shape[0],))
