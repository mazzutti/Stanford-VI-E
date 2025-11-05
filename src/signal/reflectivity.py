"""Reflectivity helpers.

Canonical implementation for reflectivity and Zoeppritz routines. These
signal/math helpers are used by modeling and analysis pipelines.
"""

import numpy as np
import os
import logging
from numba import njit, prange

__all__ = [
    "ReflectivityCalculator",
    "ZoeppritzSolver",
    "reflectivity_calc",
    "zoeppritz_solver",
    "get_reflectivity_calc",
    "get_zoeppritz_solver",
    "configure_reflectivity",
]

# Module logger
logger = logging.getLogger(__name__)


class ReflectivityCalculator:
    """Calculate reflectivity-related quantities.

    Encapsulates reflectivity routines and accepts Quantity or ndarray
    inputs, returning a `Quantity` when possible for semantic consistency.
    """

    def __init__(self, pad_width=((0, 0), (0, 0), (1, 0))):
        self.pad_width = pad_width


class ZoeppritzSolver:
    """Batched Zoeppritz equation solver for P-P reflection coefficients.

    Provides optimized Zoeppritz computations using Numba JIT compilation.
    Numba is a required dependency for this solver.
    """

    def __init__(self, cpu_batch: int = None):
        if cpu_batch is None:
            try:
                self.cpu_batch = int(os.environ.get("ZOEPPRITZ_CPU_BATCH", "1024"))
            except Exception:
                self.cpu_batch = 1024
        else:
            self.cpu_batch = int(cpu_batch)

    def solve(
        self,
        vp1: np.ndarray,
        vs1: np.ndarray,
        rho1: np.ndarray,
        vp2: np.ndarray,
        vs2: np.ndarray,
        rho2: np.ndarray,
        theta1_deg: float,
    ) -> np.ndarray:
        """Solve for P-P reflection coefficients.

        Args are the same as the prior function. Returns complex128 array
            shaped like the inputs.
        """
        spatial_shape = vp1.shape
        N = int(np.prod(spatial_shape)) if spatial_shape else 1
        vp1f = vp1.reshape(N)
        vs1f = vs1.reshape(N)
        rho1f = rho1.reshape(N)
        vp2f = vp2.reshape(N)
        vs2f = vs2.reshape(N)
        rho2f = rho2.reshape(N)

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
        return rp_flat.reshape(spatial_shape)


# Numba-compiled Gaussian solver is always available (Numba is a required dependency)


@njit
def _solve_4x4_numba(A, b):
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
            if akk == 0:
                continue
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
    vp1f, vs1f, rho1f, vp2f, vs2f, rho2f, theta1, theta2f, phi1f, phi2f
):
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


# Module-level singletons for reuse across the package
# Use a simple initialization approach that avoids the broken LazyObjectProxy
_reflectivity_calc_instance: ReflectivityCalculator | None = None
_zoeppritz_solver_instance: ZoeppritzSolver | None = None


def _get_reflectivity_calc_instance() -> ReflectivityCalculator:
    """Lazy initialize reflectivity_calc singleton."""
    global _reflectivity_calc_instance
    if _reflectivity_calc_instance is None:
        _reflectivity_calc_instance = ReflectivityCalculator()
    return _reflectivity_calc_instance


def _get_zoeppritz_solver_instance() -> ZoeppritzSolver:
    """Lazy initialize zoeppritz_solver singleton."""
    global _zoeppritz_solver_instance
    if _zoeppritz_solver_instance is None:
        _zoeppritz_solver_instance = ZoeppritzSolver()
    return _zoeppritz_solver_instance


# Provide backward-compatible module-level access
# These are initialized on first use to avoid breaking existing code
class _LazyReflectivityProxy:
    """Lazy proxy for backward compatibility."""

    def __getattr__(self, name):
        return getattr(_get_reflectivity_calc_instance(), name)


class _LazyZoeppritzProxy:
    """Lazy proxy for backward compatibility."""

    def __getattr__(self, name):
        return getattr(_get_zoeppritz_solver_instance(), name)


# Create module-level proxies that initialize on first access
reflectivity_calc = _LazyReflectivityProxy()
zoeppritz_solver = _LazyZoeppritzProxy()


def configure_reflectivity(
    use_numba: bool | None = None, cpu_batch: int | None = None
) -> None:
    """Configure module-level reflectivity singletons.

    Args:
        use_numba: If True/False, set solver numba usage. If None, leave unchanged.
        cpu_batch: If provided, set the CPU batch size for the Zoeppritz solver.

    This convenience function updates the two module-level singletons
    (`reflectivity_calc`, `zoeppritz_solver`) so callers can centrally tune
    performance without constructing new objects.
    """
    if cpu_batch is not None:
        solver = _get_zoeppritz_solver_instance()
        try:
            solver.cpu_batch = int(cpu_batch)
        except Exception:
            pass


def get_reflectivity_calc(config: dict | None = None) -> "ReflectivityCalculator":
    """Return the module-level reflectivity_calc singleton when `config` is None,
    otherwise return a new ReflectivityCalculator instance.

    This follows the repository convention of providing `get_*` helpers for
    module-level lazy singletons to simplify testing and dependency injection.
    """
    if config is None:
        return _get_reflectivity_calc_instance()
    return ReflectivityCalculator()


def get_zoeppritz_solver(config: dict | None = None) -> "ZoeppritzSolver":
    """Return the module-level zoeppritz_solver singleton when `config` is None,
    otherwise return a new ZoeppritzSolver instance with optional config.
    """
    if config is None:
        return _get_zoeppritz_solver_instance()
    cpu_batch = None
    if isinstance(config, dict):
        cpu_batch = config.get("cpu_batch", None)
    return ZoeppritzSolver(cpu_batch=cpu_batch)
