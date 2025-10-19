import numpy as np
import os

# Try to import numba; if not present we keep falling back to pure-NumPy code.
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


def reflectivity_from_ai(ai_time):
    """Calculate normal-incidence reflectivity from an AI cube.

    Args:
        ai_time (np.ndarray): 3D acoustic impedance cube in the time domain.

    Returns:
        np.ndarray: 3D reflectivity cube.
    """
    ai1 = ai_time[..., :-1]
    ai2 = ai_time[..., 1:]
    rc = (ai2 - ai1) / (ai2 + ai1 + 1e-9)
    return np.pad(rc, ((0, 0), (0, 0), (1, 0)), "constant")


def solve_zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1_deg):
    """Pure-NumPy batched solver for the full Zoeppritz equations (P-P).

    The implementation builds the 4x4 linear system per spatial point and
    solves them in CPU batches to limit peak memory. Returns complex128
    P-P reflection coefficients with the same spatial shape as inputs.
    """
    # Optionally use numba-accelerated implementation if available and
    # explicitly enabled via the environment variable ZOEPPRITZ_USE_NUMBA=1.
    use_numba = os.environ.get("ZOEPPRITZ_USE_NUMBA", "0") == "1" and _NUMBA_AVAILABLE
    if use_numba:
        # Precompute per-point complex angles using NumPy (scimath) and pass
        # those arrays into the numba kernel. numba's nopython mode does not
        # support numpy.lib.scimath, so we must do this step in Python.
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

    # Basic input validation
    if not (
        vp1.shape == vs1.shape == rho1.shape == vp2.shape == vs2.shape == rho2.shape
    ):
        raise ValueError("All property arrays must have the same spatial shape")

    spatial_shape = vp1.shape
    theta1 = np.deg2rad(theta1_deg)

    # Flatten spatial arrays so we can stream small batches through the 4x4 solver
    N = int(np.prod(spatial_shape)) if spatial_shape else 1
    vp1_flat = vp1.reshape(N)
    vs1_flat = vs1.reshape(N)
    rho1_flat = rho1.reshape(N)
    vp2_flat = vp2.reshape(N)
    vs2_flat = vs2.reshape(N)
    rho2_flat = rho2.reshape(N)

    # Ray parameter (per-point)
    p_flat = np.sin(theta1) / vp1_flat

    # Use scimath to handle evanescent waves
    theta2_flat = np.lib.scimath.arcsin(p_flat * vp2_flat)
    phi1_flat = np.lib.scimath.arcsin(p_flat * vs1_flat)
    phi2_flat = np.lib.scimath.arcsin(p_flat * vs2_flat)

    # Scalar incident-angle trig terms
    cth1 = np.cos(theta1)
    sth1 = np.sin(theta1)

    Rp_flat = np.empty(N, dtype=np.complex128)

    # Allow tuning CPU batch size via environment variable
    try:
        cpu_batch = int(os.environ.get("ZOEPPRITZ_CPU_BATCH", "1024"))
    except Exception:
        cpu_batch = 1024

    for i0 in range(0, N, cpu_batch):
        i1 = min(N, i0 + cpu_batch)
        # Slice batch
        vp1b = vp1_flat[i0:i1]
        vs1b = vs1_flat[i0:i1]
        rho1b = rho1_flat[i0:i1]
        vp2b = vp2_flat[i0:i1]
        vs2b = vs2_flat[i0:i1]
        rho2b = rho2_flat[i0:i1]

        theta2b = theta2_flat[i0:i1]
        phi1b = phi1_flat[i0:i1]
        phi2b = phi2_flat[i0:i1]

        b = i1 - i0
        Ai = np.empty((b, 4, 4), dtype=np.complex128)
        Bi = np.empty((b, 4, 1), dtype=np.complex128)

        # Fill Ai
        Ai[:, 0, 0] = cth1
        Ai[:, 0, 1] = -np.sin(phi1b)
        Ai[:, 0, 2] = np.cos(theta2b)
        Ai[:, 0, 3] = np.sin(phi2b)

        Ai[:, 1, 0] = sth1
        Ai[:, 1, 1] = np.cos(phi1b)
        Ai[:, 1, 2] = -np.sin(theta2b)
        Ai[:, 1, 3] = np.cos(phi2b)

        Ai[:, 2, 0] = rho1b * vp1b * np.cos(2 * phi1b)
        Ai[:, 2, 1] = -rho1b * vs1b * np.sin(2 * phi1b)
        Ai[:, 2, 2] = -rho2b * vp2b * np.cos(2 * phi2b)
        Ai[:, 2, 3] = -rho2b * vs2b * np.sin(2 * phi2b)

        Ai[:, 3, 0] = rho1b * vs1b * (vs1b / vp1b) * np.sin(2 * theta1)
        Ai[:, 3, 1] = rho1b * vs1b * np.cos(2 * phi1b)
        Ai[:, 3, 2] = rho2b * vs2b * (vs2b / vp2b) * np.sin(2 * theta2b)
        Ai[:, 3, 3] = -rho2b * vs2b * np.cos(2 * phi2b)

        # Fill Bi
        Bi[:, 0, 0] = cth1
        Bi[:, 1, 0] = -sth1
        Bi[:, 2, 0] = -rho1b * vp1b * np.cos(2 * phi1b)
        Bi[:, 3, 0] = rho1b * vs1b * (vs1b / vp1b) * np.sin(2 * theta1)

        # Solve batch
        Xi = np.linalg.solve(Ai, Bi)
        Rp_flat[i0:i1] = Xi[:, 0, 0]

    return Rp_flat.reshape(spatial_shape)


if _NUMBA_AVAILABLE:

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
