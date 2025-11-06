"""Metrics scaffolding for resampling backends.


Provides a PlanFingerprint and a simple in-memory BackendMetrics collector
that records selection counts and cumulative runtimes per backend and plan
fingerprint. This is intentionally small and test-friendly.
"""


from __future__ import annotations


from dataclasses import dataclass
from typing import Dict, Tuple
import hashlib


from src.processing.resampling._plan import ResamplePlan


@dataclass(frozen=True)
class PlanFingerprint:
    """Immutable fingerprint of a resample plan for metrics tracking.

    Captures key plan parameters and a hash of the Vp array to enable
    identification of similar plans across multiple invocations.

    Attributes
    ----------
    ni, nj, nz, nt : int
        Dimensions of the plan.
    dt : float
        Time sample interval.
    uniform_twt : bool
        Whether uniform TWT sampling is used.
    vp_hash : str
        MD5 hash of the velocity array (or dimension signature if too large).
    """

    ni: int
    nj: int
    nz: int
    nt: int
    dt: float
    uniform_twt: bool
    vp_hash: str

    @classmethod
    def from_plan(cls, plan: ResamplePlan) -> "PlanFingerprint":
        """Create a fingerprint from a ResamplePlan.

        Parameters
        ----------
        plan : ResamplePlan
            The plan to fingerprint (must have ni, nj, nz, nt, dt, uniform_twt, vp_arr).

        Returns
        -------
        PlanFingerprint
            Fingerprint of the plan.
        """
        arr = plan.vp_arr
        h = hashlib.md5()
        try:
            buf = arr.ravel().view("uint8")
            if buf.nbytes <= 1024 * 1024:
                # Hash entire array if small enough
                h.update(buf.tobytes())
            else:
                # For large arrays, hash first and last chunks to reduce overhead
                h.update(buf[:1024].tobytes())
                h.update(buf[-1024:].tobytes())
        except Exception:
            # Fallback: hash plan dimensions
            h.update(str((plan.ni, plan.nj, plan.nz)).encode())

        return cls(
            ni=plan.ni,
            nj=plan.nj,
            nz=plan.nz,
            nt=plan.nt,
            dt=float(plan.dt),
            uniform_twt=bool(plan.uniform_twt),
            vp_hash=h.hexdigest(),
        )


class BackendMetrics:
    """In-memory metrics collector for backend selection and runtimes.

    Tracks how many times each backend was selected and cumulative runtime
    per backend/plan combination.

    Attributes
    ----------
    selection_counts : dict[str, int]
        Number of times each backend was selected.
    runtimes : dict[tuple[str, str], float]
        Cumulative runtime in seconds per (backend_name, vp_hash) pair.
    """

    def __init__(self) -> None:
        self.selection_counts: Dict[str, int] = {}
        self.runtimes: Dict[Tuple[str, str], float] = {}

    def record_selection(self, backend_name: str) -> None:
        """Record that a backend was selected.

        Parameters
        ----------
        backend_name : str
            Name of the backend selected.
        """
        self.selection_counts[backend_name] = (
            self.selection_counts.get(backend_name, 0) + 1
        )

    def record_runtime(
        self, backend_name: str, fingerprint: PlanFingerprint, seconds: float
    ) -> None:
        """Record runtime for a backend on a specific plan.

        Parameters
        ----------
        backend_name : str
            Name of the backend.
        fingerprint : PlanFingerprint
            Fingerprint of the plan.
        seconds : float
            Runtime in seconds.
        """
        key = (backend_name, fingerprint.vp_hash)
        self.runtimes[key] = self.runtimes.get(key, 0.0) + float(seconds)

    def get_selection_count(self, backend_name: str) -> int:
        """Get number of times a backend was selected.

        Parameters
        ----------
        backend_name : str
            Name of the backend.

        Returns
        -------
        int
            Number of selections (0 if never selected).
        """
        return int(self.selection_counts.get(backend_name, 0))

    def get_runtime(self, backend_name: str, fingerprint: PlanFingerprint) -> float:
        """Get cumulative runtime for a backend on a plan.

        Parameters
        ----------
        backend_name : str
            Name of the backend.
        fingerprint : PlanFingerprint
            Fingerprint of the plan.

        Returns
        -------
        float
            Cumulative runtime in seconds (0.0 if no prior runs).
        """
        return float(self.runtimes.get((backend_name, fingerprint.vp_hash), 0.0))


__all__ = [
    "PlanFingerprint",
    "BackendMetrics",
]
