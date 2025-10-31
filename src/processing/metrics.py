"""Metrics scaffolding for resampling backends.

Provides a PlanFingerprint and a simple in-memory BackendMetrics collector
that records selection counts and cumulative runtimes per backend and plan
fingerprint. This is intentionally small and test-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import hashlib

from src.utils.facades import LazyObjectProxy
from src.processing.resample_plan import ResamplePlan


@dataclass(frozen=True)
class PlanFingerprint:
    ni: int
    nj: int
    nz: int
    nt: int
    dt: float
    uniform_twt: bool
    vp_hash: str

    @classmethod
    def from_plan(cls, plan: ResamplePlan) -> "PlanFingerprint":
        # plan is expected to have ni,nj,nz,nt,dt,uniform_twt and vp_arr
        # compute a short hash from vp_arr sample
        arr = plan.vp_arr
        h = hashlib.md5()
        try:
            buf = arr.ravel().view("uint8")
            if buf.nbytes <= 1024 * 1024:
                h.update(buf.tobytes())
            else:
                # sample first/last chunks
                h.update(buf[:1024].tobytes())
                h.update(buf[-1024:].tobytes())
        except Exception:
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
    """In-memory metrics collector for backend selection and runtimes."""

    def __init__(self) -> None:
        # counts: {backend_name: int}
        self.selection_counts: Dict[str, int] = {}
        # runtimes: {(backend_name, vp_hash): cumulative_seconds}
        self.runtimes: Dict[Tuple[str, str], float] = {}

    def record_selection(self, backend_name: str) -> None:
        self.selection_counts[backend_name] = (
            self.selection_counts.get(backend_name, 0) + 1
        )

    def record_runtime(
        self, backend_name: str, fingerprint: PlanFingerprint, seconds: float
    ) -> None:
        key = (backend_name, fingerprint.vp_hash)
        self.runtimes[key] = self.runtimes.get(key, 0.0) + float(seconds)

    def get_selection_count(self, backend_name: str) -> int:
        return int(self.selection_counts.get(backend_name, 0))

    def get_runtime(self, backend_name: str, fingerprint: PlanFingerprint) -> float:
        return float(self.runtimes.get((backend_name, fingerprint.vp_hash), 0.0))


# Export a module-level proxy instance for convenience.
global_metrics = LazyObjectProxy(lambda: BackendMetrics())


__all__ = ["PlanFingerprint", "BackendMetrics", "global_metrics"]


# Object-oriented facade for the metrics collector
class MetricsCollector:
    def __init__(self):
        self._metrics = BackendMetrics()

    def record_selection(self, backend_name: str) -> None:
        return self._metrics.record_selection(backend_name)

    def record_runtime(
        self, backend_name: str, fingerprint: PlanFingerprint, seconds: float
    ) -> None:
        return self._metrics.record_runtime(backend_name, fingerprint, seconds)

    def get_selection_count(self, backend_name: str) -> int:
        return self._metrics.get_selection_count(backend_name)

    def get_runtime(self, backend_name: str, fingerprint: PlanFingerprint) -> float:
        return self._metrics.get_runtime(backend_name, fingerprint)


# provide a module-level lazy proxy for MetricsCollector
metrics_collector = LazyObjectProxy(lambda: MetricsCollector())


def get_metrics_collector(
    collector: MetricsCollector | None = None,
) -> "MetricsCollector":
    """Return the provided collector or the module-level lazy singleton.

    If `collector` is provided, it is returned unchanged (useful for
    dependency injection). Otherwise the module-level lazy
    `metrics_collector` is returned.
    """
    return _impl_get_metrics_collector(collector)


def _impl_get_metrics_collector(
    collector: MetricsCollector | None = None,
) -> "MetricsCollector":
    return collector if collector is not None else metrics_collector


__all__.extend(["MetricsCollector", "metrics_collector", "get_metrics_collector"])


def get_global_metrics(inst: BackendMetrics | None = None) -> "BackendMetrics":
    """Return the provided BackendMetrics instance or the module-level lazy proxy.

    Provides a single helper consistent with the rest of the codebase.
    """
    return _impl_get_global_metrics(inst)


def _impl_get_global_metrics(inst: BackendMetrics | None = None) -> "BackendMetrics":
    return inst if inst is not None else global_metrics


__all__.append("get_global_metrics")
