"""Simple BackendManager to centralize backend registration and selection.

This is intentionally small and aligned with the existing
`backends` module API. It provides an instance that can be used for
dependency injection when needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.processing.resampling.backends.base import ResamplerBackend
from src.processing.resampling.plan import ResamplePlan
from src.processing.metrics import get_global_metrics, PlanFingerprint
from typing import Tuple
from src.processing.core.singleton import SingletonFactory


@dataclass
class _BackendEntry:
    name: str
    backend: ResamplerBackend


class BackendManager:
    """Registry and selector for ResamplerBackend implementations.

    Simple selection heuristic: prefer registered backends in registration
    order and call `supports(plan)` to find the first that supports the plan.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, ResamplerBackend] = {}
        self._verbose = False

    def register(self, name: str, backend: ResamplerBackend) -> None:
        if name in self._registry:
            raise KeyError(f"backend '{name}' already registered")
        self._registry[name] = backend

    def list_backends(self) -> List[str]:
        return list(self._registry.keys())

    def get(self, name: str) -> Optional[ResamplerBackend]:
        return self._registry.get(name)

    def get_best(self, plan: ResamplePlan) -> Optional[ResamplerBackend]:
        # simple linear scan and supports check is the default behavior.
        # If metrics are available, prefer backends with historic usage and
        # lower runtimes for plans similar to the provided plan.
        # use the module-level metrics
        metrics = get_global_metrics()

        candidates: list[Tuple[str, ResamplerBackend]] = []
        for name, backend in self._registry.items():
            try:
                if backend.supports(plan):
                    candidates.append((name, backend))
            except Exception:
                # backend misbehaved; skip
                continue

        if not candidates:
            return None

        # If no metrics available, return first candidate (preserves order)
        if metrics is None:
            name, backend = candidates[0]
            if self._verbose:
                print(f"BackendManager: selecting backend '{name}' for plan")
            return backend

        # Score candidates using selection_count (higher better) and runtime
        # (lower better).
        fingerprint = PlanFingerprint.from_plan(plan)

        def score(item: Tuple[str, ResamplerBackend]) -> float:
            name = item[0]
            sel = metrics.get_selection_count(name)
            rt = metrics.get_runtime(name, fingerprint)
            # higher selection increases score, lower runtime increases score
            # we invert runtime into a score component; add small epsilon
            # to avoid div-by-zero
            eps = 1e-6
            runtime_score = 1.0 / (rt + eps)
            return sel * 0.7 + runtime_score * 0.3

        scored = sorted(candidates, key=score, reverse=True)
        name, backend = scored[0]
        if self._verbose:
            print(
                f"BackendManager: selecting backend '{name}' for plan (metrics-aware)"
            )
        return backend

    def set_verbose(self, on: bool) -> None:
        self._verbose = bool(on)

    def is_verbose(self) -> bool:
        return self._verbose


# Module-level singleton for the BackendManager
_manager_factory: SingletonFactory[BackendManager] = SingletonFactory(
    lambda: BackendManager()
)


def get_backend_manager(manager: BackendManager | None = None) -> BackendManager:
    """Get the backend manager, optionally providing an override."""
    return _manager_factory.get(manager)


def register_backend(name: str, backend: ResamplerBackend) -> None:
    """Register a backend with the global manager."""
    get_backend_manager().register(name, backend)


def list_backends() -> List[str]:
    """List all registered backends."""
    return get_backend_manager().list_backends()


def get_best_backend(plan: ResamplePlan) -> Optional[ResamplerBackend]:
    """Get the best backend for the given resampling plan."""
    return get_backend_manager().get_best(plan)


def set_backend_verbose(on: bool) -> None:
    """Set backend manager verbosity."""
    get_backend_manager().set_verbose(on)


def is_backend_verbose() -> bool:
    """Check if backend manager is verbose."""
    return get_backend_manager().is_verbose()


__all__ = [
    "BackendManager",
    "get_backend_manager",
    "register_backend",
    "list_backends",
    "get_best_backend",
    "set_backend_verbose",
    "is_backend_verbose",
]
