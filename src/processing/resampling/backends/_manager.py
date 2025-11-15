"""Simple BackendManager to centralize backend registration and selection.


This is intentionally small and aligned with the existing
`backends` module API. It provides an instance that can be used for
dependency injection when needed.
"""

from __future__ import annotations


from src.processing.resampling.backends._base import ResamplerBackend
from src.processing.resampling._plan import ResamplePlan


class BackendManager:
    """Registry and selector for ResamplerBackend implementations.

    Simple selection heuristic: prefer registered backends in registration
    order and call `supports(plan)` to find the first that supports the plan.
    """

    def __init__(self) -> None:
        self._registry: dict[str, ResamplerBackend] = {}
        self._verbose = False

    def register(self, name: str, backend: ResamplerBackend) -> None:
        if name in self._registry:
            raise KeyError(f"backend '{name}' already registered")
        self._registry[name] = backend

    def list_backends(self) -> list[str]:
        return list(self._registry.keys())

    def get(self, name: str) -> ResamplerBackend | None:
        return self._registry.get(name)

    def get_best(self, plan: ResamplePlan) -> ResamplerBackend | None:
        """Select the best backend for the given plan.

                Returns the first backend that supports the plan, in registration order.

                Parameters
                ----------
                plan : ResamplePlan
                    The resampling plan to find a backend for.

                Returns
                -------
                ResamplerBackend | None
                    The best available backend, or None if no backend supports the plan.

        """
        candidates: list[tuple[str, ResamplerBackend]] = []
        for name, backend in self._registry.items():
            try:
                if backend.supports(plan):
                    candidates.append((name, backend))
            except Exception:
                # backend misbehaved; skip
                continue

        if not candidates:
            return None

        # Return first candidate (preserves order)
        name, backend = candidates[0]
        if self._verbose:
            print(f"BackendManager: selecting backend '{name}' for plan")
        return backend

    def set_verbose(self, on: bool) -> None:
        self._verbose = bool(on)

    def is_verbose(self) -> bool:
        return self._verbose


__all__ = [
    "BackendManager",
]
