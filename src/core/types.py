"""Small, low-dependency shared types for `src.core`.

This module should remain lightweight: only typing and small Protocols.
Callers that need concrete implementations should import their specific
submodules (e.g. `src.core.processors`, `src.core.analyzers`).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class Buildable(Protocol):
    """Protocol for objects that can be built by a builder/factory.

    Keep this tiny and free of heavy imports so it can be used across the
    codebase without pulling in large modules at import time.
    """

    def build(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial
        """Build and return the configured object.

        Implementations should construct or assemble the final object
        and return it. The method intentionally keeps broad `*args`/kwargs
        to allow flexible builder patterns across the codebase.
        """
        raise NotImplementedError()

# Lightweight alias type for validation result containers used across modules.
ValidatorResult = dict

__all__ = ["Buildable", "ValidatorResult"]
