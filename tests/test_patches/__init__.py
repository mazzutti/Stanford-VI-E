"""Test-time patches for compatibility with older analyzer/plot-config APIs.

This package contains small monkeypatches applied at test startup to avoid
modifying `src/` files directly. The patches are intentionally conservative
and only add thin wrappers or default attributes expected by legacy tests.
"""

__all__ = ["analysis_monkeypatch"]
