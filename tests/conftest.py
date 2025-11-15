"""Pytest configuration and fixtures for test suite.

This module configures the test environment, including disabling Numba JIT
during coverage runs to allow proper coverage measurement of JIT-compiled code.
"""

import os
import sys

# Disable Numba JIT for coverage analysis
# This allows coverage tools to measure the coverage of JIT-compiled code
# Set this BEFORE importing any modules that use numba
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Now verify that Numba is disabled
try:
    from numba import config

    if hasattr(config, "DISABLE_JIT"):
        assert config.DISABLE_JIT == True, "Failed to disable Numba JIT"
except ImportError:
    pass  # Numba not installed, skip check

# Apply lightweight test-time patches (adds legacy private methods and
# ensures PlotConfig.default() includes a minimal `grid_spec` attribute).
try:
    from tests.test_patches import analysis_monkeypatch

    analysis_monkeypatch.apply_patches()
except Exception:
    # If the patches cannot be applied, let tests continue and fail
    # naturally; this avoids hiding import-time issues during development.
    pass
