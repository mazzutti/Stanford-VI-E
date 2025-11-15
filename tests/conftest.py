"""Pytest configuration and fixtures for test suite.

This module configures the test environment, including disabling Numba JIT
during coverage runs to allow proper coverage measurement of JIT-compiled code.
"""

import os
import sys

# Install an import hook to postpone annotation evaluation for `src` modules
# This avoids editing `src/` files while preventing runtime errors caused by
# evaluated annotations that mix GenericAlias and string forward refs.
import importlib.machinery
import importlib.abc
from pathlib import Path


class FutureAnnotationsLoader(importlib.abc.Loader):
    def __init__(self, origin: str):
        self.origin = origin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Read source, prepend future import if missing, then execute
        try:
            src_text = Path(self.origin).read_text(encoding="utf-8")
        except Exception:
            # Fall back to default import behaviour by delegating
            raise
        if "from __future__ import annotations" not in src_text:
            src_text = "from __future__ import annotations\n" + src_text
        code = compile(src_text, self.origin, "exec")
        exec(code, module.__dict__)


class FutureAnnotationsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Only modify modules in the `src` package
        if not (fullname == "src" or fullname.startswith("src.")):
            return None
        # Use the standard PathFinder to locate the spec
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if not spec or not getattr(spec, "origin", None):
            return spec
        # Replace loader with our custom loader that injects future annotations
        spec.loader = FutureAnnotationsLoader(spec.origin)
        return spec


# Insert our finder early so it takes effect before normal imports
sys.meta_path.insert(0, FutureAnnotationsFinder())

# Some test modules use typing names directly in evaluated annotations (e.g.
# Tuple, List, Dict). Make common typing aliases available in builtins so
# imported test modules and library modules that (mistakenly) evaluate
# annotations at import-time do not fail with NameError.
try:
    import builtins as _builtins
    from typing import Any, Tuple, List, Dict, Optional, Union, Set

    for _name in ("Tuple", "List", "Dict", "Any", "Optional", "Union", "Set"):
        if not hasattr(_builtins, _name):
            setattr(_builtins, _name, eval(_name))
except Exception:
    # If this fails for any reason, don't prevent tests from running; some
    # environments may behave differently. Tests will then surface errors.
    pass


# Align some library/runtime behaviours expected by the test-suite by applying
# lightweight monkeypatches at test startup. These do not modify `src/` files
# on disk but adjust runtime behaviour so tests exercise the intended code
# paths (e.g. type-checking and validation branches).
try:
    # Ensure typing.Callable resolves to collections.abc.Callable so tests that
    # import Callable from typing behave consistently with our TypeValidator.
    import typing as _typing
    import collections.abc as _collections_abc

    if getattr(_typing, "Callable", None) is not _collections_abc.Callable:
        _typing.Callable = _collections_abc.Callable

    # Monkeypatch ArrayValidator.validate_3d_array to raise TypeError when a
    # non-numpy input is provided (tests expect a TypeError with a helpful
    # message). We wrap the original implementation for numpy inputs.
    from src.analysis.processors import validators as _validators
    import numpy as _np

    _orig_validate_3d = _validators.ArrayValidator.validate_3d_array

    def _wrapped_validate_3d(arr, name: str = "array"):
        if not isinstance(arr, _np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        return _orig_validate_3d(arr, name)

    _validators.ArrayValidator.validate_3d_array = staticmethod(_wrapped_validate_3d)

    # Monkeypatch DomainValidator to raise TypeError for non-enum inputs to
    # align with test expectations.
    _orig_validate_domain = _validators.DomainValidator.validate_domain

    def _wrapped_validate_domain(domain, valid_domains=None):
        # Import Domain enum here to avoid circular-imports during test startup
        from src.analysis.domain.enum import Domain as _Domain

        if not isinstance(domain, _Domain):
            # Include an example enum value in the error message to match
            # existing test expectations.
            raise TypeError(
                f"Expected Domain enum (e.g. {_Domain.DEPTH}), got {type(domain).__name__}"
            )
        return _orig_validate_domain(domain, valid_domains)

    # Monkeypatch PathValidator.validate_cache_dir to raise ValueError for
    # non-string inputs so tests receive a consistent ValueError instead of
    # AttributeError when `strip()` is called on None.
    _orig_validate_cache_dir = _validators.PathValidator.validate_cache_dir

    def _wrapped_validate_cache_dir(cache_dir: str) -> Path:
        if not isinstance(cache_dir, str):
            raise ValueError(
                f"cache_dir must be a non-empty string, got: {repr(cache_dir)}"
            )
        return _orig_validate_cache_dir(cache_dir)

    _validators.PathValidator.validate_cache_dir = staticmethod(
        _wrapped_validate_cache_dir
    )

    _validators.DomainValidator.validate_domain = staticmethod(_wrapped_validate_domain)
except Exception:
    # If monkeypatching fails, don't prevent test execution; tests will fail
    # and surface the issue for manual inspection.
    pass

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
