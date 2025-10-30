#!/usr/bin/env python
"""Seismic modeling pipeline helpers.

This module exposes a small programmatic `main()` used by the package-level
CLI. The orchestration is implemented in `src.modeling.api.run_full_modeling`
to keep this file compact and easy to test.
"""

from .common import time, Path

from . import common as analysis

import logging
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)


__all__ = [
    "SeismogramAnalyzer",
    "seismogram_analyzer",
    "get_seismogram_analyzer",
]


# Object-oriented facade for seismogram utilities
class SeismogramAnalyzer:
    def run_command(self, cmd, description, prefix: str = ""):
        return _impl_run_command(cmd, description, prefix=prefix)

    def clear_cache(self, patterns=None, prefix: str = ""):
        return _impl_clear_cache(patterns=patterns, prefix=prefix)

    def check_file_exists(self, filepath, description):
        return _impl_check_file_exists(filepath, description)

    def open_file(self, filepath, description, prefix: str = ""):
        return _impl_open_file(filepath, description, prefix=prefix)

    def main(self, *args, **kwargs):
        return _impl_main(*args, **kwargs)


# Module-level lazy proxy for the seismogram facade
seismogram_analyzer = LazyObjectProxy(lambda: SeismogramAnalyzer())


# Prefer the OO facade and module-level lazy proxy. Callers should use
# `seismogram_analyzer` or request an instance via `get_seismogram_analyzer()`.
__all__.extend(["SeismogramAnalyzer", "seismogram_analyzer", "get_seismogram_analyzer"])


def get_seismogram_analyzer(
    instance: SeismogramAnalyzer | None = None,
) -> "SeismogramAnalyzer":
    """Return provided SeismogramAnalyzer or module-level lazy singleton."""
    return _impl_get_seismogram_analyzer(instance)


def _impl_get_seismogram_analyzer(
    instance: SeismogramAnalyzer | None = None,
) -> "SeismogramAnalyzer":
    return instance if instance is not None else seismogram_analyzer


# Thin procedural wrappers have been removed in favor of using the
# `seismogram_analyzer` proxy or requesting an instance via
# `get_seismogram_analyzer()`. The canonical implementations are the
# `_impl_*` functions and the SeismogramAnalyzer facade methods.


def _impl_run_command(cmd, description, prefix: str = ""):
    # Run a shell command with simple timing and error reporting. We avoid
    # importing centralized subprocess helpers to keep the analysis modules
    # free of subprocess orchestration dependencies.
    import subprocess
    import time

    if description:
        logger.info("%s%s", prefix, description)
    t0 = time.time()
    try:
        res = subprocess.run(cmd, shell=True, check=True)
        logger.info("%s✓ Completed in %.1f seconds", prefix, time.time() - t0)
        return res
    except Exception as e:
        logger.error("%s✗ Error running command: %s", prefix, e)
        return None


def _impl_clear_cache(patterns=None, prefix: str = ""):
    if patterns is None:
        patterns = ["avo_*.npz", "ai_*.npz", "ei_*.npz"]
    return analysis.clear_cache(patterns=patterns, prefix=prefix)


def _impl_check_file_exists(filepath, description):
    path = Path(filepath)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("\u2713 Found: %s", description)
        logger.info("  Path: %s", filepath)
        logger.info("  Size: %.1f MB", size_mb)
        return True
    else:
        logger.error("\u2717 Missing: %s", description)
        logger.error("  Expected: %s", filepath)
        return False


def _impl_open_file(filepath, description, prefix: str = ""):
    ok = analysis.open_file(filepath, description=description, prefix=prefix)
    if ok:
        try:
            time.sleep(1)
        except Exception:
            pass
    return ok


def _impl_main(
    *,
    cache_dir: str = ".cache",
    venv_python: str | None = None,
    skip_cleanup: bool = False,
    verbose: bool = False,
):
    """Programmatic entrypoint for the seismogram/regenerate pipeline.

    This function is intentionally small: it delegates the full orchestration
    to `src.modeling.api.run_full_modeling` so callers (including the package
    `__main__`) can invoke the pipeline without duplication.
    """
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    if not skip_cleanup:
        _impl_clear_cache(prefix="    ")

    try:
        from src.modeling.api import run_full_modeling

        run_full_modeling(
            cache_dir=cache_dir,
            skip_cleanup=skip_cleanup,
            verbose=verbose,
            add_avo_noise=False,
            add_ei_noise=False,
        )
        return True
    except Exception as e:
        raise RuntimeError(f"Seismic modeling failed (in-process): {e}") from e
