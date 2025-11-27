"""Entry point for seismic modeling workflow.

This module provides the main entry point for the application. It handles
process cleanup, logging, and tool dispatch.

Usage:
    python -m src [--run-tool TOOL] [--cache-dir DIR] ...

For details on available tools and options, see src.cli module.
"""

# The module intentionally performs call-time (local) imports in `main()`
# to avoid import-time side-effects. These are intentional and safe
# in this codebase.

import atexit
import logging
import multiprocessing
import os
import signal
import sys as _sys
import threading
import time
import warnings
from typing import Any

logger = logging.getLogger(__name__)

# Entry point remains intentionally explicit; keep startup wiring clear


def _terminate_children_on_exit(timeout: float = 1.0) -> None:
    """Attempt to terminate leftover multiprocessing children at exit."""
    try:
        children = multiprocessing.active_children()
        if not children:
            return

        for p in children:
            try:
                p.terminate()
            except (OSError, RuntimeError):
                # Best-effort termination; ignore OS-level errors
                pass

        t0 = time.time()
        while any(p.is_alive() for p in children) and (time.time() - t0) < timeout:
            time.sleep(0.01)

        for p in children:
            try:
                if p.is_alive() and p.pid is not None:
                    os.kill(p.pid, signal.SIGKILL)
            except (OSError, PermissionError):
                # Ignore failures when killing processes (permissions, already gone)
                pass
    except (OSError, RuntimeError):
        # Defensive: ignore OS/runtime errors while attempting cleanup
        pass


# Register cleanup on exit
atexit.register(_terminate_children_on_exit)


def _join_non_daemon_threads(timeout: float = 0.2) -> None:
    """Attempt a short, non-blocking join on non-daemon threads at exit.

    This is defensive: joining only for a short timeout reduces the chance
    that Thread objects are garbage-collected while interpreter globals are
    torn down (which can trigger spurious ``__del__`` errors). We keep the
    join timeout small to avoid delaying shutdown.
    """
    try:
        main = threading.main_thread()
        for t in threading.enumerate():
            if t is main:
                continue
            # Skip threads that are already daemonized (they won't block exit)
            if getattr(t, "daemon", False):
                continue
            try:
                if t.is_alive():
                    t.join(timeout=timeout)
            except Exception:
                # Best-effort: ignore any errors while joining
                logger.debug("Exception while joining thread %s", getattr(t, "name", t))
    except Exception:
        # Defensive: swallow any error during interpreter teardown
        pass


# Try to join non-daemon threads to reduce spurious __del__ errors
atexit.register(_join_non_daemon_threads)

# Filter known resource tracker warnings
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects",
)


def main() -> bool:
    """Main entry point for the application.

    Handles argument parsing, tool dispatch, and modeling pipeline orchestration.

    Returns
    -------
    bool
        True if successful
    """
    # Local imports to avoid import-time side-effects and import cycles.
    # This `main` function is a high-level CLI orchestrator that wires
    # parsing, optional tool dispatch and the modeling pipeline. It is
    # intentionally explicit; suppress a few complexity warnings.

    from src.cli import ParserFactory, modeling

    # (previously disabled import-outside-toplevel here; now removed as redundant)
    # Handle tool forwarding via `--` sentinel
    if "--" in _sys.argv:
        sep = _sys.argv.index("--")
        tool_argv = _sys.argv[sep + 1 :]  # noqa: E203
        parse_argv = _sys.argv[1:sep]  # noqa: E203
    else:
        tool_argv = None
        parse_argv = None

    # Parse arguments
    parser = ParserFactory.modeling_parser()
    args = (
        parser.parse_args(parse_argv) if parse_argv is not None else parser.parse_args()
    )

    # Configure logging
    ParserFactory.configure_logging(getattr(args, "verbose", False))

    # Lazy-register default pluggable implementations (avoid import-time side-effects)
    try:
        # Register default backends lazily to avoid importing heavy backends at
        # module import time.

        from src.processing.resampling.backends._implementations import (
            register_default_backends,
        )

        # (previously disabled import-outside-toplevel here; now removed as redundant)
        register_default_backends()
    except (ImportError, ModuleNotFoundError) as exc:
        # Optional backends may not be available in minimal environments; treat
        # missing optional dependencies as a non-fatal condition and continue
        # startup while logging the cause for diagnostics.
        logger.debug(
            "Default backend registration skipped due to import error: %s", exc
        )

    # Dispatch tool if requested
    # The following parsing block is intentionally nested for permissive
    # parsing of user-provided tool argv; silence the nested-blocks warning
    # for this small, explicit parser.

    if getattr(args, "run_tool", None):
        # Build kwargs from the already-parsed modeling args
        call_kwargs = dict(vars(args))

        # If the user provided tool-specific argv after `--`, merge them into
        # the kwargs. We perform a simple, permissive parse: flags (like
        # `--plot`) become True, `--key value` pairs are converted to the
        # appropriate Python types when possible (int, bool), and keys are
        # normalized to underscore style to match Python parameter names.
        if tool_argv:
            targv = list(tool_argv)
            i = 0
            while i < len(targv):
                tok = targv[i]
                if not tok.startswith("--"):
                    i += 1
                    continue
                key = tok.lstrip("-").replace("-", "_")
                val: Any = True
                # Peek next token to see if it's a value
                if i + 1 < len(targv) and not str(targv[i + 1]).startswith("--"):
                    raw = targv[i + 1]
                    # Convert common types
                    low = raw.lower()
                    if low in ("true", "false"):
                        val = low == "true"
                    else:
                        try:
                            ival = int(raw)
                            val = ival
                        except (ValueError, TypeError):
                            try:
                                fval = float(raw)
                                val = fval
                            except (ValueError, TypeError):
                                val = raw
                    i += 1

                call_kwargs[key] = val
                i += 1

        result = ParserFactory.run_tool(args.run_tool, kwargs=call_kwargs)
        return bool(result) if result is not None else True

    # Run main modeling pipeline
    ParserFactory.maybe_cleanup(args)

    props_depth, _, _, grid_spec = modeling.load_data()
    modeling.run_modeling(props_depth, args, grid_spec)
    modeling.save_results()

    return True


if __name__ == "__main__":
    main()
