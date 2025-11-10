"""Entry point for seismic modeling workflow.

This module provides the main entry point for the application. It handles
process cleanup, logging, and tool dispatch.

Usage:
    python -m src [--run-tool TOOL] [--cache-dir DIR] ...

For details on available tools and options, see src.cli module.
"""

import atexit
import logging
import multiprocessing
import os
import signal
import time
import warnings

logger = logging.getLogger(__name__)


def _terminate_children_on_exit(timeout: float = 1.0) -> None:
    """Attempt to terminate leftover multiprocessing children at exit."""
    try:
        children = multiprocessing.active_children()
        if not children:
            return

        for p in children:
            try:
                p.terminate()
            except Exception:
                pass

        t0 = time.time()
        while any(p.is_alive() for p in children) and (time.time() - t0) < timeout:
            time.sleep(0.01)

        for p in children:
            try:
                if p.is_alive() and p.pid is not None:
                    os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        pass


# Register cleanup on exit
atexit.register(_terminate_children_on_exit)

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
    import sys as _sys

    from src.cli import ParserFactory, modeling

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

    # Dispatch tool if requested
    if getattr(args, "run_tool", None):
        result = ParserFactory.run_tool(
            args.run_tool, argv=tool_argv, kwargs=dict(vars(args))
        )
        return bool(result) if result is not None else True

    # Run main modeling pipeline
    ParserFactory.maybe_cleanup(args)

    props_depth, _, _, grid_spec = modeling.load_data()
    modeling.run_modeling(props_depth, args, grid_spec)
    modeling.save_results()

    return True


if __name__ == "__main__":
    main()
