"""Argument parsing and CLI infrastructure for seismic modeling.

This module provides unified argument parsing across the application,
including common arguments, modeling-specific arguments, and tool registry.

Components:
    - ParserFactory: Central parser creation and argument management
    - Tool decorator: Register and dispatch CLI tools
"""

from __future__ import annotations

import argparse
import inspect
import logging
import warnings as _warnings
from pathlib import Path
from typing import Any

from src.io.grid import GridSpec

logger = logging.getLogger(__name__)

# Some CLI helper functions perform imports at call-time to avoid heavy
# imports and import cycles during module import. Keep lazy imports and
# prefer per-import suppression (added where imports are performed).

__all__ = ["ParserFactory", "tool"]

class ParserFactory:
    """Factory for creating and managing argument parsers."""

    _registered_tools: dict[str, Any] = {}

    @staticmethod
    def common_parser(add_help: bool = True) -> argparse.ArgumentParser:
        """Return the shared argparse parser used across plotting and tools.

        This mirrors the original common_parser contract used elsewhere in the
        project and provides a small set of options used by many scripts.
        """
        parser = argparse.ArgumentParser(add_help=add_help)
        parser.add_argument(
            "--domain",
            choices=["depth", "time"],
            default="depth",
            help="Domain for processing/visualization (default: depth)",
        )
        parser.add_argument(
            "--cache-dir", default=".cache", help="Directory for cache files"
        )
        parser.add_argument(
            "--backend", default=None, help="Optional matplotlib backend override"
        )
        return parser

    @staticmethod
    def modeling_parser() -> argparse.ArgumentParser:
        """Return the parser for the main modeling workflow."""
        # Use argparse `parents` to inherit the common arguments via the
        # public API instead of accessing argparse internals.
        common = ParserFactory.common_parser(add_help=False)
        parser = argparse.ArgumentParser(
            description="Complete seismic forward modeling (AVO)",
            parents=[common],
        )

        parser.add_argument(
            "--add-avo-noise",
            action="store_true",
            help="Add angle-dependent noise to AVO seismograms (SNR=20dB)",
        )
        parser.add_argument(
            "--skip-cleanup",
            action="store_true",
            help="Skip automatic cleanup of old cache files before regeneration",
        )
        try:
            run_tool_choices = ParserFactory.available_tools()
        except (RuntimeError, OSError):
            run_tool_choices = None

        parser.add_argument(
            "--run-tool",
            choices=run_tool_choices,
            default=None,
            help="Run a single centralized tool and exit (convenience for scripted runs)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging across tools",
        )
        parser.add_argument(
            "--no-generate-plots",
            action="store_true",
            help="Do not generate visualization plots (only compute attributes)",
        )
        parser.add_argument(
            "--save-npz-only",
            action="store_true",
            help="Compute attributes and save cache .npz file only; skip plots and ranking",
        )
        parser.add_argument(
            "--angles-list",
            type=str,
            default=None,
            help="Comma-separated list of angles to use for AVO (e.g. '0,5,10,15')",
        )
        parser.add_argument(
            "--plot-type",
            choices=["2d", "3d"],
            default="2d",
            help=(
                "Type of plot for visualization tools: '2d' for PNG (matplotlib) "
                "or '3d' for HTML (Plotly)"
            ),
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="docs/images",
            help="Output directory for generated plots and visualizations",
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            default=".",
            help="Root directory containing data files (for data loading tools)",
        )
        return parser

    @staticmethod
    def attach_common_args(parser: argparse.ArgumentParser) -> None:
        """Attach the canonical common-args to an existing parser."""
        # Add the small set of common arguments explicitly via the public
        # `add_argument` API instead of touching argparse internals.
        parser.add_argument(
            "--domain",
            choices=["depth", "time"],
            default="depth",
            help="Domain for processing/visualization (default: depth)",
        )
        parser.add_argument(
            "--cache-dir",
            default=".cache",
            help="Directory for cache files",
        )
        parser.add_argument(
            "--backend",
            default=None,
            help="Optional matplotlib backend override",
        )

    @staticmethod
    def get_plot_config(
        args: argparse.Namespace,
    ) -> tuple[str, dict[str, str], Any]:
        # `args` parameter exists for compatibility with higher-level callers
        # and future extensions; silence unused-argument lint here.

        """Get plotting configuration - returns (data_path, file_map, grid_spec)."""

        data_path = "."
        file_map = {"vp": "P-wave Velocity", "facies": "Facies"}
        grid_spec = GridSpec((150, 200, 200), dz=1.0, dt=0.001)

        return data_path, file_map, grid_spec

    @staticmethod
    def start_plot_main(
        description: str = "Plotting script",
    ) -> tuple[argparse.Namespace, str, dict[str, str], Any]:
        """Common startup for plotting scripts.

        Returns: (args, data_path, file_map, grid_spec)
        """
        parser = argparse.ArgumentParser(description=description)
        ParserFactory.attach_common_args(parser)
        args = parser.parse_args()

        try:
            ParserFactory.configure_logging(getattr(args, "verbose", False))
        except (RuntimeError, OSError):
            pass

        data_path, file_map, grid_spec = ParserFactory.get_plot_config(args)

        return args, data_path, file_map, grid_spec

    @staticmethod
    def parse_common_args(argv: list[str] | None = None) -> argparse.Namespace:
        """Parse common arguments."""
        common = ParserFactory.common_parser(add_help=False)
        return common.parse_args(args=argv)

    @staticmethod
    def configure_logging(verbose: bool = False) -> None:
        """Configure Python logging for the process based on verbose flag."""

        level = logging.DEBUG if verbose else logging.INFO

        root = logging.getLogger()
        root.setLevel(level)

        if not root.handlers:
            logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
        else:
            for h in root.handlers:
                try:
                    h.setLevel(level)
                except (AttributeError, TypeError):
                    pass

        # Reduce verbosity of noisy loggers when not verbose
        if not verbose:
            for name in (
                "matplotlib",
                "matplotlib.font_manager",
                "numba",
                "matplotlib.pyplot",
            ):
                try:
                    logging.getLogger(name).setLevel(logging.WARNING)
                except (AttributeError, TypeError):
                    pass

    @staticmethod
    def maybe_cleanup(args: argparse.Namespace) -> None:
        """Perform optional cache cleanup based on parsed args."""
        if getattr(args, "skip_cleanup", False):
            return

        cache_dir = getattr(args, "cache_dir", ".cache")
        logger.info("%s", "\n" + "=" * 70)
        logger.info("PRUNING CACHE FILES")
        logger.info("%s", "=" * 70)
        from src.io.pruning import (
            Pruner,
            PruneStrategy,
        )

        cache_path = Path(cache_dir)
        if cache_path.exists():
            strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
            pruner = Pruner(strategy)
            result = pruner.prune(cache_path)
            logger.info(
                "✓ Removed %d files (%.1f MB freed)",
                result.count,
                result.bytes_freed / (1024**2),
            )
        else:
            logger.info("Cache directory does not exist")
        logger.info("%s", "=" * 70)

    @staticmethod
    def available_tools() -> list[str]:
        """Return the list of tool names supported by --run-tool."""
        return sorted(getattr(ParserFactory, "_registered_tools", {}).keys())

    @staticmethod
    def tool(func: Any = None, *, name: str | list[str] | None = None) -> Any:
        """Decorator to mark a callable as a CLI tool.

        Usage:
            @ParserFactory.tool
            def my_tool(): ...

            @ParserFactory.tool(name='alias')
            def my_tool(): ...
        """

        def _register(f: Any) -> Any:
            try:
                cli_names: list[str] = []
                if name is None:
                    cli_names = [f.__name__]
                else:
                    if isinstance(name, str):
                        cli_names = [name]
                    else:
                        try:
                            cli_names = list(name)
                        except (TypeError, ValueError):
                            cli_names = [str(name)]
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "@ParserFactory.tool must be applied to a function"
                ) from exc

            # Normalize names
            norm_names: list[str] = []
            for cli_name in cli_names:
                # Coerce to str and strip whitespace (covers str and non-str inputs)
                norm_names.append(str(cli_name).strip())

            for cli_name in norm_names:
                if cli_name in ParserFactory._registered_tools:
                    _warnings.warn(
                        f"Registering tool '{cli_name}' will overwrite existing registration",
                        UserWarning,
                    )

                existing = globals().get(cli_name)
                if existing is not None and existing is not f:
                    _warnings.warn(
                        f"Tool name '{cli_name}' shadows an existing global symbol",
                        UserWarning,
                    )

                ParserFactory._registered_tools[cli_name] = f

            return f

        if func is not None and callable(func) and name is None:
            return _register(func)

        return _register

    @staticmethod
    def run_tool(
        tool_name: str,
        argv: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch and run a centralized tool by name."""
        tool_name = tool_name.strip()

        registry = getattr(ParserFactory, "_registered_tools", {})
        fn = registry.get(tool_name)
        if fn is None or not callable(fn):
            available = ParserFactory.available_tools()
            raise SystemExit(
                f"Unknown tool: {tool_name!s}. Available tools: {', '.join(available)}"
            )

        try:
            if argv is not None and kwargs is None:
                raise SystemExit(
                    "argv-style emulation has been removed from run_tool().\n"
                    "Call ParserFactory.run_tool(name, kwargs=dict(...)) or "
                    "invoke the tool directly with explicit keyword args."
                )

            full_kwargs = kwargs or {}

            try:
                sig = inspect.signature(fn)
                parameters = sig.parameters
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
                )
                if accepts_var_kw:
                    call_kwargs = full_kwargs
                else:
                    allowed = set(parameters.keys())
                    call_kwargs = {k: v for k, v in full_kwargs.items() if k in allowed}
            except (TypeError, ValueError):
                call_kwargs = full_kwargs

            if call_kwargs:
                return fn(**call_kwargs)
            return fn()
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            raise SystemExit(f"Error running tool '{tool_name}': {exc}") from exc

# Convenience alias
tool = ParserFactory.tool
