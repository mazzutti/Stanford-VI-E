"""Argument parsing and CLI infrastructure for seismic modeling.

This module provides unified argument parsing across the application,
including common arguments, modeling-specific arguments, and tool registry.

Components:
    - ParserFactory: Central parser creation and argument management
    - Tool decorator: Register and dispatch CLI tools
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

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
        import argparse

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
        import argparse

        parser = argparse.ArgumentParser(
            description="Complete seismic forward modeling (AVO)"
        )
        # Add common args
        common = ParserFactory.common_parser(add_help=False)
        for action in common._actions:
            parser._add_action(action)

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
        except Exception:
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
        return parser

    @staticmethod
    def attach_common_args(parser: argparse.ArgumentParser) -> None:
        """Attach the canonical common-args to an existing parser."""
        common = ParserFactory.common_parser(add_help=False)
        for action in common._actions:
            parser._add_action(action)

    @staticmethod
    def get_plot_config(
        args: argparse.Namespace,
    ) -> tuple[str, dict[str, str], Any]:
        """Get plotting configuration - returns (DATA_PATH, FILE_MAP, grid_spec)."""
        from src.io.grid import GridSpec

        DATA_PATH = "."
        FILE_MAP = {"vp": "P-wave Velocity", "facies": "Facies"}
        grid_spec = GridSpec((150, 200, 200), dz=1.0, dt=0.001)

        return DATA_PATH, FILE_MAP, grid_spec

    @staticmethod
    def start_plot_main(
        description: str = "Plotting script",
    ) -> tuple[argparse.Namespace, str, dict[str, str], Any]:
        """Common startup for plotting scripts.

        Returns: (args, DATA_PATH, FILE_MAP, grid_spec)
        """
        import argparse

        parser = argparse.ArgumentParser(description=description)
        ParserFactory.attach_common_args(parser)
        args = parser.parse_args()

        try:
            ParserFactory.configure_logging(getattr(args, "verbose", False))
        except Exception:
            pass

        DATA_PATH, FILE_MAP, grid_spec = ParserFactory.get_plot_config(args)

        return args, DATA_PATH, FILE_MAP, grid_spec

    @staticmethod
    def parse_common_args(argv: list[str] | None = None) -> argparse.Namespace:
        """Parse common arguments."""
        common = ParserFactory.common_parser(add_help=False)
        return common.parse_args(args=argv)

    @staticmethod
    def configure_logging(verbose: bool = False) -> None:
        """Configure Python logging for the process based on verbose flag."""
        import logging as _logging

        level = _logging.DEBUG if verbose else _logging.INFO

        root = _logging.getLogger()
        root.setLevel(level)

        if not root.handlers:
            _logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
        else:
            for h in root.handlers:
                try:
                    h.setLevel(level)
                except Exception:
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
                    _logging.getLogger(name).setLevel(_logging.WARNING)
                except Exception:
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
        from pathlib import Path
        from src.io.pruning import Pruner, PruneStrategy

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
        import warnings as _warnings

        def _register(f: Any) -> Any:
            try:
                cli_names = []
                if name is None:
                    cli_names = [f.__name__]
                else:
                    if isinstance(name, str):
                        cli_names = [name]
                    else:
                        try:
                            cli_names = list(name)
                        except Exception:
                            cli_names = [str(name)]
            except Exception:
                raise TypeError("@ParserFactory.tool must be applied to a function")

            # Normalize names
            norm_names = []
            for cli_name in cli_names:
                if isinstance(cli_name, str):
                    norm_names.append(cli_name.strip())
                else:
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
        if isinstance(tool_name, str):
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

            import inspect

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
            except Exception:
                call_kwargs = full_kwargs

            if call_kwargs:
                return fn(**call_kwargs)
            else:
                return fn()
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(f"Error running tool '{tool_name}': {exc}") from exc


# Convenience alias
tool = ParserFactory.tool
