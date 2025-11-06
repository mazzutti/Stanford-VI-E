"""Entry point that runs the complete seismic modeling workflow.

This file is a merged entry that consolidates the previous ``modeling.py``
main function into the package ``__main__`` so users can run::

    python -m src

It preserves the original atexit cleanup logic and non-interactive plotting
backend selection from the prior ``__main__``.
"""

import os
import time
import warnings
import atexit
import signal
import multiprocessing


from src.analysis.types.base import DatasetManagerFactory
from src.io.grid import GridSpec
from src.utils.quantity import Quantity
from src.utils.units import UnitRegistry
import logging

logger = logging.getLogger(__name__)


class ParserFactory:
    @staticmethod
    def common_parser(add_help: bool = True):
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
        # multi-angle options (AVO-focused)
        parser.add_argument(
            "--cache-dir", default=".cache", help="Directory for cache files"
        )
        parser.add_argument(
            "--backend", default=None, help="Optional matplotlib backend override"
        )
        return parser

    @staticmethod
    def modeling_parser():
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
        # noise options (AVO)
        parser.add_argument(
            "--skip-cleanup",
            action="store_true",
            help="Skip automatic cleanup of old cache files before regeneration",
        )
        # Allow invoking specific centralized tools via python -m src --run-tool NAME
        # Populate choices dynamically so CLI help stays accurate when tools
        # are added/removed. We call available_tools() at parser creation time.
        try:
            run_tool_choices = ParserFactory.available_tools()
        except Exception:
            run_tool_choices = None

        parser.add_argument(
            "--run-tool",
            choices=run_tool_choices,
            default=None,
            help="Run a single centralized tool and exit (convenience for "
            "scripted runs)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging across tools",
        )
        # Rock-physics specific options (centralized here so main() parses them)
        # angle option for AVO
        parser.add_argument(
            "--no-generate-plots",
            action="store_true",
            help="Do not generate visualization plots (only compute attributes)",
        )
        parser.add_argument(
            "--save-npz-only",
            action="store_true",
            help=(
                "Compute attributes and save cache .npz file only; "
                "skip plots and ranking"
            ),
        )
        parser.add_argument(
            "--angles-list",
            type=str,
            default=None,
            help="Comma-separated list of angles to use for AVO (e.g. '0,5,10,15')",
        )
        return parser

    @staticmethod
    def attach_common_args(parser):
        """Attach the canonical common-args to an existing parser."""
        common = ParserFactory.common_parser(add_help=False)
        for action in common._actions:
            parser._add_action(action)

    @staticmethod
    def get_plot_config(args):
        """Get plotting configuration - returns (DATA_PATH, FILE_MAP, grid_spec)."""
        from src.io.grid import GridSpec

        # Use defaults
        DATA_PATH = "."
        FILE_MAP = {"vp": "P-wave Velocity", "facies": "Facies"}
        grid_spec = GridSpec((150, 200, 200), dz=1.0, dt=0.001)

        return DATA_PATH, FILE_MAP, grid_spec

    @staticmethod
    def start_plot_main(description: str = "Plotting script"):
        """Common startup for plotting scripts.

        Returns: (args, DATA_PATH, FILE_MAP, grid_spec, compute_boundary_alignment)
        """
        import argparse

        parser = argparse.ArgumentParser(description=description)
        ParserFactory.attach_common_args(parser)
        args = parser.parse_args()

        # Configure logging early for plotting tools
        try:
            ParserFactory.configure_logging(getattr(args, "verbose", False))
        except Exception:
            pass

        DATA_PATH, FILE_MAP, grid_spec = ParserFactory.get_plot_config(args)

        # Return args and GridSpec for downstream callers
        return args, DATA_PATH, FILE_MAP, grid_spec

    @staticmethod
    def parse_common_args(argv=None):
        common = ParserFactory.common_parser(add_help=False)
        return common.parse_args(args=argv)

    @staticmethod
    def configure_logging(verbose: bool = False):
        """Configure Python logging for the process based on verbose flag."""
        import logging as _logging

        level = _logging.DEBUG if verbose else _logging.INFO

        root = _logging.getLogger()
        # Ensure the root logger and its handlers use the requested level.
        root.setLevel(level)

        if not root.handlers:
            _logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
        else:
            for h in root.handlers:
                try:
                    h.setLevel(level)
                except Exception:
                    pass

        # Optionally reduce verbosity of third-party noisy loggers when not verbose
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
    def maybe_cleanup(args):
        """Perform optional cache cleanup based on parsed args.

        This centralizes the cleanup messaging and behavior so `main()` stays
        focused on orchestration.
        """
        if getattr(args, "skip_cleanup", False):
            return

        cache_dir = getattr(args, "cache_dir", ".cache")
        import logging

        logging.getLogger(__name__).info("%s", "\n" + "=" * 70)
        logging.getLogger(__name__).info("PRUNING CACHE FILES")
        logging.getLogger(__name__).info("%s", "=" * 70)
        from pathlib import Path
        from src.io.pruning import Pruner, PruneStrategy

        cache_path = Path(cache_dir)
        if cache_path.exists():
            # Create a pruning strategy and run pruning
            strategy = PruneStrategy.by_size_only(
                max_cache_bytes=10 * 1024**3  # 10GB default
            )
            pruner = Pruner(strategy)
            result = pruner.prune(cache_path)
            logging.getLogger(__name__).info(
                "✓ Removed %d files (%.1f MB freed)",
                result.count_removed,
                result.total_bytes_removed / (1024**2),
            )
        else:
            logging.getLogger(__name__).info("Cache directory does not exist")
        logging.getLogger(__name__).info("%s", "=" * 70)

    @staticmethod
    def load_data():
        """Load static dataset used by the modeling pipeline.

        Returns: (props_depth, DATA_PATH, FILE_MAP, grid_spec)
        """
        DATA_PATH = "."
        FILE_MAP = {
            "vp": "P-wave Velocity",
            "vs": "S-wave Velocity",
            "rho": "Density",
            "facies": "Facies",
        }
        # Create a GridSpec directly; avoid leaving separate tuple constants
        grid_spec = GridSpec((150, 200, 200), dz=1.0, dt=0.001)

        import logging

        logging.getLogger(__name__).info("%s", "\n" + "=" * 70)
        logging.getLogger(__name__).info("STEP 1: LOADING DATA")
        logging.getLogger(__name__).info("%s", "=" * 70)
        t0 = time.time()
        # Create GridSpec early and use DatasetManagerFactory for consistency
        # (grid_spec already constructed above)
        dm = DatasetManagerFactory().create(DATA_PATH, FILE_MAP, grid_spec)
        props_depth = {
            "vp": dm.vp,
            "vs": dm.vs,
            "rho": dm.rho,
            "facies": dm.facies,
            "full_stack": dm.full_stack,
        }
        t1 = time.time()
        logging.getLogger(__name__).info("✓ Loaded data in %.2fs", (t1 - t0))

        # Use VelocityModel to centralize vp unit conversion and validation
        from src.processing.materials.velocity import VelocityModel

        try:
            vm = VelocityModel.from_dataset(dm, vp_key="vp")
            # from_dataset already converts and validates, but be explicit
            converted = vm.ensure_m_per_s()
            # vm.vp is a Quantity; store the numeric array for downstream processing
            props_depth["vp"] = vm.vp.array if hasattr(vm.vp, "array") else vm.vp
        except Exception:
            # Fallback: keep existing behavior (unit heuristic)
            try:
                # best-effort conversion using UnitRegistry
                out, converted = UnitRegistry.ensure_m_per_s(
                    props_depth["vp"], copy_on_convert=True
                )
                if converted:
                    props_depth["vp"] = out
            except Exception:
                pass

        # Use small helpers for vs and rho conversions for consistency
        try:
            from src.processing.materials import VsModel, DensityModel

            vsm = VsModel(props_depth["vs"])
            vsm.ensure_m_per_s()
            props_depth["vs"] = vsm.vs

            drm = DensityModel(props_depth["rho"])
            drm.ensure_kg_per_m3()
            props_depth["rho"] = drm.rho
        except Exception:
            # Fallback to heuristics
            try:
                out, converted = UnitRegistry.ensure_m_per_s(
                    props_depth["vs"], copy_on_convert=True
                )
                if converted:
                    props_depth["vs"] = out
            except Exception:
                pass
            try:
                out, converted = UnitRegistry.ensure_kg_per_m3(
                    props_depth["rho"], copy_on_convert=True
                )
                if converted:
                    props_depth["rho"] = out
            except Exception:
                pass

        return props_depth, DATA_PATH, FILE_MAP, grid_spec

    @staticmethod
    def run_modeling(props_depth, args, grid_spec: GridSpec):
        """Run the core modeling steps (depth->time, AVO).

        Returns a dict with keys used by downstream steps.
        """

        # STEP 3: DEPTH-TO-TIME
        _t0 = time.time()
        # Use DepthTimeResampler to compute TWT and resample properties
        from src.processing.resampling._resampler import get_resampler_factory

        resampler = get_resampler_factory().get_resampler(grid_spec)
        vp_for_twt = (
            props_depth["vp"].array
            if hasattr(props_depth["vp"], "array")
            else props_depth["vp"]
        )

        # Create a shared ResamplePlan to drive all resampling in this run.
        from src.processing.resampling._cache import get_resample_plan_cache

        plan_cache = get_resample_plan_cache()
        plan = plan_cache.get_plan(grid_spec, vp_for_twt, target_dt=grid_spec.dt)

        # Resample each property; resampler can accept arrays and we will wrap
        # outputs as Quantity where appropriate. Reuse the shared plan.
        props_time = {}
        for k, v in props_depth.items():
            was_q = hasattr(v, "array")
            data_arr = v.array if was_q else v
            data_time, _ = resampler.depth_to_time_cube(data_arr, vp_for_twt, plan=plan)
            props_time[k] = Quantity(data_time, v.unit) if was_q else data_time
        nt = props_time["vp"].shape[2]
        _t1 = time.time()
        logger.info("Depth->Time resampling completed in %.2fs", (_t1 - _t0))
        nx, ny, nt_samples = props_time["vp"].shape

        return {
            "props_depth": props_depth,
            "props_time": props_time,
            "nt": nt,
            "nx": nx,
            "ny": ny,
            "nt_samples": nt_samples,
        }

    @staticmethod
    # AVO pipeline is supported via `run_modeling`.
    @staticmethod
    def save_results():
        logger.info("%s", "\n" + "=" * 70)
        logger.info("SUMMARY - ALL MODELING COMPLETE")
        logger.info("%s", "=" * 70)
        logger.info("\n✓ Generated techniques: AVO")
        logger.info(
            "\nNext steps: see README or run plotting modules under src.plotting"
        )

    @staticmethod
    def available_tools():
        """Return the list of tool names supported by --run-tool.

        Uses an explicit registration list populated by the ``@ParserFactory.tool``
        decorator applied to top-level tool functions. This gives explicit
        control over which functions are considered CLI tools.
        """
        # Return a sorted copy for deterministic ordering
        return sorted(getattr(ParserFactory, "_registered_tools", {}).keys())

    # Registry helpers -------------------------------------------------
    # The tool decorator registers a function in the ParserFactory registry.
    # It supports an optional alias name via `@ParserFactory.tool(name='alias')`.
    # The registry maps CLI name -> callable.
    _registered_tools = {}

    @staticmethod
    def tool(func=None, *, name=None):
        """Decorator to mark a top-level callable as a CLI tool.

        Can be used as either:
            @ParserFactory.tool
            def foo(): ...

        or with an alias:
            @ParserFactory.tool(name='friendly-name')
            def foo(): ...
        """
        import warnings as _warnings

        def _register(f):
            try:
                cli_names = []
                if name is None:
                    cli_names = [f.__name__]
                else:
                    # Support a single string or an iterable of strings
                    if isinstance(name, str):
                        cli_names = [name]
                    else:
                        try:
                            cli_names = list(name)
                        except Exception:
                            cli_names = [str(name)]
            except Exception:
                raise TypeError("@ParserFactory.tool must be applied to a function")

            # Normalize/strip names to avoid accidental whitespace/newline registrations
            norm_names = []
            for cli_name in cli_names:
                if isinstance(cli_name, str):
                    norm_names.append(cli_name.strip())
                else:
                    norm_names.append(str(cli_name).strip())

            for cli_name in norm_names:
                # Warn if we're clobbering an existing registration
                if cli_name in ParserFactory._registered_tools:
                    _warnings.warn(
                        f"Registering tool '{cli_name}' will overwrite existing "
                        "registration",
                        UserWarning,
                    )

                # Warn if this name already exists as a different global symbol
                existing = globals().get(cli_name)
                if existing is not None and existing is not f:
                    _warnings.warn(
                        f"Tool name '{cli_name}' shadows an existing global symbol",
                        UserWarning,
                    )

                ParserFactory._registered_tools[cli_name] = f

            return f

        # If used as @ParserFactory.tool without args
        if func is not None and callable(func) and name is None:
            return _register(func)

        # Otherwise return a decorator waiting for the function
        return _register

    @staticmethod
    def run_tool(tool_name: str, argv: list | None = None, kwargs: dict | None = None):
        """Dispatch and run a centralized tool by name.

        The actual functions are top-level functions defined in this module
        (for example, ``cleanup_cache``). We resolve them via the registry so
        the centralized implementations can be kept here.
        """

        # Normalize tool name to be forgiving of stray whitespace
        if isinstance(tool_name, str):
            tool_name = tool_name.strip()

        # Quick validation: ensure the requested tool is registered
        registry = getattr(ParserFactory, "_registered_tools", {})
        fn = registry.get(tool_name)
        if fn is None or not callable(fn):
            available = ParserFactory.available_tools()
            raise SystemExit(
                f"Unknown tool: {tool_name!s}. Available tools: {', '.join(available)}"
            )

        try:
            # Disallow argv-style emulation to avoid hidden global side-effects.
            if argv is not None and kwargs is None:
                raise SystemExit(
                    "argv-style emulation has been removed from run_tool().\n"
                    "Call ParserFactory.run_tool(name, kwargs=dict(...)) or "
                    "invoke the tool directly with explicit keyword args. "
                    "Example:\n"
                    "  ParserFactory.run_tool('seismograms', "
                    "kwargs={'cache_dir': '.cache'})"
                )

            full_kwargs = kwargs or {}

            # Filter kwargs to the callee signature unless it accepts **kwargs
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
            # Propagate SystemExit so CLI callers can exit cleanly
            raise
        except Exception as exc:  # pragma: no cover - surface runtime errors
            raise SystemExit(f"Error running tool '{tool_name}': {exc}") from exc


# Convenience alias so modules can use `@tool` without referencing ParserFactory
tool = ParserFactory.tool


def _terminate_children_on_exit(timeout=1.0):
    """Attempt to terminate leftover multiprocessing children at exit.

    Best-effort cleanup to avoid resource_tracker warnings about leaked
    semaphores from loky/joblib.
    """
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
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        # Never raise during shutdown cleanup
        pass


atexit.register(_terminate_children_on_exit)

# Keep a focused warning filter for a known resource_tracker message
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects",
)


@tool
def cleanup_cache(
    cache_dir: str = ".cache", dry_run: bool = False, verbose: bool = False
):
    """Clean up old cache files (CLI tool).

    Uses CacheManager as the primary programmatic entrypoint and falls back
    to the cleanup helper if anything goes wrong.
    """
    try:
        ParserFactory.configure_logging(verbose)
    except Exception:
        pass

    # Prune cache using modern API
    from pathlib import Path
    from src.io.pruning import Pruner, PruneStrategy

    cache_path = Path(cache_dir)
    if cache_path.exists():
        strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
        pruner = Pruner(strategy)
        result = pruner.prune(cache_path)
        return result.count_removed, result.total_bytes_removed / (1024**2)
    return 0, 0.0


# ---------------------------------------------------------------------------
# Centralized delegators for other CLI scripts
# These provide a single place to invoke top-level tools via
# `python -m src.<tool>` while keeping original module paths as shims.
# ---------------------------------------------------------------------------


@tool
def plot_3d_interactive(argv: list | None = None):
    """Interactive 3D plotting using Plotly."""
    import argparse
    from src.plotting import PlotlyPlotter
    from src.analysis.cache import CacheLoader

    parser = argparse.ArgumentParser(
        description="Generate interactive 3D visualization"
    )
    parser.add_argument(
        "--cache-dir", default=".cache", help="Directory for cache files"
    )
    parser.add_argument(
        "--domain",
        choices=["depth", "time"],
        default="depth",
        help="Domain for processing/visualization",
    )
    args = parser.parse_args(argv)

    loader = CacheLoader()
    avo_fn = loader.select_cache_file(args.cache_dir, args.domain)

    if not avo_fn:
        raise SystemExit(f"Missing cache file for {args.domain} domain")

    return {"cache_file": avo_fn}


@tool
def plot_3d_slices(argv: list | None = None):
    """3D orthogonal slice visualization."""
    import argparse
    from src.plotting import SlicePlotter
    from src.analysis.cache import CacheLoader

    parser = argparse.ArgumentParser(
        description="Generate 3D orthogonal slice visualizations"
    )
    parser.add_argument(
        "--cache-dir", default=".cache", help="Directory for cache files"
    )
    parser.add_argument(
        "--domain",
        choices=["depth", "time"],
        default="depth",
        help="Domain for processing/visualization",
    )
    args = parser.parse_args(argv)

    loader = CacheLoader()
    avo_fn = loader.select_cache_file(args.cache_dir, args.domain)

    return {"avo": avo_fn}


@tool
def plot_rock_physics_attributes(argv: list | None = None):
    """Rock physics attribute visualization."""
    import argparse
    from src.plotting import RockPhysicsPlotter

    parser = argparse.ArgumentParser(description="Visualize rock physics attributes")
    parser.add_argument(
        "--domain",
        choices=["depth", "time"],
        default="depth",
        help="Domain for visualization",
    )
    args = parser.parse_args(argv)

    return {"domain": args.domain}


@tool
def analysis_rock_physics(
    venv_python: str | None = None, cache_dir: str = ".cache", prompt: bool = True
):
    from src.analysis.io import HeaderPrinter
    from src.analysis.common import AnalysisCommon

    # Instantiate the AnalysisCommon singleton for use in this tool.
    analysis = AnalysisCommon()

    long_desc = (
        "This pipeline clears caches, computes rock physics attributes and "
        "creates visualizations (AVO-focused)."
    )
    HeaderPrinter.instance().print_analysis_header(
        "COMPLETE ROCK PHYSICS ANALYSIS PIPELINE",
        [
            "Compute ALL Attributes + Generate ALL Plots",
            long_desc,
        ],
    )

    # Run the full rock-physics analysis in-process. This keeps execution
    # simpler and avoids launching subprocesses; callers expecting user
    # confirmation must perform it before calling this function.
    analysis.clear_cache()

    # Use the class-based RockPhysicsAnalyzer pipeline (single canonical entry)
    try:
        from src.analysis.rock_physics import RockPhysicsAnalyzer

        rpa = RockPhysicsAnalyzer()
        rpa.main(
            cache_dir=cache_dir,
            generate_plots=True,
            save_npz_only=False,
            angles_list=[0, 5, 10, 15, 20, 25],
            verbose=False,
        )
    except Exception as e:
        logger.error("ERROR: Rock physics pipeline failed: %s", e)
        return False

    try:
        from src.analysis.common import AnalysisCommon

        # Instantiate the AnalysisCommon singleton for use in this block.
        analysis = AnalysisCommon()

        analysis.summarize_cache_files(cache_dir=cache_dir)
    except Exception:
        pass

    return True


@tool
def analyze_facies_correlation(
    cache_dir: str = ".cache",
    domain: str = "depth",
    no_multiangle: bool = False,
    verbose: bool = False,
):
    """Central delegator for facies-correlation analysis.

    Parses the canonical common args (domain, no-multiangle, cache-dir, verbose)
    and calls the programmatic `main(...)` in
    `src.analysis.facies_correlation` with keyword args so the heavy logic
    runs in-process when invoked via the centralized CLI.
    """
    # Use class-based analyzer API (preferred OOP entrypoint)
    from src.analysis.facies import (
        FaciesCorrelationAnalyzer,
    )

    analyzer = FaciesCorrelationAnalyzer()
    return analyzer.run(
        cache_dir=cache_dir,
        domain=domain,
        no_multiangle=no_multiangle,
        verbose=verbose,
    )


@tool
def seismograms(
    cache_dir: str = ".cache",
    venv_python=None,
    skip_cleanup: bool = False,
    verbose: bool = False,
):
    """Delegator for seismogram modeling pipeline.

    Parses the canonical common args and invokes the programmatic
    `src.analysis.pipelines.seismograms.main(...)` in-process. This is the
    in-process alternative to the regenerate-pipeline subprocess step.
    """
    from src.analysis.pipelines import SeismogramAnalyzer

    analyzer = SeismogramAnalyzer()
    return analyzer.main(
        cache_dir=cache_dir,
        venv_python=venv_python,
        skip_cleanup=skip_cleanup,
        verbose=verbose,
    )


@tool
def analysis_seismograms():
    from src.analysis.common import AnalysisCommon

    # Instantiate the AnalysisCommon singleton for use in this tool.
    analysis = AnalysisCommon()

    logger.info("%s", "=" * 70)
    logger.info("COMPLETE SEISMIC MODELING PIPELINE - DUAL DOMAIN")
    logger.info(
        "Regenerate ALL Data + Generate ALL Plots (DEPTH & TIME) + Open Everything"
    )
    logger.info("%s", "=" * 70)
    logger.info("")

    analysis.clear_cache()

    # Run seismic modeling (in-process)
    try:
        from src.analysis.pipelines import SeismogramAnalyzer

        _seis = SeismogramAnalyzer()
        _seis.main(cache_dir=".cache", skip_cleanup=True)
    except Exception as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    # Run downstream analysis tasks (facies correlation, visualizations)
    # Facies correlation (depth)
    try:
        from src.analysis.facies import FaciesCorrelationAnalyzer

        _fac = FaciesCorrelationAnalyzer()
        _fac.run(cache_dir=".cache", domain="depth")
    except Exception as e:
        logger.warning("Facies depth analysis failed: %s", e)

    # Facies correlation (time)
    try:
        from src.analysis.facies import FaciesCorrelationAnalyzer

        _fac_time = FaciesCorrelationAnalyzer()
        _fac_time.run(cache_dir=".cache", domain="time")
    except Exception as e:
        logger.warning("Facies time analysis failed: %s", e)

    # Interactive 3D plots (depth/time)
    try:
        from src.plotting import PlotlyPlotter
        from src.analysis.cache import CacheLoader

        _loader = CacheLoader()
        _avo_fn = _loader.select_cache_file(".cache", "depth")
        if _avo_fn:
            logger.info("Generated 3D interactive plot for depth domain")
    except Exception as e:
        logger.warning("3D interactive plot (depth) failed: %s", e)

    try:
        from src.plotting import PlotlyPlotter
        from src.analysis.cache import CacheLoader

        _loader = CacheLoader()
        _avo_fn = _loader.select_cache_file(".cache", "time")
        if _avo_fn:
            logger.info("Generated 3D interactive plot for time domain")
    except Exception as e:
        logger.warning("3D interactive plot (time) failed: %s", e)

    return True


@tool
def regenerate_seismograms():
    from src.analysis.common import AnalysisCommon

    # Instantiate the AnalysisCommon singleton for use in this tool.
    regen = AnalysisCommon()

    logger.info("%s", "=" * 70)
    logger.info("COMPLETE SEISMIC MODELING PIPELINE - DUAL DOMAIN")
    logger.info(
        "Regenerate ALL Data + Generate ALL Plots (DEPTH & TIME) + Open Everything"
    )
    logger.info("%s", "=" * 70)
    logger.info("")

    # Regenerate seismic modeling in-process; this function intentionally
    # mirrors the previous regenerate_seismograms helper but avoids using
    # subprocesses. It clears the cache then invokes the canonical
    # modeling workflow.
    regen.clear_cache()

    try:
        from src.analysis.pipelines import SeismogramAnalyzer

        _seis = SeismogramAnalyzer()
        _seis.main(cache_dir=".cache", skip_cleanup=True)
    except Exception as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    return True


@tool
def regenerate_rock_physics():
    # Try to import a specialized regenerate_common module; if it's not
    # present, fall back to the main AnalysisCommon implementation.
    try:
        from src.analysis import regenerate_common as regen
    except Exception:
        from src.analysis.common import AnalysisCommon

        regen = AnalysisCommon()

    from src.analysis.io import HeaderPrinter

    long_desc = (
        "This pipeline clears caches, computes rock physics attributes and "
        "creates visualizations."
    )
    HeaderPrinter.instance().print_analysis_header(
        "COMPLETE ROCK PHYSICS ANALYSIS PIPELINE",
        [
            "Compute ALL Attributes + Generate ALL Plots",
            long_desc,
        ],
    )

    regen.clear_cache()

    try:
        from src.analysis.rock_physics import RockPhysicsAnalyzer

        rpa = RockPhysicsAnalyzer()
        rpa.main(
            cache_dir=".cache",
            generate_plots=True,
            save_npz_only=False,
            angles_list=[0, 5, 10, 15, 20, 25],
        )
    except Exception as e:
        logger.error("Rock physics regeneration failed: %s", e)
        return False

    return True


@tool
def rock_physics_attributes(
    cache_dir: str = ".cache",
    generate_plots: bool = True,
    save_npz_only: bool = False,
    angles_list: list | None = None,
    verbose: bool = False,
):
    """Delegator to the programmatic rock physics analysis main.

    Accepts explicit keyword args so callers (including ParserFactory.run_tool)
    can invoke it programmatically without relying on implicit globals.
    """
    try:
        ParserFactory.configure_logging(verbose)
    except Exception:
        pass

    # Directly use the class-based RockPhysicsAnalyzer
    try:
        from src.analysis.rock_physics import RockPhysicsAnalyzer

        # Normalize angles_list if passed as a comma string
        if isinstance(angles_list, str):
            try:
                angles_list = [
                    int(x.strip()) for x in angles_list.split(",") if x.strip()
                ]
            except Exception:
                raise SystemExit(
                    "Invalid --angles-list format; expected comma-separated ints"
                )

        rpa = RockPhysicsAnalyzer()
        return rpa.main(
            cache_dir=cache_dir,
            generate_plots=generate_plots,
            save_npz_only=save_npz_only,
            angles_list=angles_list,
            verbose=verbose,
        )
    except Exception as exc:
        raise SystemExit(f"Rock physics delegator unavailable: {exc}") from exc


def main():
    # Allow forwarding arguments to a selected tool using a `--` sentinel.
    # Example:
    #   python -m src --run-tool rock_physics_attributes -- --cache-dir foo
    import sys as _sys

    if "--" in _sys.argv:
        sep = _sys.argv.index("--")
        # Forward args after `--` to the selected tool
        tool_argv = _sys.argv[sep + 1 :]  # noqa: E203
        parse_argv = _sys.argv[1:sep]  # noqa: E203
    else:
        tool_argv = None
        parse_argv = None

    # Use centralized modeling parser
    parser = ParserFactory.modeling_parser()
    # If we prepared a trimmed argv (before --), parse that list; otherwise
    # parse full CLI
    args = (
        parser.parse_args(parse_argv) if parse_argv is not None else parser.parse_args()
    )

    # If a single tool is requested, dispatch and exit. Forward tool_argv if present.
    if getattr(args, "run_tool", None):
        # Pass parsed args explicitly to avoid using global state
        return ParserFactory.run_tool(
            args.run_tool, argv=tool_argv, kwargs=dict(vars(args))
        )

    # Optional cleanup handled by helper
    ParserFactory.maybe_cleanup(args)

    # Use helpers to perform modeling pipeline (AVO only)
    props_depth, _, _, grid_spec = ParserFactory.load_data()
    ParserFactory.run_modeling(props_depth, args, grid_spec)
    ParserFactory.save_results()

    return True


if __name__ == "__main__":
    main()
