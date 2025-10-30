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

import numpy as np

from src.io import data_loader
from src.io.grid import GridSpec
from src.modeling import modeling as modeling_utils
from src.signal import wavelets
from src.utils.quantity import Quantity
from src.utils.units import UnitRegistry
import hashlib
import logging

logger = logging.getLogger(__name__)


__all__ = ["ParserFactory", "main"]


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
        parser.add_argument(
            "--no-multiangle",
            action="store_true",
            help="Disable multi-angle EI processing and use single-angle fallback",
        )
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
            description="Complete seismic forward modeling (AVO + AI + EI)"
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
            "--no-ei-noise",
            action="store_true",
            help=(
                "Disable frequency-dependent noise for EI seismogram "
                "(noise is ON by default)"
            ),
        )
        parser.add_argument(
            "--ei-noise-snr",
            type=float,
            default=None,
            help="Target SNR for EI noise in dB",
        )
        parser.add_argument(
            "--ei-noise-seed",
            type=int,
            default=None,
            help="Random seed for reproducible EI noise",
        )
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
        parser.add_argument(
            "--ei-angle",
            type=int,
            default=10,
            help="Single-angle EI (degrees) to treat as the optimal EI (default: 10)",
        )
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
            help="Comma-separated list of EI angles to compute (e.g. '0,5,10,15')",
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
        """Normalize plotting args and return (DATA_PATH, FILE_MAP, grid_spec)."""
        from src.plotting.helpers.plot import prepare_plotting_args, default_plot_config

        prepare_plotting_args(args)
        plot_cfg = default_plot_config()
        gs = plot_cfg.grid_spec
        # Return the canonical GridSpec (avoid returning separate tuple constants)
        return plot_cfg.data_path, plot_cfg.file_map, gs

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

        from src.plotting.helpers.plot import compute_boundary_alignment

        # Return args and GridSpec for downstream callers
        return args, DATA_PATH, FILE_MAP, grid_spec, compute_boundary_alignment

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
        logging.getLogger(__name__).info("CLEANING UP OLD CACHE FILES")
        logging.getLogger(__name__).info("%s", "=" * 70)
        from src.io.cache import cache_for_dir

        # prefer explicit cache_for_dir to obtain a CacheManager for the requested dir
        removed, size_mb = cache_for_dir(cache_dir).cleanup_old_cache(dry_run=False)
        if removed > 0:
            logging.getLogger(__name__).info(
                "✓ Removed %d old files (%.1f MB freed)", removed, size_mb
            )
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
        # Create GridSpec early and prefer DatasetManager for richer API
        # (grid_spec already constructed above)
        dm = data_loader.DatasetManager.from_stanfordsix(DATA_PATH, FILE_MAP, grid_spec)
        props_depth = dm.data
        t1 = time.time()
        logging.getLogger(__name__).info("✓ Loaded data in %.2fs", (t1 - t0))

        # Use VelocityModel to centralize vp unit conversion and validation
        from src.processing.velocity import VelocityModel

        try:
            vm = VelocityModel.from_dataset(dm, vp_key="vp")
            # from_dataset already converts and validates, but be explicit
            converted = vm.ensure_m_per_s()
            # vm.vp is a Quantity; store the numeric array for backward compat
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
        """Run the core modeling steps (multi-angle EI, weighted product,
        depth->time, AVO, AI).

        Returns a dict with keys used by downstream steps.
        """
        # STEP 2: MULTI-ANGLE EI (depth domain)
        # rock_physics_attributes lives under src.analysis; import explicitly

        t0 = time.time()
        # Build RockPhysicsModel to centralize unit handling and EI helpers
        from src.processing.rock_physics import RockPhysicsModel

        rpm = RockPhysicsModel.from_props(props_depth, grid_spec)
        rpm.ensure_units()
        ei_multiangle_results = rpm.compute_ei_multiangle(
            [0, 5, 10, 15, 20, 25], show_progress=True
        )
        t1 = time.time()
        logger.info("Computed multi-angle EI in %.2fs", (t1 - t0))

        # compute_ei_multiangle returns a dict with keys like 'ei_stack'
        ei_depth = ei_multiangle_results.get("ei_stack")
        props_depth["ei"] = ei_depth

        # Weighted product EI
        weighted_results = rpm.compute_ei_weighted_product(
            litho_angles=[15, 10, 20, 25],
            fluid_angles=[30, 35, 25, 40],
            litho_weight=0.7,
            fluid_weight=0.3,
            show_progress=True,
        )

        props_depth["ei_litho"] = weighted_results["ei_litho"]
        props_depth["ei_fluid"] = weighted_results["ei_fluid"]
        props_depth["ei_product"] = weighted_results["ei_product"]

        # Update EI cache with weighted product
        ei_cache_file = ei_multiangle_results.get("cache_file")
        if ei_cache_file:
            try:
                ei_cache_data = dict(np.load(ei_cache_file))
                ei_cache_data["ei_litho"] = weighted_results["ei_litho"]
                ei_cache_data["ei_fluid"] = weighted_results["ei_fluid"]
                ei_cache_data["ei_product"] = weighted_results["ei_product"]
                ei_cache_data["weighted_config"] = str(weighted_results["config"])
                # use centralized cache helper
                from src.io.cache import cache_for_dir

                cache_for_dir(getattr(args, "cache_dir", ".cache")).save_npz(
                    ei_cache_file, ei_cache_data
                )
            except Exception:
                pass

        # STEP 3: DEPTH-TO-TIME
        _t0 = time.time()
        # Use DepthTimeResampler to compute TWT and resample properties
        from src.processing.resampler import resampler_factory

        resampler = resampler_factory.get_resampler(grid_spec)
        vp_for_twt = (
            props_depth["vp"].array
            if hasattr(props_depth["vp"], "array")
            else props_depth["vp"]
        )
        ni, nj, nz = vp_for_twt.shape

        # Create a shared ResamplePlan to drive all resampling in this run.
        from src.processing.resample_cache import get_resample_plan_cache

        plan_cache = get_resample_plan_cache()
        plan = plan_cache.get_plan(grid_spec, vp_for_twt, target_dt=grid_spec.dt)

        # Resample each property; resampler can accept arrays and we will wrap
        # outputs as Quantity where appropriate. Reuse the shared plan.
        props_time = {}
        for k, v in props_depth.items():
            was_q = hasattr(v, "array")
            data_arr = v.array if was_q else v
            data_time, dt = resampler.depth_to_time_cube(
                data_arr, vp_for_twt, plan=plan
            )
            props_time[k] = Quantity(data_time, v.unit) if was_q else data_time
        nt = props_time["vp"].shape[2]
        _t1 = time.time()
        logger.info("Depth->Time resampling completed in %.2fs", (_t1 - _t0))
        nx, ny, nt_samples = props_time["vp"].shape

        # STEP 4: AVO
        wavelet_avo = wavelets.ricker_wavelet(f_peak=26, dt=grid_spec.dt)
        # Use the model_cache wrappers which provide cached_* helpers
        from src.modeling import model_cache as modeling_cache

        angle_gathers, full_stack_avo = modeling_cache.cached_avo(
            props_time,
            [0, 5, 10, 15],
            wavelet_avo,
            use_quality_weighting=True,
            add_noise=getattr(args, "add_avo_noise", False),
            snr_db=20,
        )

        # STEP 5: AI
        wavelet_ai = wavelets.ricker_wavelet(f_peak=30, dt=grid_spec.dt)
        modeling_cache.cached_ai_seismogram(props_time, wavelet_ai)

        # STEP 5b: CACHE DEPTH DATA
        modeling_cache.cached_avo_depth(props_depth, [0, 5, 10, 15])
        modeling_cache.cached_ai_depth(props_depth)

        return {
            "props_depth": props_depth,
            "props_time": props_time,
            "nt": nt,
            "nx": nx,
            "ny": ny,
            "nt_samples": nt_samples,
            "ei_multiangle_results": ei_multiangle_results,
            "weighted_results": weighted_results,
        }

    @staticmethod
    def run_ei(model_outputs, args, grid_spec: GridSpec):
        """Compute EI seismograms and produce caches.

        Returns a dict with saved cache file path and related items.
        """
        props_time = model_outputs["props_time"]
        nt = model_outputs["nt"]
        nx = model_outputs["nx"]
        ny = model_outputs["ny"]
        nt_samples = model_outputs["nt_samples"]

        wavelet_ei = wavelets.ricker_wavelet(f_peak=45, dt=grid_spec.dt)
        from scipy.signal import fftconvolve

        EI_ANGLES = [0, 5, 10, 15, 20, 25]

        ei_angle_seismograms = []
        for angle_idx, angle in enumerate(EI_ANGLES):
            ei_time_angle = modeling_utils.modeling_engine.compute_ei_angle(
                props_time["vp"], props_time["vs"], props_time["rho"], angle
            )
            from src.signal.reflectivity import reflectivity_calc

            ei_refl_angle = reflectivity_calc.reflectivity_from_ai(ei_time_angle)
            # Unwrap Quantity if necessary
            ei_refl_arr = (
                ei_refl_angle.array
                if hasattr(ei_refl_angle, "array")
                else ei_refl_angle
            )
            ei_seis_angle = np.zeros((nx, ny, nt_samples))
            for i in range(nx):
                for j in range(ny):
                    trace = fftconvolve(ei_refl_arr[i, j, :], wavelet_ei, mode="same")
                    ei_seis_angle[i, j, :] = trace
            if getattr(args, "add_ei_noise", False):
                angle_seed = (
                    getattr(args, "ei_noise_seed", None)
                    if getattr(args, "ei_noise_seed", None) is not None
                    else 42
                ) + angle_idx
                from src.modeling.modeling import modeling_engine

                ei_seis_angle = modeling_engine.add_ei_noise(
                    ei_seis_angle,
                    frequency_hz=45,
                    snr_db=getattr(args, "ei_noise_snr", None),
                    include_rock_physics_error=True,
                    spatial_correlation_length=3,
                    seed=angle_seed,
                )
            ei_angle_seismograms.append(ei_seis_angle)

        # Create optimal stack (boundary-correlation weighting)
        from scipy.ndimage import sobel

        boundary_correlations = []
        for ei_seis in ei_angle_seismograms:
            grad_time = sobel(ei_seis, axis=2, mode="constant")
            boundary_quality = np.percentile(np.abs(grad_time), 90)
            boundary_correlations.append(boundary_quality)
        boundary_correlations = np.array(boundary_correlations)
        weights = boundary_correlations / boundary_correlations.sum()
        ei_optimal_stack = np.zeros_like(ei_angle_seismograms[0])
        for seis, weight in zip(ei_angle_seismograms, weights):
            ei_optimal_stack += weight * seis

        # (Removed single-seismogram implementation)
        # Code removed: pre-stacked-impedance-based single seismogram
        # generation was deprecated in favor of multi-angle stacking and the
        # variance-weighted optimal stack computed above. The reflectivity
        # object is kept for downstream consumers where needed.
        from src.signal.reflectivity import reflectivity_calc

        ei_refl = reflectivity_calc.reflectivity_from_ai(props_time["ei"])

        # STEP 7: SAVE CACHE FILES
        cache_dir = getattr(args, "cache_dir", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        noise_suffix = (
            f"_noise{getattr(args, 'ei_noise_snr', None)}db"
            if getattr(args, "add_ei_noise", False)
            and getattr(args, "ei_noise_snr", None)
            else ("_noise" if getattr(args, "add_ei_noise", False) else "")
        )
        config_str_ei = (
            f"ei_time_multiangle_45_{grid_spec.dt}_{grid_spec.dz}_"
            f"{'_'.join(map(str, grid_spec.shape))}{noise_suffix}"
        )
        config_hash_ei = hashlib.md5(config_str_ei.encode()).hexdigest()[:20]
        ei_cache_file = f"{cache_dir}/ei_time_{config_hash_ei}.npz"

        save_dict = {
            **{f"angle_{i}": seis for i, seis in enumerate(ei_angle_seismograms)},
            "optimal_stack": ei_optimal_stack,
            "ei_refl": ei_refl,
            "time_axis": nt,
            "facies": props_time["facies"],
            "config": {
                "source": "multi-angle seismograms (time-domain stacking)",
                "angles": EI_ANGLES,
                "method": "variance-weighted stack in time domain",
                "f_peak": 45,
                "dt": grid_spec.dt,
                "dz": grid_spec.dz,
                "grid_shape": grid_spec.shape,
                "noise_enabled": getattr(args, "add_ei_noise", False),
                "noise_snr_db": getattr(args, "ei_noise_snr", None),
                "noise_seed": getattr(args, "ei_noise_seed", None),
                "num_angles": len(EI_ANGLES),
            },
        }

        logger.info("✓ Saved EI to: %s", ei_cache_file)

        return {
            "ei_cache_file": ei_cache_file,
            "save_dict": save_dict,
            "ei_angle_seismograms": ei_angle_seismograms,
            "ei_optimal_stack": ei_optimal_stack,
        }

    @staticmethod
    def save_results():
        logger.info("%s", "\n" + "=" * 70)
        logger.info("SUMMARY - ALL MODELING COMPLETE")
        logger.info("%s", "=" * 70)
        logger.info("\n✓ Generated techniques: AVO, AI, EI (multi-angle)")
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

    # Delegate to the programmatic main in src.io.cache (CacheManager.main)
    try:
        from src.io.cache import cache_for_dir

        removed, size_mb = cache_for_dir(cache_dir).main(
            dry_run=dry_run, verbose=verbose
        )
    except Exception:
        # Fallback to the lower-level helper if something goes wrong
        from src.io.cache import cache_for_dir as _cache_for_dir

        removed, size_mb = _cache_for_dir(cache_dir).cleanup_old_cache(dry_run=dry_run)


# ---------------------------------------------------------------------------
# Centralized delegators for other CLI scripts
# These provide a single place to invoke top-level tools via
# `python -m src.<tool>` while keeping original module paths as shims.
# ---------------------------------------------------------------------------


# Note: analyze_facies_correlation logic moved to
# src.analysis.facies_correlation.analyze_facies_correlation


@tool
def plot_multiangle_ei(cache_dir: str = ".cache"):
    # use top-level os import
    import numpy as np
    from src.io.cache import cache_for_dir

    # Allow callers to override cache location programmatically
    # (when invoked via ParserFactory.run_tool the TOPLEVEL_ARGS will be
    # forwarded as kwargs).
    groups = cache_for_dir(cache_dir).select_latest_cache_entries()

    # find EI entries (group keys may be 'ei' or 'ei_depth' etc.)
    ei_candidates = []
    for k, v in groups.items():
        if k.startswith("ei"):
            ei_candidates.extend(v)

    if not ei_candidates:
        logger.warning("No multi-angle EI cache file found")
        return

    # pick latest by mtime
    ei_candidates_sorted = sorted(ei_candidates, key=lambda e: e.mtime)
    latest_ei_file = str(ei_candidates_sorted[-1].path)
    ei_data = np.load(latest_ei_file)

    if "facies" not in ei_data:
        logger.warning("No facies in cache file")
        return

    facies = ei_data["facies"]

    if "angles" in ei_data:
        angles = ei_data["angles"]
        ei_volumes = []
        for angle in angles:
            key = f"ei_{int(angle)}deg"
            if key in ei_data:
                ei_volumes.append(ei_data[key])

        if len(ei_volumes) == 0:
            logger.warning("No EI volumes found in cache")
            return

        ei_results = {
            "angles": angles,
            "ei_volumes": ei_volumes,
            "ei_dict": {int(a): v for a, v in zip(angles, ei_volumes)},
        }

    from src.analysis.rock_physics_attributes import (
        plot_multiangle_ei_comparison,
        plot_multiangle_ei_facies_analysis,
    )

    # Delegate to a canonical programmatic entrypoint in the analysis module
    try:
        from src.analysis.rock_physics_attributes import plot_multiangle_ei_main

        return plot_multiangle_ei_main(cache_dir=cache_dir)
    except Exception:
        # Fall back to inline behavior if the helper isn't available
        path1 = plot_multiangle_ei_comparison(ei_results, facies, cache_dir)
        path2 = plot_multiangle_ei_facies_analysis(ei_results, facies, cache_dir)

        if path1:
            logger.info(path1)
        if path2:
            logger.info(path2)

        return path1, path2


@tool
def plot_3d_interactive(argv: list | None = None):
    from src.plotting.plot_3d_interactive import main as _main

    # Forward argv if provided; otherwise rely on underlying main to parse
    # from sys.argv or use defaults.
    return _main(argv=argv)


@tool
def plot_3d_slices(argv: list | None = None):
    from src.plotting.plot_3d_slices import main as _main

    return _main(argv=argv)


@tool
def plot_rock_physics_attributes(argv: list | None = None):
    from src.plotting.plot_rock_physics_attributes import main as _main

    # The plotting main returns the canonical 7-tuple. Forward argv if given.
    result = _main(argv=argv)
    return result


@tool
def analysis_rock_physics(
    venv_python: str | None = None, cache_dir: str = ".cache", prompt: bool = True
):
    from .analysis.header import print_analysis_header
    from src.analysis import common as analysis

    long_desc = (
        "This pipeline clears caches, runs multi-angle EI, computes rock physics "
        "attributes and creates visualizations."
    )
    print_analysis_header(
        "COMPLETE ROCK PHYSICS ANALYSIS PIPELINE",
        [
            "Compute ALL Attributes + Multi-Angle EI + Generate ALL Plots",
            long_desc,
        ],
    )

    # Run the full rock-physics analysis in-process. This keeps execution
    # simpler and avoids launching subprocesses; callers expecting user
    # confirmation must perform it before calling this function.
    analysis.clear_cache()

    try:
        from src.analysis.rock_physics_attributes import (
            load_depth_data,
            run_multiangle_analysis,
            main as _rp_main,
        )

        props = load_depth_data()
        run_multiangle_analysis(props, angles_deg=[0, 5, 10, 15, 20, 25])

        _rp_main(
            cache_dir=cache_dir, ei_angle=10, generate_plots=True, save_npz_only=False
        )
    except Exception as e:
        logger.error("ERROR: Rock physics pipeline failed: %s", e)
        return False

    try:
        from src.plotting.plot_rock_physics_attributes import main as _plot_rp

        _plot_rp()
    except Exception as e:
        logger.warning("Rock physics visualization failed: %s", e)

    try:
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
    from src.analysis.facies_correlation import main as _main

    return _main(
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
    `src.analysis.seismograms.main(...)` in-process. This is the
    in-process alternative to the regenerate-pipeline subprocess step.
    """
    from src.analysis.seismograms import main as _main

    return _main(
        cache_dir=cache_dir,
        venv_python=venv_python,
        skip_cleanup=skip_cleanup,
        verbose=verbose,
    )


@tool
def analysis_seismograms():
    from src.analysis import common as analysis

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
        from src.analysis.seismograms import main as _seis_main

        _seis_main(cache_dir=".cache", skip_cleanup=True)
    except Exception as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    # Run downstream analysis tasks (facies correlation, visualizations)
    # Facies correlation (depth)
    try:
        from src.analysis.facies_correlation import main as _fac_main

        _fac_main(cache_dir=".cache", domain="depth", no_multiangle=False)
    except Exception as e:
        logger.warning("Facies depth analysis failed: %s", e)

    # Facies correlation (time)
    try:
        from src.analysis.facies_correlation import main as _fac_main_time

        _fac_main_time(cache_dir=".cache", domain="time", no_multiangle=True)
    except Exception as e:
        logger.warning("Facies time analysis failed: %s", e)

    # Interactive 3D plots (depth/time)
    try:
        from src.plotting.plot_3d_interactive import main as _plot3d

        _plot3d(argv=["--domain", "depth"])
    except Exception as e:
        logger.warning("3D interactive plot (depth) failed: %s", e)

    try:
        from src.plotting.plot_3d_interactive import main as _plot3d_time

        _plot3d_time(argv=["--domain", "time"])
    except Exception as e:
        logger.warning("3D interactive plot (time) failed: %s", e)

    return True


@tool
def regenerate_seismograms():
    from src.analysis import common as regen

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
        from src.analysis.seismograms import main as _seis_main

        _seis_main(cache_dir=".cache", skip_cleanup=True)
    except Exception as e:
        logger.error("Seismic modeling failed: %s", e)
        return False

    return True


@tool
def regenerate_rock_physics():
    from src.analysis import regenerate_common as regen

    from .analysis.header import print_analysis_header

    long_desc = (
        "This pipeline clears caches, runs multi-angle EI, computes rock physics "
        "attributes and creates visualizations."
    )
    print_analysis_header(
        "COMPLETE ROCK PHYSICS ANALYSIS PIPELINE",
        [
            "Compute ALL Attributes + Multi-Angle EI + Generate ALL Plots",
            long_desc,
        ],
    )

    regen.clear_cache()

    try:
        from src.analysis.rock_physics_attributes import (
            load_depth_data,
            run_multiangle_analysis,
            main as rp_main,
        )

        props = load_depth_data()
        run_multiangle_analysis(props, angles_deg=[0, 5, 10, 15, 20, 25])
        rp_main(
            cache_dir=".cache", ei_angle=10, generate_plots=True, save_npz_only=False
        )
    except Exception as e:
        logger.error("Rock physics regeneration failed: %s", e)
        return False

    return True


@tool
def rock_physics_attributes(
    cache_dir: str = ".cache",
    ei_angle: int = 10,
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

    # Call the canonical programmatic main in the analysis module.
    from src.analysis.rock_physics_attributes import main as rp_main

    # Ensure angles_list is a list of ints if provided as a comma string
    if isinstance(angles_list, str):
        try:
            angles_list = [int(x.strip()) for x in angles_list.split(",") if x.strip()]
        except Exception:
            raise SystemExit(
                "Invalid --angles-list format; expected comma-separated ints"
            )

    return rp_main(
        cache_dir=cache_dir,
        ei_angle=ei_angle,
        generate_plots=generate_plots,
        save_npz_only=save_npz_only,
        angles_list=angles_list,
        verbose=verbose,
    )


def main():
    # Allow forwarding arguments to a selected tool using a `--` sentinel.
    # Example:
    #   python -m src --run-tool rock_physics_attributes -- --ei-angle 15
    #   --cache-dir foo
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

    args.add_ei_noise = not args.no_ei_noise

    # Optional cleanup handled by helper
    ParserFactory.maybe_cleanup(args)

    # Use helpers to perform modeling pipeline
    props_depth, DATA_PATH, FILE_MAP, grid_spec = ParserFactory.load_data()
    model_outputs = ParserFactory.run_modeling(props_depth, args, grid_spec)
    _ei_outputs = ParserFactory.run_ei(model_outputs, args, grid_spec)
    # ensure we reference the outputs so linters don't report unused variables
    logger.debug(
        "EI outputs: %s",
        (
            list(_ei_outputs.keys())
            if isinstance(_ei_outputs, dict)
            else str(type(_ei_outputs))
        ),
    )
    ParserFactory.save_results()

    return True


if __name__ == "__main__":
    main()
