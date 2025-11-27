"""CLI tool implementations for seismic workflows.

This module contains all tool functions registered with @tool decorator,
including cache cleanup, plotting, analysis, and regeneration workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

# Third-party imports
import matplotlib.pyplot as _plt
from matplotlib.colors import Normalize
import numpy as np
from numpy.typing import NDArray

# First-party CLI helpers
from scipy.ndimage import gaussian_filter as _gaussian_filter  # type: ignore
from src.cli.parsers import ParserFactory, tool
from src.modeling.neural_smoother import CustomDataset, CustomLoader, NeuralSmoother

from PIL import Image as _Image

# Many imports in CLI tools are intentionally performed at call-time to avoid
# heavy imports or import cycles when the CLI module is imported. Prefer
# adding a per-import suppression where that import is performed.

# The CLI performs some intentional late imports and re-exports to keep
# import-time lightweight and avoid circular dependencies. These are
# deliberate — silence import-order warnings at module level with a brief
# justification so pylint focuses on real issues.

logger = logging.getLogger(__name__)

# Shared small helpers to avoid duplication with tools_modeling
from ._tools_common import choose_html_path, save_npz  # noqa: E402

__all__ = [
    "cleanup_cache",
    "plot_3d_interactive",
    "plot_3d_slices",
    "plot_seismic_full_stack",
    "plot_rock_physics_attributes",
    "plot_original_properties",
    "analysis_rock_physics",
    "analyze_facies_correlation",
    "seismograms",
    "analysis_seismograms",
    "regenerate_seismograms",
    "regenerate_rock_physics",
    "rock_physics_attributes",
    "regenerate_all_3d_plots",
    "export_top_seismogram_layers",
    "export_top_facies_layers",
    "export_top_well_layers",
    "generate_xz_layers",
    "generate_faciesgan_dataset",
]

# Re-export plotting-related tools from the dedicated module to keep this
# file small and focused.
from .tools_plotting import (  # noqa: E402
    plot_3d_interactive,
    plot_3d_slices,
    plot_original_properties,
    plot_rock_physics_attributes,
    plot_seismic_full_stack,
    regenerate_all_3d_plots,
)


@tool
def cleanup_cache(
    cache_dir: str = ".cache", _dry_run: bool = False, verbose: bool = False
) -> tuple[int, float]:
    """Clean up old cache files (CLI tool).

    Parameters
    ----------
    cache_dir : str
        Path to cache directory
    _dry_run : bool
        If True, only report what would be cleaned
    verbose : bool
        Enable verbose logging

    Returns
    -------
    tuple[int, float]
        (number of files removed, MB freed)
    """
    ParserFactory.configure_logging(verbose)
    from src.io.pruning import (
        Pruner,
        PruneStrategy,
    )

    cache_path = Path(cache_dir)
    if cache_path.exists():
        strategy = PruneStrategy.by_size_only(max_cache_bytes=10 * 1024**3)
        pruner = Pruner(strategy)
        result = pruner.prune(cache_path)
        return result.count, result.bytes_freed / (1024**2)
    return 0, 0.0


# Modeling and analysis tools have been moved to `tools_modeling.py` to
# reduce the size of this facade module. Re-export the symbols so external
# callers can keep using the original names.
from .tools_modeling import (  # noqa: E402
    analysis_rock_physics,
    analysis_seismograms,
    analyze_facies_correlation,
    export_top_seismogram_layers,
    regenerate_seismograms,
    seismograms,
)


@tool
def export_top_facies_layers(
    cache_dir: str = ".cache",
    n_layers: int = 40 + 80,
    force_regeneration: bool = False,
    out: str | None = None,
    plot: bool = False,
    plot_out: str | None = None,
    matplotlib_only: bool = False,
) -> dict[str, str | tuple[int, int, int]]:
    """Extract the top N layers from a facies depth cache (CLI wrapper).

    Mirrors `export_top_seismogram_layers` behavior but for facies data.
    """
    ParserFactory.configure_logging(False)

    top = _get_top_layers(cache_dir, force_regeneration=force_regeneration)

    # Ensure a statically-typed 3-tuple for callers and type checkers
    if top.ndim != 3:
        raise ValueError(f"Expected facies array with 3 dimensions, got {top.shape}")
    shape_tuple: tuple[int, int, int] = (
        int(top.shape[0]),
        int(top.shape[1]),
        int(top.shape[2]),
    )

    result: dict[str, str | tuple[int, int, int]] = {"shape": shape_tuple}

    if out:
        # If the caller asked for a specific output path, save there.
        result["saved"] = str(save_npz(Path(out), facies=top))
    else:
        # If no explicit output was provided, persist a canonical copy
        # in the cache so downstream tools can find it reliably.
        try:
            cache_path = Path(cache_dir)
            cache_file = cache_path / f"facies_top_layers_{n_layers}.npz"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            saved_cache = save_npz(cache_file, facies=top)
            result["cached"] = str(saved_cache)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to save facies top layers to cache: %s", e)

    # Plotting behavior now delegates to FaciesTopLayersExtractor helpers
    if plot:

        out_dir = Path(plot_out).parent if plot_out else Path("docs/images")
        out_dir.mkdir(parents=True, exist_ok=True)

        if matplotlib_only:
            # Delegate to helper to keep top-level function small
            try:
                png = _plot_facies_matplotlib(top, n_layers, out_dir)
                if png:
                    result["png"] = str(png)
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning("Matplotlib-only facies plotting failed: %s", e)
        else:
            try:
                html = _plot_facies_plotly(top, n_layers, out_dir, out, plot_out)
                if html:
                    result["html"] = str(html)
            except (ImportError, RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to create facies interactive plot: %s", e)

    return result


def _plot_facies_matplotlib(
    top: NDArray[np.int_], n_layers: int, out_dir: Path
) -> Path | None:
    """Helper to create matplotlib PNG for top facies layers."""
    from src.plotting.helpers.components import (
        SliceExtractor,
    )
    from src.plotting.overlay_plotter import (
        OverlayPlotter,
    )

    png_path = out_dir / f"facies_top_layers_{n_layers}.png"

    fig, axes = _plt.subplots(1, 3, figsize=(15, 5))
    mid_i = top.shape[0] // 2

    # Cast shape to a fixed 3-tuple to satisfy static type checkers (`top.shape`
    # is typed as numpy._Shape which is a variable-length tuple). We ensure
    # the top array is 3D and then cast to tuple[int, int, int].
    if len(top.shape) != 3:
        raise ValueError(f"Expected facies array with 3 dimensions, got {top.shape}")
    # Explicitly construct a 3-tuple shape to satisfy static type checkers
    # (avoid unnecessary cast from numpy._Shape to tuple[int, int, int]).
    extractor_se = SliceExtractor(
        shape=(int(top.shape[0]), int(top.shape[1]), int(top.shape[2]))
    )
    op = OverlayPlotter()

    axes[0].set_title(f"Inline {mid_i}")
    # `extract_inline` expects a float-array cube; convert temporaries to float
    inline_slice = extractor_se.extract_inline(top.astype(float), mid_i)[0]
    # The plotter expects integer facies labels; convert back to int for plotting
    op.plot_facies_only(axes[0], inline_slice.astype(int))

    axes[1].set_title(f"Crossline {top.shape[1] // 2}")
    crossline_slice = extractor_se.extract_crossline(
        top.astype(float), top.shape[1] // 2
    )[0]
    op.plot_facies_only(axes[1], crossline_slice.astype(int))

    axes[2].set_title(f"Depth slice {top.shape[2] // 2}")
    depth_slice = extractor_se.extract_depthslice(top.astype(float), top.shape[2] // 2)[
        0
    ]
    op.plot_facies_only(axes[2], depth_slice.astype(int))

    _plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    _plt.close(fig)
    return png_path


def _plot_facies_plotly(
    top: NDArray[np.int_],
    n_layers: int,
    out_dir: Path,
    out: str | None,
    plot_out: str | None,
) -> Path | None:
    """Helper to create Plotly HTML for top facies layers."""
    from src.plotting.plotly_plotter import (
        PlotlyPlotter,
    )

    plotter = PlotlyPlotter()
    html_path = choose_html_path(
        plot_out, out, out_dir, f"facies_top_layers_{n_layers}_depth.html"
    )

    # Inline creation and saving to reduce temporary locals
    plotter.save_figure(
        plotter.create_figure(
            plotter.create_3d_volume(
                # create_3d_volume expects a floating ndarray; cast ints to float
                top.astype(float),
                (top.shape[0] // 2, top.shape[1] // 2, top.shape[2] // 2),
                is_categorical=True,
            ),
            title=f"Top {n_layers} layers",
        ),
        str(html_path),
    )
    return html_path


def _to_well_layers(top: NDArray[np.int_]) -> NDArray[np.int_]:
    """Convert facies top layers to well top layers.

    For each X-Z slice (crossline index j) the corresponding slice
    `top[:, j, :]` is replaced with all zeros except for a single
    randomly chosen inline index which preserves the original facies
    values along the vertical (depth) direction. This produces a
    sparse "well" trace per crossline.
    """

    # Use a numpy Generator for modern RNG behaviour and better testability
    rng = np.random.default_rng()

    # Create an output array of the same shape and dtype, filled with zeros
    wells = np.zeros_like(top)

    # Expect shape (ni, nj, nk) -> inline, crossline, depth
    if top.ndim != 3:
        raise ValueError(f"Expected 3D facies array, got shape={top.shape}")

    ni, nj, _ = top.shape

    for j in range(nj):
        i_rand = int(rng.integers(0, ni))
        # Preserve the entire vertical column at inline index i_rand
        wells[i_rand, j, :] = top[i_rand, j, :]

    return wells


@tool
def export_top_well_layers(
    cache_dir: str = ".cache",
    n_layers: int = 40 + 80,
    force_regeneration: bool = False,
    out: str | None = None,
    plot: bool = False,
    plot_out: str | None = None,
    matplotlib_only: bool = False,
) -> dict[str, str | tuple[int, int, int]]:
    """Extract the top N well-related layers from a cached dataset (CLI wrapper).

    This mirrors `export_top_facies_layers` but uses a canonical filename
    for well layers so downstream tools can distinguish outputs.
    """
    ParserFactory.configure_logging(False)

    top = _get_top_layers(cache_dir, force_regeneration=force_regeneration)

    top = _to_well_layers(top)

    # Ensure a statically-typed 3-tuple for callers and type checkers
    if top.ndim != 3:
        raise ValueError(
            f"Expected well layers array with 3 dimensions, got {top.shape}"
        )
    shape_tuple: tuple[int, int, int] = (
        int(top.shape[0]),
        int(top.shape[1]),
        int(top.shape[2]),
    )

    result: dict[str, str | tuple[int, int, int]] = {"shape": shape_tuple}

    if out:
        # If the caller asked for a specific output path, save there.
        result["saved"] = str(save_npz(Path(out), well=top))
    else:
        try:
            cache_path = Path(cache_dir)
            cache_file = cache_path / f"well_top_layers_{n_layers}.npz"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            saved_cache = save_npz(cache_file, well=top)
            result["cached"] = str(saved_cache)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to save well top layers to cache: %s", e)

    # Plotting behavior mirrors facies plotting helpers where appropriate
    if plot:
        out_dir = Path(plot_out).parent if plot_out else Path("docs/images")
        out_dir.mkdir(parents=True, exist_ok=True)

        if matplotlib_only:
            try:
                png = _plot_facies_matplotlib(top, n_layers, out_dir)
                if png:
                    result["png"] = str(png)
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning("Matplotlib-only well plotting failed: %s", e)
        else:
            try:
                html = _plot_facies_plotly(top, n_layers, out_dir, out, plot_out)
                if html:
                    result["html"] = str(html)
            except (ImportError, RuntimeError, OSError, ValueError) as e:
                logger.warning("Failed to create well interactive plot: %s", e)

    return result


@tool
def generate_faciesgan_dataset(
    input_dir: str = ".cache/images/train/",
    output_dir: str = ".cache/images/dataset/",
    steps: int = 2000,
    scale: float = 1.0,
    upsample: int = 4,
    batch_size: int = 8192,
    num_workers: int = 8,
    lr: float = 5e-4,
    scheduler: str = "onecycle",
    step_size: int = 100,
    gamma: float = 0.1,
    patience: int = 10,
    max_lr: float = 0.01,
    model_dir: str = ".cache/models/",
    force_retrain: bool = False,
    verbose: bool = False,
) -> None:
    """Generate a FaciesGAN dataset by training/rendering with the improved model.

    This CLI wrapper trains the improved Residual MLP (neural smoother) and
    writes output images (originals, rendered at multiple resolutions, loss
    plot) into `output_dir`. Returns a dict with saved paths.
    """
    ParserFactory.configure_logging(verbose)

    losses_dirs: dict[str, Path] = {}
    out_res_dirs: dict[str, Path] = {}

    resolutions = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]
    img_types: tuple[str, ...] = ("facies", "well", "seismic")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    for img_type in img_types:
        out_res_dir = out_path / img_type
        out_res_dir.mkdir(parents=True, exist_ok=True)
        out_res_dirs[img_type] = out_res_dir
        for resolution in resolutions:
            res_str = f"{resolution[0]}x{resolution[1]}"
            res_dir = out_res_dir / res_str
            res_dir.mkdir(parents=True, exist_ok=True)
        losses_dir = model_dir_path / "losses" / img_type
        losses_dir.mkdir(parents=True, exist_ok=True)
        losses_dirs[img_type] = losses_dir

    # Instantiate the NeuralSmoother with CLI-provided training params.
    ns = NeuralSmoother(
        steps=steps,
        scale=scale,
        upsample=upsample,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        scheduler_type=scheduler,
        step_size=step_size,
        gamma=gamma,
        patience=patience,
        max_lr=max_lr,
        model_dir=model_dir,
        force_retrain=force_retrain,
    )

    for img_type in img_types:
        image_list_path = Path(input_dir) / img_type
        image_list = list(image_list_path.glob("*.png"))
        for image_path in image_list:

            # Simplified CustomDataset expects a single image path: use the first image
            dataset_train = CustomDataset(str(image_path), image_type=img_type)
            loader_train = CustomLoader(
                dataset_train, batch_size=batch_size, num_workers=num_workers
            )

            num_classes = len(dataset_train.get_class_weights())
            ns.reset(num_classes=num_classes)
            losses = ns.train(loader_train)

            # For rendering we also use a single representative image (first in list)
            dataset_render = CustomDataset(str(image_path), image_type=img_type)
            loader_render = CustomLoader(
                dataset_render,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            smooth_imgs, high_res_img = ns.render(
                loader_render, resolutions=resolutions
            )

            # Save the training losses
            loss_path = losses_dirs[img_type] / f"{image_path.stem}_loss.png"
            _save_losses_plot(
                losses,
                loss_path,
                title=f"Training Loss ({image_path.name})",
            )

            # Save the high-resolution image
            arr_to_save = (high_res_img * 255).astype("uint8")
            out_highres_path = out_path / img_type / f"{image_path.stem}_highres.png"
            _Image.fromarray(arr_to_save).save(out_highres_path)

            for resolution, smooth_img in zip(resolutions, smooth_imgs):
                res_str = f"{resolution[0]}x{resolution[1]}"
                out_res_dir = out_res_dirs[img_type] / res_str
                out_res_path = out_res_dir / f"{image_path.stem}.png"
                # Normalize img_smooth_list into a Python list
                smooth_img = (smooth_img * 255).astype("uint8")
                _Image.fromarray(smooth_img).save(out_res_path)


def _get_top_layers(cache_dir: str, force_regeneration: bool) -> NDArray[np.int_]:
    """Return the top two geological facies layers as a numpy array."""
    from src.gen.facies import DefaultCacheProvider as FaciesDefaultCacheProvider
    from src.gen.facies import FaciesTopLayersExtractor

    provider = FaciesDefaultCacheProvider(cache_dir=cache_dir)
    extractor = FaciesTopLayersExtractor.from_cache_or_generate(
        cache_provider=provider,
        cache_dir=cache_dir,
        generate_if_missing=True,
        force_regeneration=force_regeneration,
    )

    return np.asarray(extractor.extract_top_two_geological_layers())


@tool
def generate_xz_layers(
    cache_dir: str = ".cache",
    n_layers: int = 120,
    output_dir: str = "images/train/",
    dpi: int = 300,
    cmap: dict[str, str | list[str]] = {
        "facies": [
            "#000000",
            "#ff0000",
            "#0000ff",
            "#00ff00",
        ],
        "well": [
            "#000000",
            "#ff0000",
            "#0000ff",
            "#00ff00",
        ],
        "seismic": "seismic",
    },
    blur_sigma: float = 1.0,
    blur_method: str = "gaussian",
    interpolation: str | None = "bilinear",
    per_facies: bool = False,
) -> dict[str, Any]:
    """Generate all X-Z (inline x depth) PNG layers from cached facies.

    Loads `.cache/facies_top_layers_{n_layers}.npz` (key: "facies") unless an
    explicit `output_dir` is provided. Writes one PNG per crossline index into a
    subdirectory and returns the output directory and number of files written.

        Parameters
        ----------
            interpolation : str | None
                Matplotlib interpolation argument passed to `imshow`. NOTE: the
                CLI now saves images by direct colormap mapping (no Axes/figure)
                so interpolation is not applied during saving. If you pass a
                non-empty interpolation value it will be accepted for CLI
                compatibility but ignored; use an explicit resampling step if
                you need bilinear/antialiased output.
        blur_method : str
            Blur method applied to the facies image; supported values are
            `'gaussian'` (default) and `'median'`. For `'gaussian'`, the radius is
            specified by `blur_sigma` (sigma). For `'median'`, `blur_sigma` is
            converted to a kernel size by `size = max(1, int(round(2*sigma + 1)))`.
    """
    ParserFactory.configure_logging(False)

    cache_path = Path(cache_dir)
    facies_file = cache_path / f"facies_top_layers_{n_layers}.npz"
    wells_file = cache_path / f"well_top_layers_{n_layers}.npz"
    seismic_file = cache_path / f"seismic_top_layers_{n_layers}.npz"

    output_path = cache_path / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    written = 0
    shape = None

    for file in [facies_file, wells_file, seismic_file]:
        if not file.exists():
            raise FileNotFoundError(f"Cache not found: {file}")

        data_type = file.stem.split("_")[0]

        # Load the saved NPZ (saved using save_npz(..., facies=top))
        data = np.load(file, allow_pickle=True)
        if data_type not in data:
            raise KeyError(
                f"Expected '{data_type}' key in {file}; found: {list(data.keys())}"
            )

        top = np.asarray(data[data_type])
        shape = top.shape

        cur_output_dir = cache_path / output_dir / data_type
        cur_output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve colormap: accept either a named cmap string or a sequence of colors
        from matplotlib.colors import ListedColormap

        if isinstance(cmap[data_type], (list, tuple)):
            cmap_obj = ListedColormap(cmap[data_type])
        else:
            cmap_obj = _plt.get_cmap(cast(str, cmap[data_type]))

        for j in range(shape[1]):
            # XZ slice for crossline index j: shape (ni, nk)
            xz: NDArray[np.int_] = top[:, j, :]

            # Optionally apply a blur to the facies array prior to rendering.
            # Two modes are supported:
            # - per_facies=False (default): blur is applied directly to the integer
            #   facies array (as a float) — preserves original behavior.
            # - per_facies=True: each unique facies label is extracted as a binary
            #   mask, the blur is applied to each mask separately, and masks are
            #   recombined by taking the label with maximum mask response per pixel.
            if blur_sigma and blur_sigma > 0.0:
                xz = _apply_blur_to_xz(xz, blur_sigma, blur_method, per_facies)

            # Map the facies array to an RGB(A) image and save directly without
            # creating a matplotlib Figure/Axes. This avoids rendering overhead
            # while producing an image equivalent to `imshow(xz.T, origin='lower')`.
            png_path = cur_output_dir / f"xz_crossline_{j:03d}.png"

            # Prepare the array the same way `imshow(xz.T, origin='lower')` would
            # present it: transpose then flip vertically to account for
            # `origin='lower'` so the saved image matches the plotted output.
            img_arr = np.flipud(xz.T)

            # Normalize values to [0,1] using the data range so the colormap
            # mapping behaves like imshow. This works for both continuous and
            # discrete/listed colormaps.
            try:
                norm = Normalize(
                    vmin=float(np.nanmin(img_arr)), vmax=float(np.nanmax(img_arr))
                )
                rgba = cmap_obj(norm(img_arr))
                # `rgba` is float [0..1], shape (H, W, 4). Convert to uint8 RGB.
                rgb = (rgba[..., :3] * 255.0).astype("uint8")

                try:
                    # Delegate saving and optional resampling to helper
                    _save_image_with_resample(rgb, png_path, dpi, interpolation)
                except Exception:
                    # Pillow may not be available in some environments; fall
                    # back to matplotlib's imsave which also accepts uint8 arrays.
                    _plt.imsave(str(png_path), rgb)
            except Exception:
                # If colormap mapping fails for any reason, fall back to saving
                # the raw integer labels as a grayscale PNG so we don't lose
                # the output entirely.
                try:
                    from PIL import Image as _Image

                    img_gray = img_arr - np.nanmin(img_arr)
                    if img_gray.max() > 0:
                        img_gray = img_gray / float(img_gray.max())
                    img_uint8 = (img_gray * 255.0).astype("uint8")
                    _Image.fromarray(img_uint8).save(png_path)
                except Exception:
                    _plt.imsave(str(png_path), img_arr)

            written += 1

    return {
        "out_dir": str(output_path),
        "written": written,
        "shape": shape,
    }


@tool
def regenerate_rock_physics() -> bool:
    """Regenerate rock physics attributes without interactive steps.

    Returns
    -------
    bool
        True if successful
    """

    from src.analysis.common import (
        AnalysisCommon,
    )
    from src.analysis.io import HeaderPrinter
    from src.analysis.rock_physics import (
        RockPhysicsAnalyzer,
    )

    regen = AnalysisCommon.instance()

    long_desc = (
        "This pipeline clears caches, computes rock physics attributes and "
        "creates visualizations."
    )
    HeaderPrinter().print_analysis_header(
        "COMPLETE ROCK PHYSICS ANALYSIS PIPELINE",
        [
            "Compute ALL Attributes + Generate ALL Plots",
            long_desc,
        ],
    )

    cast(Any, regen).clear_cache()

    try:
        rpa = RockPhysicsAnalyzer()
        rpa.run(
            cache_dir=".cache",
            generate_plots=True,
            save_npz_only=False,
            angles_list=[0, 5, 10, 15, 20, 25],
        )
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Rock physics regeneration failed: %s", e)
        return False

    return True


@tool
def rock_physics_attributes(
    cache_dir: str = ".cache",
    generate_plots: bool = True,
    save_npz_only: bool = False,
    angles_list: list[int] | str | None = None,
    verbose: bool = False,
) -> Any:
    """Programmatic entry point for rock physics attribute computation.

    Parameters
    ----------
    cache_dir : str
        Cache directory path
    generate_plots : bool
        Whether to generate visualization plots
    save_npz_only : bool
        Save only NPZ files, skip plots and ranking
    angles_list : list[int] | str | None
        Angles to use for AVO (list or comma-separated string)
    verbose : bool
        Enable verbose logging

    Returns
    -------
    Any
        Analysis results
    """
    try:
        ParserFactory.configure_logging(verbose)
    except (RuntimeError, OSError):
        pass

    from src.analysis.rock_physics import (
        RockPhysicsAnalyzer,
    )

    try:
        if isinstance(angles_list, str):
            try:
                angles_list = [
                    int(x.strip()) for x in angles_list.split(",") if x.strip()
                ]
            except (ValueError, TypeError) as exc:
                raise SystemExit(
                    "Invalid --angles-list format; expected comma-separated ints"
                ) from exc

        rpa = RockPhysicsAnalyzer()
        return rpa.run(
            cache_dir=cache_dir,
            generate_plots=generate_plots,
            save_npz_only=save_npz_only,
            angles_list=angles_list,
            verbose=verbose,
        )
    except (RuntimeError, ImportError, ValueError, OSError) as exc:
        raise SystemExit(f"Rock physics delegator unavailable: {exc}") from exc


@tool
def resample_rock_physics_to_time(
    cache_dir: str = ".cache",
    verbose: bool = False,
) -> dict[str, Any]:
    """Resample rock physics attributes from depth domain to time domain.

    This tool loads depth-domain rock physics attributes and resamples them
    to the time domain using the P-wave velocity field. The time-domain
    attributes are saved to a separate cache file for plotting.

    Parameters
    ----------
    cache_dir : str
        Cache directory path, default: .cache
    verbose : bool
        Enable verbose logging, default: False

    Returns
    -------
    dict[str, Any]
        Result dictionary with keys:
        - success: boolean indicating success
        - input_file: source depth attributes file
        - output_file: destination time attributes file
        - attributes_resampled: list of attribute names
        - error: error message if failed (optional)
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    cache_path = Path(cache_dir)

    # Load depth domain rock physics attributes
    rp_file = cache_path / "rock_physics_attributes.npz"
    if not rp_file.exists():
        error_msg = f"Depth attributes file not found: {rp_file}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    logger.info("Loading depth-domain rock physics attributes from %s", rp_file)
    rp_data = np.load(rp_file, allow_pickle=True)

    # Load Vp and prepare resampler/plan via helpers
    try:
        grid_spec, vp_depth = _load_vp_depth()
    except (RuntimeError, ValueError, OSError) as e:
        error_msg = f"Could not load Vp: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        resampler, plan = _get_resampler_and_plan(grid_spec, vp_depth)
    except (RuntimeError, ValueError, TypeError, OSError, ImportError) as e:
        error_msg = f"Could not prepare resampler: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    logger.info("Resampling rock physics attributes to time domain...")
    # Delegate attribute resampling loop to helper to reduce top-level complexity
    resampled_attrs, resampled_names = _resample_attributes(
        rp_data, resampler, vp_depth, plan
    )

    # Save to new cache file
    output_file = cache_path / "rock_physics_attributes_time.npz"
    logger.info("Saving time-domain rock physics attributes to %s", output_file)
    save_npz(output_file, **resampled_attrs)

    logger.info("✓ Done! Resampled %d attributes to time domain", len(resampled_names))

    return {
        "success": True,
        "input_file": str(rp_file),
        "output_file": str(output_file),
        "attributes_resampled": resampled_names,
    }


def _resample_attributes(
    rp_data: Any, resampler: Any, vp_depth: NDArray[Any], plan: Any
) -> tuple[dict[str, Any], list[str]]:
    """Resample listed rock-physics attributes to time domain.

    Extracted from `resample_rock_physics_to_time` to keep that function
    small and focused.
    """
    attributes_to_resample = [
        "lambda_rho",
        "mu_rho",
        "intercept",
        "gradient",
        "product",
        "scaled_gradient",
        "lambda_mu_ratio",
        "fluid_factor",
        "discrimination",
    ]

    resampled_attrs: dict[str, Any] = {}
    resampled_names: list[str] = []

    for attr_name in attributes_to_resample:
        if attr_name not in rp_data:
            continue

        attr_data = rp_data[attr_name]

        # Skip empty arrays or object arrays (discrimination)
        if (
            getattr(attr_data, "size", 0) == 0
            or getattr(attr_data, "dtype", None) is object
        ):
            logger.info("  Skipping %s (empty or object type)", attr_name)
            resampled_attrs[attr_name] = attr_data
            continue

        logger.info(
            "  Resampling %s: %s -> ... ", attr_name, getattr(attr_data, "shape", "?")
        )

        try:
            # Resample to time
            attr_time, _ = resampler.depth_to_time_cube(attr_data, vp_depth, plan=plan)
            logger.info("    → %s", getattr(attr_time, "shape", "?"))
            resampled_attrs[attr_name] = attr_time
            resampled_names.append(attr_name)
        except (RuntimeError, ValueError, IndexError) as e:
            logger.error("Failed to resample %s: %s", attr_name, e)
            continue

    return resampled_attrs, resampled_names


def _load_vp_depth() -> tuple[Any, NDArray[Any]]:
    """Load Vp property via DatasetManager and return GridSpec and Vp ndarray.

    Returns (grid_spec, vp_depth) or raises on failure.
    """
    # Lazy imports to keep module import lightweight
    from src.io.grid import GridSpec
    from src.io.loader import DatasetManager

    grid_spec = GridSpec(shape=(150, 200, 200), dz=1.0, dt=0.001)
    file_map = {"vp": "P-wave Velocity"}

    dm = DatasetManager.from_stanfordsix(".", file_map, grid_spec)
    vp_prop = dm.get_property("vp")
    if vp_prop is None:
        raise RuntimeError("Vp property not found in dataset manager")

    # Unwrap array-like wrappers (Quantity-like objects) safely
    from src.utils.quantity import to_ndarray

    vp_depth = to_ndarray(vp_prop)
    logger.info("Loaded Vp shape: %s", vp_depth.shape)
    return grid_spec, vp_depth


def _get_resampler_and_plan(grid_spec: Any, vp_depth: NDArray[Any]) -> tuple[Any, Any]:
    """Return (resampler, plan) for given grid_spec and vp_depth."""
    from src.processing.resampling._cache import (
        get_resample_plan_cache,
    )
    from src.processing.resampling._resampler import (
        resampler_factory,
    )

    resampler = resampler_factory.get_resampler(grid_spec)
    plan_cache = get_resample_plan_cache()
    plan = plan_cache.get_plan(grid_spec, vp_depth, target_dt=grid_spec.dt)
    return resampler, plan


def _apply_blur_to_xz(
    xz: NDArray[Any], blur_sigma: float, blur_method: str, per_facies: bool
) -> NDArray[Any]:
    """Apply blur to the X-Z slice according to parameters and return result.

    This centralizes the two modes supported in the CLI tool:
    - per_facies: blur each label mask independently and recombine by max response
    - global: apply gaussian or median filter to the whole array
    The function mirrors the previous inline implementation and preserves
    fallback behaviors when scipy filters fail.
    """
    if blur_sigma <= 0.0:
        return xz

    method = str(blur_method).strip().lower()

    if per_facies:
        labels = np.unique(xz)
        masks: list[NDArray[Any]] = []
        for lab in labels:
            mask = (xz == lab).astype(float)
            if method in ("gaussian", "g"):
                try:
                    mask = _gaussian_filter(mask, sigma=blur_sigma)
                except Exception:
                    try:
                        sigma = float(blur_sigma)
                        radius = max(1, int(3 * sigma))
                        x = np.arange(-radius, radius + 1)
                        kernel = np.exp(-(x**2) / (2 * sigma * sigma))
                        kernel = kernel / kernel.sum()

                        tmp = np.apply_along_axis(
                            lambda m: np.convolve(m.astype(float), kernel, mode="same"),
                            1,
                            mask,
                        )
                        mask = np.apply_along_axis(
                            lambda m: np.convolve(m.astype(float), kernel, mode="same"),
                            0,
                            tmp,
                        )
                    except Exception:
                        pass
            elif method in ("median", "m"):
                try:
                    from scipy.ndimage import median_filter as _median_filter

                    sigma = float(blur_sigma)
                    size = max(1, int(round(2.0 * sigma + 1.0)))
                    mask = _median_filter(mask, size=(size, size))
                except Exception:
                    logger.warning(
                        "Median blur failed for facies %s; skipping blur for this mask",
                        lab,
                    )
            else:
                raise ValueError(
                    f"Unknown blur method: {blur_method!r}; expected 'gaussian' or 'median'"
                )

            masks.append(mask)

        try:
            stack = np.stack(masks, axis=0)
            idx = np.argmax(stack, axis=0)
            labels_arr = np.asarray(labels)
            xz = labels_arr[idx]
        except Exception:
            logger.warning("Per-facies recombination failed; using unblurred slice")

        return xz

    # Global blur
    if method in ("gaussian", "g"):
        try:
            return _gaussian_filter(xz.astype(float), sigma=blur_sigma)
        except Exception:
            try:
                sigma = float(blur_sigma)
                radius = max(1, int(3 * sigma))
                x = np.arange(-radius, radius + 1)
                kernel = np.exp(-(x**2) / (2 * sigma * sigma))
                kernel = kernel / kernel.sum()

                tmp = np.apply_along_axis(
                    lambda m: np.convolve(m.astype(float), kernel, mode="same"),
                    1,
                    xz,
                )
                return np.apply_along_axis(
                    lambda m: np.convolve(m.astype(float), kernel, mode="same"),
                    0,
                    tmp,
                )
            except Exception:
                return xz
    elif method in ("median", "m"):
        try:
            from scipy.ndimage import median_filter as _median_filter

            sigma = float(blur_sigma)
            size = max(1, int(round(2.0 * sigma + 1.0)))
            return _median_filter(xz.astype(float), size=(size, size))
        except Exception:
            logger.warning("Median blur failed; skipping blur for this slice")
            return xz

    raise ValueError(
        f"Unknown blur method: {blur_method!r}; expected 'gaussian' or 'median'"
    )


def _save_image_with_resample(
    rgb: NDArray[np.uint8], png_path: Path, dpi: int, interpolation: str | None
) -> None:
    """Save an RGB uint8 array to ``png_path`` applying optional resampling.

    This helper resolves Pillow resampling constants in a forwards/backwards
    compatible way and falls back to ``matplotlib.imsave`` if Pillow isn't
    available or saving fails.
    """
    try:
        from PIL import Image as _Image

        # Resolve PIL resampling constants (newer Pillow exposes Image.Resampling)
        try:
            Res = getattr(_Image, "Resampling", None)
            if Res is not None:
                nearest_filter = Res.NEAREST
                bilinear_filter = Res.BILINEAR
                bicubic_filter = Res.BICUBIC
                lanczos_filter = Res.LANCZOS
            else:
                nearest_filter = getattr(_Image, "NEAREST")
                bilinear_filter = getattr(_Image, "BILINEAR")
                bicubic_filter = getattr(_Image, "BICUBIC")
                lanczos_filter = getattr(
                    _Image,
                    "LANCZOS",
                    getattr(_Image, "ANTIALIAS", getattr(_Image, "NEAREST", 0)),
                )
        except Exception:
            nearest_filter = getattr(_Image, "NEAREST", 0)
            bilinear_filter = getattr(_Image, "BILINEAR", 2)
            bicubic_filter = getattr(_Image, "BICUBIC", 3)
            lanczos_filter = getattr(_Image, "LANCZOS", 1)

        interp = None if interpolation is None else str(interpolation).strip().lower()
        if interp and interp not in ("none", "null"):
            mode = interp
            pil_filter = nearest_filter
            if mode in ("bilinear", "linear", "b"):
                pil_filter = bilinear_filter
            elif mode in ("bicubic", "c"):
                pil_filter = bicubic_filter
            elif mode in ("lanczos", "l"):
                pil_filter = lanczos_filter

            target_w = int(8 * int(dpi))
            target_h = int(6 * int(dpi))
            img_pil = _Image.fromarray(rgb)
            img_pil = img_pil.resize((target_w, target_h), resample=pil_filter)
            img_pil.save(png_path)
        else:
            _Image.fromarray(rgb).save(png_path)
    except Exception:
        # Fallback to matplotlib's imsave which accepts uint8 arrays.
        try:
            _plt.imsave(str(png_path), rgb)
        except Exception:
            logger.exception("Failed to save image to %s", png_path)


def _save_losses_plot(losses: Any, loss_path: Path, title: str | None = None) -> None:
    """Save the training `losses` sequence as a PNG at `loss_path`.

    This helper centralizes matplotlib usage and error handling so callers
    can simply pass the losses and a destination path.
    """
    try:
        fig = _plt.figure()
        _plt.plot(np.asarray(losses))
        if title:
            _plt.title(title)
        _plt.xlabel("Iteration")
        _plt.ylabel("Loss")
        fig.savefig(str(loss_path), dpi=200, bbox_inches="tight")
        _plt.close(fig)
    except Exception:
        logger.warning("Failed to save loss plot to %s", loss_path)
