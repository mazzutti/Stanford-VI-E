"""Interactive 3D plotting with Plotly.

This module provides `PlotlyPlotter` — a small, conservative wrapper around
Plotly that creates three orthogonal surface traces for a 3D volume and
produces an interactive HTML output. The implementation uses local
`typing.cast(Any, ...)` only at third-party callsites to reduce type-checker
noise while preserving runtime behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from src.plotting.helpers.base import BasePlotter
from src.plotting.helpers.colors import compute_plotly_colorscale_and_bounds
from src.plotting.helpers.configs import TraceConfig
from src.plotting.helpers.traces import make_plotly_surface_traces_from_config

logger = logging.getLogger(__name__)

# Resource directory (optional files may be missing)
_RESOURCES_DIR = Path(__file__).parent / "resources"

# Many plotting helpers use short names and occasionally accept many
# small parameters for caller convenience. Silence a small set of
# convention/refactor messages that are noise for UI glue code.


# Interaction constants (only defaults used by injected JS)
_WHEEL_ZOOM_SENSITIVITY = 2.5
_COLORBAR_DEFAULT_LEN = 0.7
_RESIZE_THROTTLE_MS = 300
_RETRY_ATTEMPTS = 5


class PlotlyPlotter(BasePlotter):
    """Small wrapper to build Plotly 3D figures for volume data.

    Methods are intentionally conservative about typing at Plotly/NumPy
    interaction points to avoid noisy diagnostics from incomplete stubs.
    """

    def create_3d_volume(
        self,
        cube: NDArray[np.floating[Any]],
        slice_indices: tuple[int, int, int],
        k_scale: float = 1.0,
        colorscale: str | list[list[float | str]] = "RdBu",
        is_categorical: bool = False,
        show_colorbar: bool = True,
        title: str | None = None,
        k_label: str | None = None,
        k_unit: str | None = None,
        # intentionally do not accept arbitrary kwargs — unused by implementation
    ) -> list[go.Surface]:
        """Create three orthogonal Plotly Surface traces for a 3D `cube`.

        Returns a list: [inline_trace, crossline_trace, depth_trace].
        """
        arr = np.asarray(cube)
        # Mark intentionally-unused public params to satisfy linters
        # (callers may still pass them; we don't use them here).
        del title, k_label

        dims = self._validate_cube_and_indices(arr, slice_indices)

        colors = compute_plotly_colorscale_and_bounds(
            arr, (dims[3], dims[4], dims[5]), colorscale, is_categorical
        )

        traces = self._create_traces(
            arr,
            *dims,
            k_scale,
            colors[0],
            colors[1],
            colors[2],
            show_colorbar,
            k_unit,
        )

        # Use logger directly with lazy interpolation to avoid building
        # large f-strings for diagnostic messages.
        logger.info(
            "created 3d volume traces: indices=(%d, %d, %d), cmin=%s, cmax=%s",
            dims[3],
            dims[4],
            dims[5],
            colors[1],
            colors[2],
        )

        return traces

    def _validate_cube_and_indices(
        self, arr: NDArray[np.floating[Any]], slice_indices: tuple[int, int, int]
    ) -> tuple[int, int, int, int, int, int]:
        """Validate the cube shape and slice indices and return dimensions.

        Returns (ni, nj, nk, inline_idx, crossline_idx, depth_idx).
        """
        if arr.size == 0:
            raise ValueError("cube cannot be empty")
        if arr.ndim != 3:
            raise ValueError(f"cube must be 3D, got shape {arr.shape}")

        ni, nj, nk = arr.shape
        inline_idx, crossline_idx, depth_idx = slice_indices

        if not 0 <= inline_idx < ni:
            raise ValueError(f"inline_idx {inline_idx} out of range [0, {ni})")
        if not 0 <= crossline_idx < nj:
            raise ValueError(f"crossline_idx {crossline_idx} out of range [0, {nj})")
        if not 0 <= depth_idx < nk:
            raise ValueError(f"depth_idx {depth_idx} out of range [0, {nk})")

        return ni, nj, nk, inline_idx, crossline_idx, depth_idx

    def _compute_colorscale_and_bounds(
        self,
        arr: NDArray[np.floating[Any]],
        inline_idx: int,
        crossline_idx: int,
        depth_idx: int,
        colorscale: str | list[list[float | str]],
        is_categorical: bool,
    ) -> tuple[str | list[list[float | str]], float, float]:
        """Compute the Plotly colorscale object and the cmin/cmax bounds."""
        # This logic is now provided by the shared helper
        # `src.plotting.helpers.colors.compute_plotly_colorscale_and_bounds`.
        # Keep this private method name to avoid breaking callers that may expect it,
        # but delegate to the shared helper for maintainability.
        return compute_plotly_colorscale_and_bounds(
            arr, (inline_idx, crossline_idx, depth_idx), colorscale, is_categorical
        )

    def _create_traces(
        self,
        arr: NDArray[np.floating[Any]],
        ni: int,
        nj: int,
        nk: int,
        inline_idx: int,
        crossline_idx: int,
        depth_idx: int,
        k_scale: float,
        colorscale_to_use: str | list[list[float | str]],
        cmin: float,
        cmax: float,
        show_colorbar: bool,
        k_unit: str | None,
    ) -> list[go.Surface]:
        """Build the three Plotly Surface traces for the requested slices."""
        # Delegate trace construction to shared helper
        # Build a small config object to reduce the number of arguments
        # passed around and improve readability.
        cfg = TraceConfig(
            k_scale=k_scale,
            colorscale_to_use=colorscale_to_use,
            cmin=cmin,
            cmax=cmax,
            show_colorbar=show_colorbar,
            k_unit=k_unit,
            colorbar_len=_COLORBAR_DEFAULT_LEN,
        )

        # These dimensional arguments are provided for callers convenience
        # but are not directly used by the trace factory helper. Mark them
        # as intentionally unused to satisfy linters.
        del ni, nj, nk

        return make_plotly_surface_traces_from_config(
            arr, inline_idx, crossline_idx, depth_idx, cfg
        )

    def create_figure(self, traces: list[go.Surface], title: str = "") -> go.Figure:
        """Build a Plotly Figure from traces and apply sane layout defaults."""
        fig: go.Figure = go.Figure(data=traces)
        # Cast to Any for update_layout to avoid partial-member stub noise
        cast(Any, fig).update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            scene={
                "xaxis": {
                    "title": {"text": "Inline (i)"},
                    "autorange": "reversed",
                    "showgrid": True,
                },
                "yaxis": {"title": {"text": "Crossline (j)"}, "showgrid": True},
                "zaxis": {
                    "title": {"text": "Depth (k)"},
                    "autorange": "reversed",
                    "showgrid": True,
                },
                "aspectmode": "data",
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.3}},
            },
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def show_figure(self, fig: go.Figure) -> None:
        """Display the interactive Plotly figure in a GUI/Notebook.

        This wraps `fig.show()` and logs the display action.
        """
        cast(Any, fig).show()
        self._log_info("displayed interactive figure")

    def save_figure(self, fig: go.Figure, filepath: str) -> None:
        """Save the Plotly figure as an HTML file and inject interaction JS.

        Writes the figure to `filepath` using Plotly's `write_html`, then
        attempts to inject optional 3D interaction scripts from the package
        resources. Failures to inject are non-fatal and logged.
        """

        cast(Any, fig).write_html(
            filepath,
            config={
                "responsive": True,
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
        # Try to inject the optional 3D interaction script; failure is non-fatal
        try:
            self.inject_3d_interaction_script(filepath)
        except OSError:
            logger.debug("inject_3d_interaction_script failed (non-fatal)")
        self._log_info(f"saved interactive figure to {filepath}")

    @staticmethod
    def inject_3d_interaction_script(filepath: str) -> None:
        """Inject a small CSS/JS block into the HTML exported by Plotly.

        The function is defensive: it removes previously injected blocks and
        writes back only if the HTML contains a <body> tag.
        """
        with open(filepath, encoding="utf-8") as f:
            html = f.read()

        start_marker = "<!-- BEGIN_PLOTLY_3D_INJECTION -->"
        end_marker = "<!-- END_PLOTLY_3D_INJECTION -->"
        script = PlotlyPlotter._get_3d_interaction_script()
        wrapped = start_marker + "\n" + script + "\n" + end_marker

        # remove existing block if present
        if start_marker in html and end_marker in html:
            s = html.index(start_marker)
            e = html.index(end_marker, s) + len(end_marker)
            html = html[:s] + html[e:]

        if "</body>" in html:
            html = html.replace("</body>", wrapped + "\n</body>")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)

    @staticmethod
    def _get_3d_interaction_script() -> str:
        """Return a simple <style>+<script> block loaded from resources (if any)."""
        css_file = _RESOURCES_DIR / "plotly_3d_styles.css"
        js_file = _RESOURCES_DIR / "plotly_3d_interaction.js"
        css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""
        js = js_file.read_text(encoding="utf-8") if js_file.exists() else ""
        # substitute a few known tokens used by the JS (defensive)
        js = js.replace("__WHEEL_ZOOM_SENSITIVITY", str(_WHEEL_ZOOM_SENSITIVITY))
        js = js.replace("__COLORBAR_DEFAULT_LEN", str(_COLORBAR_DEFAULT_LEN))
        js = js.replace("__RESIZE_THROTTLE_MS", str(int(_RESIZE_THROTTLE_MS)))
        js = js.replace("__RETRY_ATTEMPTS", str(int(_RETRY_ATTEMPTS)))
        return f"""<style>\n{css}\n</style>\n<script>\n{js}\n</script>"""
