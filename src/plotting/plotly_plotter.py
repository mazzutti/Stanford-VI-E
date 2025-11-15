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

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go

from typing import Any, List, Tuple, cast

from src.plotting.helpers.base import BasePlotter

logger = logging.getLogger(__name__)

# Resource directory (optional files may be missing)
_RESOURCES_DIR = Path(__file__).parent / "resources"

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
        slice_indices: Tuple[int, int, int],
        k_scale: float = 1.0,
        colorscale: str | List[List[float | str]] = "RdBu",
        is_categorical: bool = False,
        show_colorbar: bool = True,
        title: str | None = None,
        k_label: str | None = None,
        k_unit: str | None = None,
        **kwargs: Any,
    ) -> List[go.Surface]:
        """Create three orthogonal Plotly Surface traces for a 3D `cube`.

        Returns a list: [inline_trace, crossline_trace, depth_trace].
        """
        arr = np.asarray(cube)
        if arr.size == 0:
            raise ValueError("cube cannot be empty")
        if arr.ndim != 3:
            raise ValueError(f"cube must be 3D, got shape {arr.shape}")

        ni, nj, nk = arr.shape
        inline_idx, crossline_idx, depth_idx = slice_indices

        if not (0 <= inline_idx < ni):
            raise ValueError(f"inline_idx {inline_idx} out of range [0, {ni})")
        if not (0 <= crossline_idx < nj):
            raise ValueError(f"crossline_idx {crossline_idx} out of range [0, {nj})")
        if not (0 <= depth_idx < nk):
            raise ValueError(f"depth_idx {depth_idx} out of range [0, {nk})")

        colorscale_to_use: str | List[List[float | str]]
        if is_categorical:
            colorscale_to_use = [
                [0.0, "rgb(31,119,180)"],
                [0.33, "rgb(255,127,14)"],
                [0.67, "rgb(44,160,44)"],
                [1.0, "rgb(214,39,40)"],
            ]
            cmin, cmax = 0.0, 3.0
        else:
            colorscale_to_use = colorscale
            # Allow passing a matplotlib cmap name — try to convert, but fall
            # back to the provided string on any error.
            if isinstance(colorscale, str):
                try:
                    import matplotlib as mpl
                    import matplotlib.colors as mcolors

                    def mpl_to_plotly(
                        name: str, samples: int = 256
                    ) -> List[List[float | str]]:
                        # Use the new colormap API when available but keep this
                        # code pyright-friendly by treating the registry as Any.
                        registry = cast(Any, getattr(mpl, "colormaps", None))
                        if registry is not None:
                            try:
                                cmap = registry.get_cmap(name)
                            except Exception:
                                try:
                                    cmap = registry[name]
                                except Exception:
                                    # If the registry doesn't contain the colormap,
                                    # raise to fall back to the supplied colorscale
                                    # string in the outer scope.
                                    raise
                        else:
                            # New API not available; raise so outer code falls
                            # back to using the provided colorscale string.
                            raise
                        scalars = np.linspace(0.0, 1.0, samples)
                        colors = [mcolors.to_hex(cmap(s)) for s in scalars]
                        step = 1.0 / (len(colors) - 1)
                        return [[i * step, colors[i]] for i in range(len(colors))]

                    colorscale_to_use = mpl_to_plotly(colorscale)
                except Exception:
                    colorscale_to_use = colorscale

            # robust percentile-based scaling
            slice_inline = arr[inline_idx, :, :]
            slice_crossline = arr[:, crossline_idx, :]
            slice_depth = arr[:, :, depth_idx]
            p_inline = float(np.percentile(np.abs(slice_inline), 99.5))
            p_crossline = float(np.percentile(np.abs(slice_crossline), 99.5))
            p_depth = float(np.percentile(np.abs(slice_depth), 99.5))
            vmax = float(max(p_inline, p_crossline, p_depth))
            cmax = float(vmax) if vmax != 0.0 else 1.0
            cmin = -cmax

        traces: List["go.Surface"] = []

        # coordinate arrays; cast to Any at NumPy callsites to avoid stub noise
        i_range = cast(Any, np.arange(ni))
        j_range = cast(Any, np.arange(nj))
        k_range = cast(Any, np.arange(nk) * k_scale)

        # Inline (I-K plane) — X: I, Y: constant J, Z: K
        I_inline, K_inline = np.meshgrid(i_range, k_range)
        J_inline = cast(Any, np.full_like(I_inline, float(crossline_idx), dtype=float))
        inline_data = arr[:, crossline_idx, :].T
        traces.append(
            go.Surface(
                x=cast(Any, I_inline),
                y=J_inline,
                z=cast(Any, K_inline),
                surfacecolor=cast(Any, inline_data),
                colorscale=cast(Any, colorscale_to_use),
                cmin=cmin,
                cmax=cmax,
                showscale=False,
                name=f"Inline {inline_idx}",
            )
        )

        # Crossline (J-K plane) — X: constant I, Y: J, Z: K
        J_cross, K_cross = np.meshgrid(j_range, k_range)
        I_cross = cast(Any, np.full_like(J_cross, float(inline_idx), dtype=float))
        cross_data = arr[inline_idx, :, :].T
        traces.append(
            go.Surface(
                x=I_cross,
                y=cast(Any, J_cross),
                z=cast(Any, K_cross),
                surfacecolor=cast(Any, cross_data),
                colorscale=cast(Any, colorscale_to_use),
                cmin=cmin,
                cmax=cmax,
                showscale=False,
                name=f"Crossline {crossline_idx}",
            )
        )

        # Depth (I-J plane) — X: I, Y: J, Z: constant K
        I_z, J_z = np.meshgrid(i_range, j_range)
        K_z = cast(Any, np.full_like(I_z, float(depth_idx) * k_scale, dtype=float))
        z_data = arr[:, :, depth_idx].T
        traces.append(
            go.Surface(
                x=cast(Any, I_z),
                y=cast(Any, J_z),
                z=K_z,
                surfacecolor=cast(Any, z_data),
                colorscale=cast(Any, colorscale_to_use),
                cmin=cmin,
                cmax=cmax,
                showscale=show_colorbar,
                name="Depth slice",
                colorbar=(
                    dict(
                        title=(f"Value ({k_unit})" if k_unit else "Value"),
                        thickness=20,
                        len=_COLORBAR_DEFAULT_LEN,
                    )
                    if show_colorbar
                    else None
                ),
            )
        )

        self._log_info(
            f"created 3d volume traces: indices=({inline_idx}, {crossline_idx}, {depth_idx}), cmin={cmin}, cmax={cmax}"
        )

        return traces

    def create_figure(self, traces: List["go.Surface"], title: str = "") -> "go.Figure":
        """Build a Plotly Figure from traces and apply sane layout defaults."""
        fig: "go.Figure" = go.Figure(data=traces)
        # Cast to Any for update_layout to avoid partial-member stub noise
        cast(Any, fig).update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            scene=dict(
                xaxis=dict(
                    title=dict(text="Inline (i)"), autorange="reversed", showgrid=True
                ),
                yaxis=dict(title=dict(text="Crossline (j)"), showgrid=True),
                zaxis=dict(
                    title=dict(text="Depth (k)"), autorange="reversed", showgrid=True
                ),
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def show_figure(self, fig: go.Figure) -> None:
        cast(Any, fig).show()
        self._log_info("displayed interactive figure")

    def save_figure(self, fig: go.Figure, filepath: str) -> None:
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
        except Exception:
            logger.debug("inject_3d_interaction_script failed (non-fatal)")
        self._log_info(f"saved interactive figure to {filepath}")

    @staticmethod
    def inject_3d_interaction_script(filepath: str) -> None:
        """Inject a small CSS/JS block into the HTML exported by Plotly.

        The function is defensive: it removes previously injected blocks and
        writes back only if the HTML contains a <body> tag.
        """
        with open(filepath, "r", encoding="utf-8") as f:
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
