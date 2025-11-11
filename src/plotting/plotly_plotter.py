"""Interactive 3D plotting with Plotly.

Provides PlotlyPlotter class for creating interactive 3D visualizations.
Replaces plot_3d_interactive.py with cleaner OOP interface.
"""

import logging
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go

from src.plotting.helpers.base import BasePlotter

logger = logging.getLogger(__name__)


class PlotlyPlotter(BasePlotter):
    """Plotter for interactive 3D visualizations using Plotly.

    Creates interactive Surface traces for visualizing seismic data
    and other 3D volumes.
    """

    def create_3d_volume(
        self,
        cube: NDArray[np.floating[Any]],
        slice_indices: Tuple[int, int, int],
        title: str = "",
        k_scale: float = 1.0,
        k_label: str = "K",
        k_unit: str = "",
        colorscale: str | List[List[float | str]] = "RdBu",
        is_categorical: bool = False,
        show_colorbar: bool = True,
    ) -> List[go.Surface]:
        """Create Plotly Surface traces for three orthogonal slices.

        Args:
            cube: 3D data array (I, J, K)
            slice_indices: Tuple of (idx_i, idx_j, idx_k)
            title: Plot title
            k_scale: Vertical scale factor
            k_label: Label for vertical axis
            k_unit: Unit for vertical axis
            colorscale: Plotly colorscale name
            is_categorical: Whether data is categorical
            show_colorbar: Whether to show colorbar

        Returns:
            List of Plotly Surface traces
        """
        arr = np.asarray(cube)
        ni, nj, nk = arr.shape
        idx_i, idx_j, idx_k = slice_indices

        # Determine color scale and limits
        cmin: float | int
        cmax: float | int
        colorscale_to_use: str | List[List[float | str]]
        if is_categorical:
            colorscale_to_use = [
                [0, "rgb(31, 119, 180)"],
                [0.33, "rgb(255, 127, 14)"],
                [0.67, "rgb(44, 160, 44)"],
                [1, "rgb(214, 39, 40)"],
            ]
            cmin = 0
            cmax = 3
        else:
            # Accept either a named Plotly colorscale or a matplotlib cmap name
            colorscale_to_use = colorscale
            if isinstance(colorscale, str):
                try:
                    # try to import matplotlib and convert named cmap to plotly list
                    import matplotlib.cm as mcm
                    import matplotlib.colors as mcolors

                    def mpl_to_plotly(cmap_name: str, samples: int = 256):
                        cmap = mcm.get_cmap(cmap_name)
                        scalars = np.linspace(0, 1, samples)
                        colors = [mcolors.to_hex(cmap(s)) for s in scalars]
                        step = 1.0 / (len(colors) - 1)
                        return [[i * step, colors[i]] for i in range(len(colors))]

                    # convert common matplotlib cmap names
                    colorscale_to_use = mpl_to_plotly(colorscale)
                except Exception:
                    colorscale_to_use = colorscale
            slice_inline = arr[idx_i, :, :]
            slice_crossline = arr[:, idx_j, :]
            slice_k = arr[:, :, idx_k]

            p_inline = np.percentile(np.abs(slice_inline), 99.5)
            p_crossline = np.percentile(np.abs(slice_crossline), 99.5)
            p_k = np.percentile(np.abs(slice_k), 99.5)

            vmax: float = max(p_inline, p_crossline, p_k)
            cmax = float(vmax)
            cmin = -cmax
            if cmax == 0:
                cmax = 1.0
                cmin = -1.0

        traces = []

        # Inline slice (constant I) - plot in X-Z plane with I on X (constant), K on Z
        # X-axis should show the full Inline range (0-ni), Y should be constant at idx_j
        j_range = np.arange(nj)
        k_range = np.arange(nk) * k_scale
        i_range = np.arange(ni)

        # For inline slice: vary I and K, keep J constant at idx_j
        I_inline, K_inline = np.meshgrid(i_range, k_range)
        J_inline = np.full_like(I_inline, idx_j, dtype=float)
        inline_data = arr[:, idx_j, :].T  # shape (nk, ni)

        trace_inline = go.Surface(
            x=I_inline,  # X-axis is Inline (I) - varies 0-150
            y=J_inline,  # Y-axis is Crossline (J) - constant at idx_j
            z=K_inline,  # Z-axis is Depth (K)
            surfacecolor=inline_data,
            colorscale=colorscale_to_use,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            name=f"Inline {idx_i}",
        )
        traces.append(trace_inline)

        # Crossline slice (constant J) - plot in Y-Z plane with J on Y (constant), K on Z
        # Y-axis should show the full Crossline range (0-nj), X should be constant at idx_i
        J_cross, K_cross = np.meshgrid(j_range, k_range)
        I_cross = np.full_like(J_cross, idx_i, dtype=float)
        cross_data = arr[idx_i, :, :].T  # shape (nk, nj)

        trace_cross = go.Surface(
            x=I_cross,  # X-axis is Inline (I) - constant at idx_i
            y=J_cross,  # Y-axis is Crossline (J) - varies 0-200
            z=K_cross,  # Z-axis is Depth (K)
            surfacecolor=cross_data,
            colorscale=colorscale_to_use,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            name=f"Crossline {idx_j}",
        )
        traces.append(trace_cross)

        # Depth slice (constant K) - plot in X-Y plane with I on X, J on Y
        # arr[:, :, idx_k] has shape (ni, nj) = (150, 200)
        I_z, J_z = np.meshgrid(i_range, j_range)  # shape (nj, ni) = (200, 150)
        K_z = np.full_like(I_z, idx_k * k_scale, dtype=float)
        z_data = arr[:, :, idx_k].T  # transpose to (nj, ni) to match meshgrid shape

        trace_z = go.Surface(
            x=I_z,  # X-axis is Inline (I) - varies 0-150
            y=J_z,  # Y-axis is Crossline (J) - varies 0-200
            z=K_z,  # Z-axis is Depth (K) - constant
            surfacecolor=z_data,
            colorscale=colorscale_to_use,
            cmin=cmin,
            cmax=cmax,
            showscale=show_colorbar,
            name=f"Depth slice",
            colorbar=(
                dict(
                    title="Value",
                    thickness=20,
                    len=0.7,
                )
                if show_colorbar
                else None
            ),
        )
        traces.append(trace_z)

        self._log_info(
            f"created 3d volume traces: indices=({idx_i}, {idx_j}, {idx_k}), "
            f"cmin={cmin}, cmax={cmax}"
        )

        return traces

    def create_figure(
        self,
        traces: List[go.Surface],
        title: str = "",
        width: int = 1000,
        height: int = 800,
    ) -> go.Figure:
        """Create an interactive Plotly figure from traces.

        Args:
            traces: List of Plotly traces
            title: Figure title
            width: Figure width in pixels (ignored; figure is responsive)
            height: Figure height in pixels (ignored; figure is responsive)

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=traces)

        # Don't set explicit width/height when autosize=True
        # This allows the figure to be truly responsive and fill available space
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            scene=dict(
                xaxis=dict(
                    title=dict(text="Inline (i)"),
                    autorange="reversed",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGrey",
                ),
                yaxis=dict(
                    title=dict(text="Crossline (j)"),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGrey",
                ),
                zaxis=dict(
                    title=dict(text="Depth (k)"),
                    autorange="reversed",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGrey",
                ),
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3), center=dict(x=0, y=0, z=0)),
            ),
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig

    def show_figure(self, fig: go.Figure) -> None:
        """Display an interactive Plotly figure.

        Args:
            fig: Plotly Figure object
        """
        fig.show()
        self._log_info("displayed interactive figure")

    def save_figure(self, fig: go.Figure, filepath: str) -> None:
        """Save an interactive Plotly figure to HTML.

        Args:
            fig: Plotly Figure object
            filepath: Output HTML file path
        """
        fig.write_html(
            filepath,
            config={
                "responsive": True,
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        # Inject interactive script for responsive 3D interaction
        # Use the centralized static method instead
        self.inject_3d_interaction_script(filepath)

        self._log_info(f"saved interactive figure to {filepath}")

    @staticmethod
    def inject_3d_interaction_script(filepath: str) -> None:
        """Static method to inject tick-preserving JavaScript into HTML files.

        This is the centralized injection method for all 3D plots to ensure
        consistent tick preservation behavior across all plot types.

        Args:
            filepath: Path to the HTML file to enhance
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()

            # JavaScript that preserves camera zoom + axis ticks + CSS for fullscreen
            script = """<style>
* {
  box-sizing: border-box !important;
}
html, body {
  height: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  display: flex !important;
  flex-direction: column !important;
}
body > * {
  flex: 1 !important;
  width: 100% !important;
  height: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}
.plotly-graph-div {
  height: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  display: block !important;
  flex: 1 !important;
}
.gl-container, .user-select-none, svg.main-svg {
  width: 100% !important;
  height: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}
#scene {
  position: absolute !important;
  left: 0 !important;
  top: 0 !important;
  width: 100% !important;
  height: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}
</style>
<script>
// Simple fullscreen setup
function setupFullscreen() {
  try {
    const div = document.querySelector('.plotly-graph-div');
    if (div) {
      div.style.height = '100%';
      div.style.width = '100%';
      div.style.margin = '0';
      div.style.padding = '0';
    }
    
    // Also ensure body and html take up full space
    const htmlEl = document.documentElement;
    const bodyEl = document.body;
    if (htmlEl) {
      htmlEl.style.height = '100%';
      htmlEl.style.width = '100%';
      htmlEl.style.margin = '0';
      htmlEl.style.padding = '0';
      htmlEl.style.overflow = 'hidden';
    }
    if (bodyEl) {
      bodyEl.style.height = '100%';
      bodyEl.style.width = '100%';
      bodyEl.style.margin = '0';
      bodyEl.style.padding = '0';
      bodyEl.style.overflow = 'hidden';
      bodyEl.style.display = 'flex';
      bodyEl.style.flexDirection = 'column';
    }
  } catch(e) { }
}

// Ensure fullscreen on load
document.addEventListener('DOMContentLoaded', setupFullscreen);
setTimeout(setupFullscreen, 100);

// Enhanced zoom control - more aggressive zoom in/out
function enableAggressiveZoom() {
  const div = document.querySelector('.plotly-graph-div');
  if (!div) return;
  
  // Store the default wheel scale - INCREASED for much more aggressive zoom
  let wheelScale = 2.5; // Much more aggressive - 2.5x sensitivity
  
  // Centralized wheel handling: zoom camera + recreate depth trace with scaled colorbar.
  // Recreating the trace (vs restyle) forces Plotly's renderer to redraw the colorbar visually.
  function performWheelZoom(e, source) {
    // Get current camera eye position
    const sceneCamera = (div && div._fullLayout && div._fullLayout.scene && div._fullLayout.scene.camera) ? div._fullLayout.scene.camera : null;
    if (sceneCamera) {
      const camera = sceneCamera;
      const eye = camera.eye;

      // Calculate zoom direction based on wheel delta - more aggressive multiplier
      const zoomFactor = e.deltaY > 0 ? (1 + (wheelScale - 1) * 0.5) : (1 - (wheelScale - 1) * 0.5);

      // New camera position (zoom towards/away from center)
      const newEye = {
        x: eye.x / zoomFactor,
        y: eye.y / zoomFactor,
        z: eye.z / zoomFactor
      };

      // Apply camera change via relayout (use promise API when available)
      try {
        const rel = Plotly.relayout(div, {'scene.camera.eye': newEye});
      } catch(e) {
        // Silently handle relayout errors
      }

      // Scale colorbar by recreating trace with new colorbar.len
      if (window._originalColorbars && Array.isArray(window._originalColorbars)) {
        window._originalColorbars.forEach(function(cb) {
          try {
            const traceIdx = cb.index;
            if (div && div.data && div.data[traceIdx]) {
              const origLen = (cb && cb.len) ? cb.len : 0.5;
              const newLen = Math.max(0.15, Math.min(0.95, origLen / zoomFactor));
              
              const currentTrace = div.data[traceIdx];
              const updatedTrace = JSON.parse(JSON.stringify(currentTrace));
              if (updatedTrace.colorbar) {
                updatedTrace.colorbar.len = newLen;
                Plotly.deleteTraces(div, [traceIdx]);
                Plotly.addTraces(div, [updatedTrace], [traceIdx]);
              }
            }
          } catch(e) {
            // Silently handle trace recreation errors
          }
        });
      }
    }
  }

  // Scale colorbar when window is resized (responsive scaling based on viewport)
  let lastWindowWidth = window.innerWidth;
  let lastWindowHeight = window.innerHeight;
  
  function scaleColorbarOnResize() {
    const currentWidth = window.innerWidth;
    const currentHeight = window.innerHeight;
    const widthRatio = currentWidth / (lastWindowWidth || currentWidth);
    const heightRatio = currentHeight / (lastWindowHeight || currentHeight);
    
    // Use geometric mean to scale colorbar proportionally
    const resizeRatio = Math.sqrt(widthRatio * heightRatio);
    
    if (window._originalColorbars && Array.isArray(window._originalColorbars)) {
      window._originalColorbars.forEach(function(cb) {
        try {
          const traceIdx = cb.index;
          if (div && div.data && div.data[traceIdx]) {
            const origLen = (cb && cb.len) ? cb.len : 0.5;
            const newLen = Math.max(0.15, Math.min(0.95, origLen * resizeRatio));
            
            // Clone and recreate trace with new colorbar length
            const currentTrace = div.data[traceIdx];
            const updatedTrace = JSON.parse(JSON.stringify(currentTrace));
            if (updatedTrace.colorbar) {
              updatedTrace.colorbar.len = newLen;
              Plotly.deleteTraces(div, [traceIdx]);
              Plotly.addTraces(div, [updatedTrace], [traceIdx]);
            }
          }
        } catch(e) {
          // Silently handle trace recreation errors
        }
      });
    }
    
    // Update last dimensions for next resize
    lastWindowWidth = currentWidth;
    lastWindowHeight = currentHeight;
  }
  
  // Throttle resize events (max once per 300ms)
  let resizeTimer = null;
  window.addEventListener('resize', function() {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(scaleColorbarOnResize, 300);
  }, false);
}

// CRITICAL: Ensure Z-axis reversed and titles persist on all layout changes
const div = document.querySelector('.plotly-graph-div');
window._isApplyingFix = false;
// When true, restoreAxisProperties will reapply captured colorbar lengths.
// Default=false to allow dynamic colorbar scaling during user interactions.
window._forceRestoreColorbars = false;
window._originalTitles = null;

if (div) {
  // Enable aggressive zoom after a short delay to let Plotly initialize
  setTimeout(enableAggressiveZoom, 1000);
  
  // Extract original titles from Plotly's internal layout
  function captureOriginalTitles() {
    try {
      // Try _fullLayout first (where Plotly stores processed layout)
      let scene = (div._fullLayout && div._fullLayout.scene) ? div._fullLayout.scene : null;
      
      // Fallback to layout.scene
      if (!scene && div.layout && div.layout.scene) {
        scene = div.layout.scene;
      }
      
      if (!scene) {
        return false;
      }
      
  const xAxis = scene.xaxis;
  const yAxis = scene.yaxis;
  const zAxis = scene.zaxis;
      
      // Extract title - could be string or object with text property
      const xTitle = xAxis && xAxis.title ? (typeof xAxis.title === 'object' ? xAxis.title.text : xAxis.title) : null;
      const yTitle = yAxis && yAxis.title ? (typeof yAxis.title === 'object' ? yAxis.title.text : yAxis.title) : null;
      const zTitle = zAxis && zAxis.title ? (typeof zAxis.title === 'object' ? zAxis.title.text : zAxis.title) : null;
      
      if (xTitle && yTitle && zTitle) {
        window._originalTitles = {
          xaxis: {text: xTitle},
          yaxis: {text: yTitle},
          zaxis: {text: zTitle}
        };
        // Capture original colorbar lengths (if present) from traces in _fullData
        try {
          const fd = div._fullData || div.data || [];
          const cbs = [];
          for (let i = 0; i < fd.length; i++) {
            const t = fd[i];
            if (t && t.colorbar) {
              // colorbar.len may be undefined; default to 0.5
              const len = t.colorbar.len || 0.5;
              cbs.push({index: i, len: len});
            }
          }
          if (cbs.length) {
            window._originalColorbars = cbs;
          }
        } catch(e) { }
        return true;
      } else {
        return false;
      }
    } catch(e) {
      return false;
    }
  }
  
  // Restore axis titles and Z-axis settings
  function restoreAxisProperties() {
    if (!window._originalTitles) {
      return;
    }
    
    const updates = {};
    
    // Restore Z-axis autorange
    updates['scene.zaxis.autorange'] = 'reversed';
    
    // Restore titles - handle both string and object formats
    if (window._originalTitles.xaxis) {
      // Plotly expects an object with .text property for titles
      if (typeof window._originalTitles.xaxis === 'object') {
        updates['scene.xaxis.title'] = window._originalTitles.xaxis;
      } else {
        updates['scene.xaxis.title'] = {text: window._originalTitles.xaxis};
      }
    }
    if (window._originalTitles.yaxis) {
      if (typeof window._originalTitles.yaxis === 'object') {
        updates['scene.yaxis.title'] = window._originalTitles.yaxis;
      } else {
        updates['scene.yaxis.title'] = {text: window._originalTitles.yaxis};
      }
    }
    if (window._originalTitles.zaxis) {
      if (typeof window._originalTitles.zaxis === 'object') {
        updates['scene.zaxis.title'] = window._originalTitles.zaxis;
      } else {
        updates['scene.zaxis.title'] = {text: window._originalTitles.zaxis};
      }
    }
    
    // Automatic colorbar restoration intentionally disabled.
    // Restoring colorbar lengths here would interfere with user-driven
    // dynamic scaling during wheel zoom or window resize.

    // Apply other layout updates (titles, z autorange)
    Plotly.relayout(div, updates);
  }
  
  // Try to capture titles immediately
  setTimeout(function() {
    if (!captureOriginalTitles()) {
      // Retry a few times if not ready
      let attempts = 0;
      const retryInterval = setInterval(function() {
        if (captureOriginalTitles() || attempts++ > 5) {
          clearInterval(retryInterval);
        }
      }, 100);
    }
  }, 500);
  
  div.on('plotly_relayout', function(data) {
    try {
      // Try to capture titles if we haven't yet
      if (!window._originalTitles) {
        captureOriginalTitles();
      }
      
      // PREVENT INFINITE LOOP
      if (window._isApplyingFix) {
        window._isApplyingFix = false;
        return;
      }
      
      const scene = div.layout ? div.layout.scene : null;
      if (!scene) {
        return;
      }
      
      // Check if properties need fixing
      let needsFix = false;
      
      // Check Z-axis autorange
      const zAutorange = scene.zaxis ? scene.zaxis.autorange : null;
      if (zAutorange !== 'reversed') {
        needsFix = true;
      }
      
      // Check axis titles
      const xTitle = scene.xaxis && scene.xaxis.title ? scene.xaxis.title : null;
      const yTitle = scene.yaxis && scene.yaxis.title ? scene.yaxis.title : null;
      const zTitle = scene.zaxis && scene.zaxis.title ? scene.zaxis.title : null;
      
      if ((xTitle === 'X' || xTitle === null) || 
          (yTitle === 'Y' || yTitle === null) || 
          (zTitle === 'Z' || zTitle === null)) {
        needsFix = true;
      }
      
      if (needsFix) {
        window._isApplyingFix = true;
        setTimeout(function() {
          restoreAxisProperties();
          setTimeout(function() { 
            window._isApplyingFix = false;
          }, 50);
        }, 10);
      }
    } catch(e) { 
      window._isApplyingFix = false;
    }
  });
}
</script>
            """

            # Insert script before </body>.
            # First remove any previously injected script block to avoid duplicates
            # (older runs appended the same script repeatedly).
            start_marker = "<!-- BEGIN_PLOTLY_3D_INJECTION -->"
            end_marker = "<!-- END_PLOTLY_3D_INJECTION -->"
            wrapped = start_marker + "\n" + script + "\n" + end_marker

            try:
                if start_marker in html_content and end_marker in html_content:
                    s = html_content.index(start_marker)
                    e = html_content.index(end_marker, s) + len(end_marker)
                    html_content = html_content[:s] + html_content[e:]
            except Exception:
                # If something goes wrong, fall back to a safer replace of markers
                html_content = html_content.replace(start_marker, "").replace(
                    end_marker, ""
                )

            if "</body>" in html_content:
                html_content = html_content.replace("</body>", wrapped + "\n</body>")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html_content)
        except Exception as e:
            logger.error(f"Error injecting 3D interaction script: {e}")
